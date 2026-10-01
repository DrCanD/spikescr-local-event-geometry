"""Forward, trace and replacement kernels from the recorded experiment.

No alternative spiking model or batched candidate implementation is used.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

SJ_FUNCTIONAL = None

def reset_spiking_state(model):
    if SJ_FUNCTIONAL is None:
        raise RuntimeError("Spiking state reset has not been configured")
    SJ_FUNCTIONAL.reset_net(model)

def attention_mask_from_lengths(lengths: Tensor, time_steps: int) -> Tensor:
    return torch.arange(time_steps, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

def upstream_softmax_sum(output: Tensor) -> Tensor:
    if output.ndim != 3:
        raise RuntimeError(f"Expected SpikeSCR output [T,B,C], got {tuple(output.shape)}")
    return torch.softmax(output.float(), dim=2).sum(dim=0)

def trace_output_tensor(output: Any, stage_name: str) -> Tensor:
    if not isinstance(output, Tensor):
        raise RuntimeError(
            f"Trace stage {stage_name} returned {type(output).__name__}; "
            "the frozen architecture contract requires a Tensor boundary"
        )
    if output.ndim < 2:
        raise RuntimeError(f"Trace stage {stage_name} has invalid shape {tuple(output.shape)}")
    return output

def captured_model_forward(
    model: nn.Module,
    batch: Tensor,
    device: torch.device,
    stages: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Tensor]]:
    if batch.ndim != 3:
        raise ValueError("Expected [B,T,F] batch for internal tracing")
    captured: dict[str, Tensor] = {}
    handles: list[Any] = []

    def make_hook(stage_name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if stage_name in captured:
                raise RuntimeError(f"Trace stage executed more than once: {stage_name}")
            captured[stage_name] = trace_output_tensor(output, stage_name).detach()
        return hook

    reset_spiking_state(model)
    try:
        for stage in stages:
            handles.append(stage["module"].register_forward_hook(make_hook(stage["stage"])))
        batch_device = batch.to(device, non_blocking=True)
        lengths = torch.full(
            (batch_device.shape[0],), batch_device.shape[1], dtype=torch.long, device=device
        )
        with torch.no_grad():
            output = model(
                batch_device,
                attention_mask_from_lengths(lengths, int(batch_device.shape[1])),
            )
            scores = upstream_softmax_sum(output)
        if set(captured) != {stage["stage"] for stage in stages}:
            raise RuntimeError(f"Incomplete internal trace: {sorted(captured)}")
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite score during internal tracing")
        return scores.detach().cpu().numpy().astype(np.float32), captured
    finally:
        for handle in handles:
            handle.remove()
        reset_spiking_state(model)

def infer_trace_batch_axes(
    traces: dict[str, Tensor], batch_size: int
) -> dict[str, int]:
    axes: dict[str, int] = {}
    for name, tensor in traces.items():
        candidates = [axis for axis, size in enumerate(tensor.shape) if int(size) == int(batch_size)]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Cannot identify unique batch axis for {name}: shape={tuple(tensor.shape)} "
                f"batch={batch_size} candidates={candidates}"
            )
        axes[name] = int(candidates[0])
    return axes

def clean_trace_replacement(clean: Tensor, output: Tensor, batch_axis: int) -> Tensor:
    if output.ndim != clean.ndim:
        raise RuntimeError("Patch clean/candidate ranks differ")
    if int(clean.shape[batch_axis]) != 1:
        raise RuntimeError("Clean patch reference does not have singleton batch axis")
    target_shape = list(output.shape)
    for axis, (clean_size, output_size) in enumerate(zip(clean.shape, output.shape)):
        if axis == batch_axis:
            continue
        if int(clean_size) != int(output_size):
            raise RuntimeError(
                f"Patch shape disagreement at axis {axis}: clean={tuple(clean.shape)} "
                f"candidate={tuple(output.shape)}"
            )
    return clean.to(device=output.device, dtype=output.dtype).expand(*target_shape)

def patched_model_scores(
    model: nn.Module,
    batch: Tensor,
    device: torch.device,
    stage: dict[str, Any],
    clean_trace: Tensor,
    batch_axis: int,
) -> np.ndarray:
    def patch_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Tensor:
        tensor = trace_output_tensor(output, stage["stage"])
        return clean_trace_replacement(clean_trace, tensor, batch_axis)

    reset_spiking_state(model)
    handle = stage["module"].register_forward_hook(patch_hook)
    try:
        batch_device = batch.to(device, non_blocking=True)
        lengths = torch.full(
            (batch_device.shape[0],), batch_device.shape[1], dtype=torch.long, device=device
        )
        with torch.no_grad():
            output = model(
                batch_device,
                attention_mask_from_lengths(lengths, int(batch_device.shape[1])),
            )
            scores = upstream_softmax_sum(output)
        if not torch.isfinite(scores).all():
            raise FloatingPointError(f"Non-finite patched scores at {stage['stage']}")
        return scores.detach().cpu().numpy().astype(np.float32)
    finally:
        handle.remove()
        reset_spiking_state(model)

def singleton_trace_metrics(candidate: Tensor, clean: Tensor) -> np.ndarray:
    if tuple(candidate.shape) != tuple(clean.shape):
        raise RuntimeError(
            f"Singleton trace shape mismatch: {tuple(candidate.shape)} / {tuple(clean.shape)}"
        )
    candidate_flat = candidate.float().reshape(1, -1)
    clean_flat = clean.float().reshape(1, -1)
    delta = candidate_flat - clean_flat
    metrics = torch.stack(
        [
            delta.abs().mean(dim=1),
            torch.sqrt(torch.mean(delta * delta, dim=1)),
            torch.linalg.vector_norm(delta, dim=1)
            / torch.clamp(torch.linalg.vector_norm(clean_flat, dim=1), min=1.0e-12),
            1.0 - F.cosine_similarity(candidate_flat, clean_flat, dim=1, eps=1.0e-12),
            ((candidate_flat > 0.5) != (clean_flat > 0.5)).float().mean(dim=1),
            delta.mean(dim=1),
        ],
        dim=1,
    )
    if not torch.isfinite(metrics).all():
        raise FloatingPointError("Non-finite exact-singleton trace metric")
    return metrics[0].detach().cpu().numpy().astype(np.float32)
