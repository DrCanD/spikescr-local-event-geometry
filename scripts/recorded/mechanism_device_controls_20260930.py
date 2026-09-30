#!/usr/bin/env python3
"""Validation-only padding, q/k layout, and paired CPU/GPU controls.

Uses existing Drive inputs. No training or official-test access.
Original weights and source files are immutable. q/k intervention is diagnostic
and is disabled for the candidate replay. Partial results resume by contract.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
import zipfile
from pathlib import Path

import numpy as np

VERSION = "ssc-mechanism-device-controls-20260930-v1"
PROJECT = Path("/content/drive/MyDrive/Research/SIPT_SSC")
CONTROL_REL = Path("experiments/execution_controls_20260930")
LAUNCHER_NAME = "SSC_GPU_Execution_Control_20260930.py"
LAUNCHER_SHA = "baed6fd8c53f2549e69ee8d8043b011931d87f0e7fc8d5cee6566eec5a293c2c"
HISTORY_REL = CONTROL_REL / "results/gpu_9d747669fffeabdb"
HISTORY_CONTRACT = "0ec56d8ca12ea9395a7e9c0c3103b87cf6d742c7148cd080eae062ffb857d0f5"
AUDIT_REL = Path("experiments/eb18_spikescr_exact_b1_geometry/seed312_epoch282_validation_exact_b1_v1")
GEOMETRY_NAME = "EB18_EXACT_B1_CANDIDATE_GEOMETRY.npz"
GEOMETRY_SHA = "56dc58f2c99b54ac0ef413d6609ea94024b86f1a5f591ce101d56fe323b7f222"
EXPECTED_TRANSITIONS = np.array([699250, 5157, 9031, 11632])
PAD_BINS = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 100000)]


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cpu_model_name():
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unavailable"


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def read_npz(path, contract=None):
    with np.load(path, allow_pickle=False) as z:
        if contract is not None and str(z["contract_sha256"]) != contract:
            raise ValueError("Saved results have a different contract: " + str(path))
        return {name: z[name].copy() for name in z.files}


def duration(seconds):
    if seconds is None:
        return "hesaplaniyor"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Progress:
    """Heartbeat starts before imports/staging; no silent long-running phase."""
    def __init__(self, interval=20):
        self.interval = interval
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.event = threading.Event()
        self.name = "Baslatiliyor"
        self.done = self.total = self.origin_done = 0
        self.phase_started = self.started
        self.last_emit = 0
        self.output = None
        self.log = None
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _heartbeat(self):
        while not self.event.wait(self.interval):
            self.emit()

    def phase(self, name, total=0, done=0):
        with self.lock:
            self.name = name
            self.done = self.origin_done = int(done)
            self.total = int(total)
            self.phase_started = time.monotonic()
        self.emit()

    def update(self, done, force=False):
        with self.lock:
            self.done = int(done)
        if force or time.monotonic() - self.last_emit >= 5:
            self.emit()

    def attach(self, output):
        self.output = Path(output)
        self.log = self.output / ("run_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ".log")

    def message(self, message):
        with self.lock:
            print(message, flush=True)
            if self.log:
                with self.log.open("a", encoding="utf-8") as f:
                    f.write(str(message) + "\n")

    def emit(self):
        with self.lock:
            now = time.monotonic()
            advanced = self.done - self.origin_done
            eta = (now - self.phase_started) * (self.total - self.done) / advanced if self.total and advanced > 0 else None
            pct = 100 * self.done / self.total if self.total else None
            count = f"{self.done:,}/{self.total:,} (%{pct:.1f})" if self.total else "islem suruyor"
            self.message(f"[{time.strftime('%H:%M:%S')}] {self.name} | {count} | gecen {duration(now-self.started)} | asama kalan {duration(eta)}")
            self.last_emit = now
            if self.output:
                atomic_json(self.output / "progress.json", {
                    "phase": self.name, "done": self.done, "total": self.total,
                    "elapsed_seconds": now - self.started, "phase_eta_seconds": eta,
                    "complete": False, "official_test_access": False,
                })

    def close(self):
        self.event.set()
        self.thread.join(timeout=2)


def transition(label, clean, candidate):
    label, clean, candidate = np.broadcast_arrays(label, clean, candidate)
    out = np.zeros(label.shape, dtype=np.uint8)
    change = clean != candidate
    out[change & (clean == label)] = 1
    out[change & (clean != label) & (candidate == label)] = 2
    out[change & (clean != label) & (candidate != label)] = 3
    return out


def distribution(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"samples": 0}
    return {"samples": int(len(x)), "exact_zero": int(np.sum(x == 0)),
            "nonzero": int(np.sum(x != 0)), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)), "p99": float(np.quantile(x, .99)),
            "maximum": float(np.max(x)),
            "above_1e-5": int(np.sum(x > 1e-5)),
            "above_1e-4": int(np.sum(x > 1e-4)),
            "above_1e-3": int(np.sum(x > 1e-3))}


def record_analyses(history, cpu, output):
    """A/C: existing arrays only; no model inference and no causal conclusion."""
    ids = history["source_index"]
    for key, other in [("source_index", "source_index"), ("label", "label"),
                       ("native_horizon", "horizon"), ("input_sha256", "input_sha256")]:
        if not np.array_equal(history[key], cpu[other]):
            raise ValueError("Historical CPU/GPU input mismatch: " + key)
    if not np.array_equal(ids, np.arange(9981)):
        raise ValueError("Expected all 9,981 source identities")
    gap = history["padded_horizon"] - history["native_horizon"]
    if np.any(gap < 0):
        raise ValueError("Negative padding")
    flip = history["B1_native_prediction"] != history["B1_padding_matched_prediction"]
    native = history["B1_native_scores"].astype(np.float64)
    ordered = np.sort(native, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    score_max = np.max(np.abs(cpu["scores"].astype(float) - native), axis=1)
    if int(flip.sum()) != 344 or int(np.sum(cpu["prediction"] != history["B1_native_prediction"])) != 2:
        raise ValueError("Historical paired counts do not reproduce the completed run")
    if np.any(flip & (gap == 0)):
        raise ValueError("Padding contrast changes a source receiving no padding")
    rows = []
    for lo, hi in PAD_BINS:
        use = (gap >= lo) & (gap <= hi)
        n, f = int(use.sum()), int(flip[use].sum())
        rows.append({"padding_bins_min": lo, "padding_bins_max": hi,
                     "sources": n, "changed_labels": f, "changed_fraction": f/n if n else 0.0})
    write_csv(output / "A_padding_by_gap.csv", rows)
    write_csv(output / "A_C_per_source.csv", [
        {"source_index": int(ids[j]), "label": int(history["label"][j]),
         "native_horizon": int(history["native_horizon"][j]),
         "padded_horizon": int(history["padded_horizon"][j]), "padding_bins": int(gap[j]),
         "native_margin": float(margin[j]),
         "native_margin_per_bin": float(margin[j]/history["native_horizon"][j]),
         "padding_label_changed": int(flip[j]), "CPU_GPU_score_max_abs": float(score_max[j]),
         "CPU_GPU_label_changed": int(cpu["prediction"][j] != history["B1_native_prediction"][j])}
        for j in range(len(ids))])
    summary = {"padding_label_changes": int(flip.sum()), "padding_bins": rows,
               "CPU_GPU_source_max_abs_score_difference": distribution(score_max),
               "CPU_GPU_score_difference_per_native_bin": distribution(score_max/history["native_horizon"]),
               "CPU_GPU_label_changes": 2,
               "interpretation": "Historical execution comparison. Gap association is descriptive; padded-prefix and extra-output effects are diagnosed separately in new GPU forwards."}
    atomic_json(output / "A_C_recorded_summary.json", summary)
    return summary


def layout_owners(T, B, N):
    """Source identity carried by each LIF membrane lane after the raw reshape.

    A contiguous [B, heads, T, D] tensor is interpreted as [T, B, N].
    The actual source owning lane (b,n) can change along its declared time axis.
    No spike values or predictions enter this structural calculation.
    """
    offset = np.arange(B*N, dtype=np.int64)
    previous = offset // (T*N)
    changes = np.zeros(B*N, dtype=np.int32)
    for t in range(1, T):
        owner = (t*B*N + offset) // (T*N)
        changes += owner != previous
        previous = owner
    counts = changes + 1
    return {"T": T, "B": B, "N": N, "membrane_lanes": B*N,
            "original_sources_per_lane_min": int(counts.min()),
            "original_sources_per_lane_max": int(counts.max()),
            "fraction_lanes_containing_multiple_sources": float(np.mean(counts > 1)),
            "intervened_logical_sources_per_lane": 1}


@contextlib.contextmanager
def qk_intervention(model, which="qk"):
    """Isolate logical sources at q_lif2/k_lif2, preserving the public B1 map.

    This is NOT a replacement of the model's canonical temporal convention.
    Each source keeps exactly its original singleton head/time flattening.
    Only batch mixing at the two declared LIF boundaries is removed.
    """
    saved = []
    try:
        for module_name, attention in model.named_modules():
            if attention.__class__.__name__ != "MS_SSA":
                continue
            heads = int(attention.num_heads)
            for branch in ("q", "k"):
                if branch not in which:
                    continue
                node = getattr(attention, branch + "_lif2")
                original = node.forward
                def isolated(self, x, original=original, heads=heads):
                    if x.ndim != 3:
                        raise ValueError("Unexpected q/k LIF input rank")
                    T, B, N = x.shape
                    if B == 1:
                        return original(x)
                    if N % heads:
                        raise ValueError("Head dimension mismatch")
                    # Undo raw reshape, form independent singleton-convention
                    # sequences, and return the caller's expected raw layout.
                    independent = x.reshape(B, heads, T, N//heads).reshape(B, T, N).permute(1, 0, 2).contiguous()
                    y = original(independent)
                    return y.permute(1, 0, 2).contiguous().reshape(B, heads, T, N//heads).reshape(T, B, N)
                saved.append((node, original))
                node.forward = types.MethodType(isolated, node)
        if len(saved) != (4 if which == "qk" else 2):
            raise ValueError("Expected two attention blocks and their declared q/k nodes")
        yield
    finally:
        for node, original in saved:
            node.forward = original


def load_runner(root):
    spec = importlib.util.spec_from_file_location("ssc_frozen_runner", root / "model_runner/run_singleton_gradient_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(project, work, progress):
    old = project / CONTROL_REL / LAUNCHER_NAME
    if not old.is_file() or sha_file(old) != LAUNCHER_SHA:
        raise ValueError("Executed Drive launcher is missing or changed: " + str(old))
    parsed = ast.parse(old.read_text())
    const = {n.targets[0].id: ast.literal_eval(n.value) for n in parsed.body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
             and isinstance(n.value, (ast.Constant, ast.List, ast.Tuple, ast.Dict))}
    support = base64.b64decode(const["SUPPORT_B64"])
    if hashlib.sha256(support).hexdigest() != const["SUPPORT_SHA"]:
        raise ValueError("Embedded support hash changed")
    root = work / "frozen"
    root.mkdir(parents=True, exist_ok=True)
    progress.phase("1/7 — Kayitli kaynak kodu ve referanslar aciliyor")
    with zipfile.ZipFile(io.BytesIO(support)) as z:
        for name in z.namelist():
            p = Path(name)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Unsafe support path")
        z.extractall(root)
    inventory = json.loads((root / "SUPPORT_MANIFEST.json").read_text())
    for name, expected in inventory.items():
        if sha_file(root / name) != expected:
            raise ValueError("Support hash mismatch: " + name)
    cache = project / const["CACHE_REL"]
    if sha_file(cache / "manifest.json") != const["CACHE_SHA"]:
        raise ValueError("Validation cache manifest changed")
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["samples"] != 9981 or manifest["split"] != "valid" or not manifest["completed"]:
        raise ValueError("Only the complete frozen validation cache is accepted")
    copies = [(cache / "manifest.json", root / "assets/valid_cache/manifest.json", const["CACHE_SHA"])]
    for row in manifest["shards"]:
        name = row["filename"]
        if Path(name).name != name:
            raise ValueError("Unexpected shard name")
        copies.append((cache / "shards" / name, root / "assets/valid_cache/shards" / name, row["sha256"]))
    seed = project / const["SEED_REL"]
    copies.append((seed / "best_valid.pt", root / "assets/best_valid.pt", const["CHECKPOINT_SHA"]))
    progress.phase("1/7 — Validation parcalari ve checkpoint Drive'dan dogrulaniyor", len(copies))
    for j, (source, destination, expected) in enumerate(copies):
        progress.message(f"  [{j+1}/{len(copies)}] {source.name}")
        if not destination.is_file() or sha_file(destination) != expected:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".copying")
            shutil.copyfile(source, temporary)
            if sha_file(temporary) != expected:
                raise ValueError("Copied input hash mismatch: " + str(source))
            os.replace(temporary, destination)
        progress.update(j+1, force=True)
    geometry = project / AUDIT_REL / GEOMETRY_NAME
    if sha_file(geometry) != GEOMETRY_SHA:
        raise ValueError("Canonical 725,070-candidate geometry changed")
    history_path = project / HISTORY_REL / "predictions.npz"
    history_contract = json.loads((project / HISTORY_REL / "contract.json").read_text())
    if history_contract["contract_sha256"] != HISTORY_CONTRACT or sha_json(history_contract["payload"]) != HISTORY_CONTRACT:
        raise ValueError("Completed GPU control contract changed")
    history = read_npz(history_path, HISTORY_CONTRACT)
    audit_contract = json.loads((project / AUDIT_REL / "EB18_FROZEN_CONTRACT.json").read_text())
    if audit_contract["contract_sha256"] != const.get("EB18_CONTRACT", "ded89cf6e33d61ab4ae6d24a51f920a69e9a9cef0e9ae88741a2f91edabeaa6e"):
        raise ValueError("Canonical audit parent changed")
    progress.phase("1/7 — Sabit bagimliliklar Drive wheel dosyalarindan hazirlaniyor", 3)
    overlay = work / "overlay"
    overlay.mkdir(exist_ok=True)
    wheels = [project / "upstream_cache/verified_wheels" / name for name in (
        "spikingjelly-0.0.0.0.14-py3-none-any.whl",
        "rotary_embedding_torch-0.8.4-py3-none-any.whl", "einops-0.8.0-py3-none-any.whl")]
    for wheel in wheels:
        if not wheel.is_file():
            raise FileNotFoundError("Verified Drive wheel is missing: " + str(wheel))
    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
               "--no-index", "--no-deps", "--upgrade", "--target", str(overlay)] + [str(p) for p in wheels]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        progress.message(result.stdout)
        raise RuntimeError("Pinned wheel installation failed")
    progress.update(3, force=True)
    sys.path.insert(0, str(overlay))
    return root, read_npz(geometry), history, {
        "launcher_sha256": LAUNCHER_SHA, "support_sha256": const["SUPPORT_SHA"],
        "geometry_sha256": GEOMETRY_SHA, "history_predictions_sha256": sha_file(history_path),
        "history_contract_sha256": HISTORY_CONTRACT,
        "audit_contract_sha256": audit_contract["contract_sha256"],
        "wheel_sha256": {p.name: sha_file(p) for p in wheels},
        "checkpoint_sha256": const["CHECKPOINT_SHA"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--work", type=Path, default=Path("/content/ssc_mechanism_controls_20260930"))
    parser.add_argument("--records-only", action="store_true")
    parser.add_argument("--local-support", type=Path)
    parser.add_argument("--local-history", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    progress = Progress()
    output = None
    try:
        if args.records_only:
            if not args.local_support or not args.local_history or not args.output:
                parser.error("--records-only requires --local-support, --local-history, --output")
            args.output.mkdir(parents=True, exist_ok=True)
            progress.attach(args.output)
            progress.phase("Kayitli A/C dizileri analiz ediliyor", 9981)
            h = read_npz(args.local_history / "predictions.npz", HISTORY_CONTRACT)
            c = read_npz(args.local_support / "results_cpu_v2/validation_merged/validation_B1_predictions.npz")
            summary = record_analyses(h, c, args.output)
            progress.update(9981, force=True)
            progress.message(json.dumps(summary, indent=2))
            return
        root, geometry, history, provenance = prepare(args.project, args.work, progress)
        runner = load_runner(root)
        progress.phase("2/7 — CUDA ve sabit model yukleniyor")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Colab'da GPU secin; CPU fallback yapilmadi.")
        gpu, torch, functional, neuron, gpu_env = runner.build_model(root / "assets/SpikeSCR", root / "assets/best_valid.pt", "cuda:0", 1)
        cpu, _, _, _, cpu_env = runner.build_model(root / "assets/SpikeSCR", root / "assets/best_valid.pt", "cpu", 1)
        refs, panel_hash = runner.load_panel(root / "assets/canonical_clean_panel.npz")
        runtime = {"GPU": gpu_env, "CPU": cpu_env, "GPU_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
                   "python": platform.python_version(), "numpy": np.__version__,
                   "platform": platform.platform(), "CPU_model": cpu_model_name()}
        payload = {"version": VERSION, "script_sha256": sha_file(__file__),
                   "provenance": provenance, "runtime": runtime, "panel_sha256": panel_hash,
                   "cpu_threads": 1, "candidate_chunk_size": 128, "official_test_access": False,
                   "data_role": "frozen validation only; no training or test inference",
                   "canonical_original_audit_device": "CUDA; the later full-validation CPU replay is a separate record",
                   "qk_intervention": "isolate sources at q_lif2/k_lif2 while retaining the exact public singleton convention",
                   "D_scope": "all 25,820 originally class-changing candidates on both CPU and GPU; this is not a new exhaustive map of originally preserved candidates"}
        contract = sha_json(payload)
        output = args.project / CONTROL_REL / "mechanism_results" / ("controls_" + contract[:16])
        output.mkdir(parents=True, exist_ok=True)
        if (output / "contract.json").exists() and json.loads((output / "contract.json").read_text())["contract_sha256"] != contract:
            raise ValueError("Existing output belongs to another contract")
        atomic_json(output / "contract.json", {"contract_sha256": contract, "payload": payload})
        archived_code = output / Path(__file__).name
        shutil.copyfile(__file__, archived_code)
        progress.attach(output)
        progress.message("GPU: " + gpu_env["device_name"])
        progress.message("Kalici sonuc klasoru: " + str(output))
        progress.message("Resuming is permitted only for the same source, checkpoint, code, GPU and runtime contract.")
        progress.phase("2/7 — Validation cache ve aday kimlikleri dogrulaniyor")
        data = runner.ValidationData(root / "assets/valid_cache")
        if len(geometry["source_index"]) != 725070 or not np.array_equal(np.bincount(geometry["transition_code"], minlength=4), EXPECTED_TRANSITIONS):
            raise ValueError("Canonical transition counts changed")
        check_geometry(geometry, refs, data, history, progress)
        cpu_ref = read_npz(root / "results_cpu_v2/validation_merged/validation_B1_predictions.npz")
        progress.phase("3/7 — A/C: kayitli padding ve skor dagilimlari", 9981)
        record_summary = record_analyses(history, cpu_ref, output)
        progress.update(9981, force=True)
        progress.message("A/C mevcut kayitlardan tamamlandi; yeni inference yapilmadi.")
        clean = clean_gate(gpu, cpu, torch, functional, data, refs, contract, output, progress)
        pad_summary = padding_diagnosis(gpu, torch, functional, data, history, output, contract, progress)
        pair_summary = pair_diagnosis(gpu, torch, functional, data, history, output, contract, progress)
        batch_summary = full_batch_diagnosis(gpu, torch, functional, data, history, output, contract, progress)
        device_summary = device_replay(gpu, cpu, torch, functional, data, geometry, clean, output, contract, progress)
        summary = {"contract_sha256": contract, "complete": True, "official_test_access": False,
                   "recorded_A_C": record_summary, "padding_decomposition": pad_summary,
                   "B_pair_controls": pair_summary, "B_full_validation": batch_summary,
                   "D_device_replay": device_summary, "canonical_clean_gate": clean["summary"],
                   "elapsed_seconds_this_invocation": time.monotonic()-progress.started}
        atomic_json(output / "summary.json", summary)
        progress.phase("7/7 — Sonuc dosyalari ve tamamlanma kaydi", 1)
        inventory = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha_file(p)}
                     for p in sorted(output.rglob("*"))
                     if p.is_file() and p.suffix in (".npz", ".csv", ".py", ".json")
                     and p.name not in ("progress.json", "result_manifest.json", "failure.json")}
        atomic_json(output / "result_manifest.json", {"contract_sha256": contract, "complete": True, "files": inventory})
        progress.update(1, force=True)
        progress.close()
        atomic_json(output / "progress.json", {"contract_sha256": contract, "complete": True,
                    "phase": "TAMAMLANDI", "CPU_candidates": 25820, "GPU_candidates": 25820,
                    "validation_sources_B": 9981, "official_test_access": False})
        progress.message("\nTAMAMLANDI — sonuclar Drive'a kaydedildi.")
        progress.message("B original/reversed etiket farki: " + str(batch_summary["original_order_changed_labels"]))
        progress.message("B q/k kaynak ayirimi sonrasi fark: " + str(batch_summary["isolated_order_changed_labels"]))
        progress.message("D CPU-GPU ayni aday etiketi orani: " + f"{100*device_summary['CPU_GPU_prediction_agreement']:.4f}%")
        progress.message("D CPU adverse tanigi olan kaynaklar: " + str(device_summary["CPU_adverse_sources"]))
        progress.message("D GPU adverse tanigi olan kaynaklar: " + str(device_summary["GPU_adverse_sources"]))
        progress.message("summary.json dosyasini paylasmaniz yeterli; ham diziler de saklandi.")
    except BaseException:
        if output:
            atomic_json(output / "failure.json", {"complete": False, "error": traceback.format_exc(), "official_test_access": False})
        progress.message(traceback.format_exc())
        raise
    finally:
        progress.close()


# Remaining experiment functions are defined below before main is called.

def check_geometry(geometry, refs, data, history, progress):
    if set(np.unique(geometry["source_index"])) != set(refs):
        raise ValueError("Panel identities differ from the canonical map")
    if not np.array_equal(transition(geometry["label"], geometry["clean_prediction"],
                                     geometry["candidate_prediction"]), geometry["transition_code"]):
        raise ValueError("Canonical transitions do not reconstruct")
    progress.phase("2/7 — 100 kaynakta aday kuralinin ve girdilerin butunlugu", 100)
    for j, sid in enumerate(sorted(refs)):
        x, label = data.get(sid)
        mask = geometry["source_index"] == sid
        source = {k: v[mask] for k, v in geometry.items()}
        n = len(source["candidate_index"])
        t, u, f = source["from_bin"], source["to_bin"], source["feature"]
        expected = 2*int(np.count_nonzero(x)) - int(np.count_nonzero(x[0])) - int(np.count_nonzero(x[-1]))
        if n != expected or not np.array_equal(source["candidate_index"], np.arange(n)):
            raise ValueError("Neighborhood coverage or candidate ordering changed")
        if (np.any(t < 0) or np.any(t >= len(x)) or np.any(u < 0) or np.any(u >= len(x))
                or np.any(f < 0) or np.any(f >= 140) or np.any(np.abs(t-u) != 1)):
            raise ValueError("Illegal candidate coordinate")
        codes = (t.astype(np.int64)*len(x)+u)*140+f
        if len(np.unique(codes)) != n or not np.array_equal(x[t, f], source["multiplicity"]):
            raise ValueError("Repeated candidate or mismatched multiplicity")
        if not np.all(source["label"] == label) or label != refs[sid]["label"]:
            raise ValueError("Source labels differ")
        if not np.all(source["clean_prediction"] == refs[sid]["prediction"]):
            raise ValueError("Canonical clean predictions differ")
        if (int(history["native_horizon"][sid]) != len(x)
                or str(history["input_sha256"][sid]) != hashlib.sha256(x.tobytes()).hexdigest()):
            raise ValueError("Source input differs from completed GPU controls")
        progress.update(j+1)
    progress.update(100, force=True)


def scores(model, torch, functional, inputs, lengths, device, prefix=None):
    functional.reset_net(model)
    try:
        inputs = np.ascontiguousarray(inputs, dtype=np.float32)
        value = torch.from_numpy(inputs).to(device)
        mask = torch.arange(inputs.shape[1], device=device)[None, :] < torch.as_tensor(
            np.ascontiguousarray(lengths), device=device)[:, None]
        with torch.no_grad():
            probability = torch.softmax(model(value, mask).float(), dim=2)
            full = probability.sum(dim=0).cpu().numpy().astype(np.float32)
            if prefix is not None:
                if len(inputs) != 1:
                    raise ValueError("Prefix diagnosis is singleton only")
                pre = probability[:prefix].sum(dim=0).cpu().numpy().astype(np.float32)
                extra = probability[prefix:].sum(dim=0).cpu().numpy().astype(np.float32)
                return full[0], pre[0], extra[0]
        if full.shape != (len(inputs), 35) or not np.isfinite(full).all():
            raise ValueError("Invalid score output")
        return full
    finally:
        functional.reset_net(model)


def clean_gate(gpu, cpu, torch, functional, data, refs, contract, output, progress):
    path = output / "clean_panel.npz"
    if path.exists():
        z = read_npz(path, contract)
        if not np.array_equal(z["source_index"], sorted(refs)):
            raise ValueError("Saved clean panel identities differ")
    else:
        ids = np.asarray(sorted(refs), dtype=np.int32)
        native_cpu, native_gpu, isolated_gpu = [], [], []
        progress.phase("3/7 — 100 temiz kaynakta CPU/GPU ve singleton mudahale kontrolu", 300)
        done = 0
        for sid in ids:
            x, _ = data.get(sid)
            native_cpu.append(scores(cpu, torch, functional, x[None], [len(x)], "cpu")[0])
            done += 1; progress.update(done)
            native_gpu.append(scores(gpu, torch, functional, x[None], [len(x)], "cuda:0")[0])
            done += 1; progress.update(done)
            with qk_intervention(gpu):
                isolated_gpu.append(scores(gpu, torch, functional, x[None], [len(x)], "cuda:0")[0])
            done += 1; progress.update(done)
        z = {"source_index": ids, "CPU_scores": np.stack(native_cpu), "GPU_scores": np.stack(native_gpu),
             "GPU_qk_isolated_scores": np.stack(isolated_gpu),
             "canonical_scores": np.stack([refs[int(sid)]["scores"] for sid in ids]),
             "label": np.asarray([refs[int(sid)]["label"] for sid in ids], dtype=np.int16)}
        atomic_npz(path, contract_sha256=np.asarray(contract), **z)
        progress.update(300, force=True)
    # A diagnostic change must not change the singleton function.
    if not np.array_equal(z["GPU_scores"], z["GPU_qk_isolated_scores"]):
        raise ValueError("q/k intervention changed singleton scores; no downstream results are eligible")
    canonical_pred = z["canonical_scores"].argmax(axis=1)
    gpu_pred, cpu_pred = z["GPU_scores"].argmax(axis=1), z["CPU_scores"].argmax(axis=1)
    # Cross-device clean-label shifts are outcomes, not grounds for discarding
    # a run. Each new candidate type is computed against its own device's clean
    # prediction. Canonical-type retention remains a separate reported result.
    result = {int(sid): {"CPU": z["CPU_scores"][j], "GPU": z["GPU_scores"][j],
                        "label": int(z["label"][j])} for j, sid in enumerate(z["source_index"])}
    result["summary"] = {
        "sources": 100, "CPU_GPU_clean_label_disagreements": int(np.sum(cpu_pred != gpu_pred)),
        "GPU_vs_canonical_clean_label_disagreements": int(np.sum(gpu_pred != canonical_pred)),
        "CPU_vs_canonical_clean_label_disagreements": int(np.sum(cpu_pred != canonical_pred)),
        "GPU_vs_canonical_max_abs_score": float(np.max(np.abs(z["GPU_scores"]-z["canonical_scores"]))),
        "CPU_vs_canonical_max_abs_score": float(np.max(np.abs(z["CPU_scores"]-z["canonical_scores"]))),
        "singleton_intervention_bitwise_equal": True,
    }
    atomic_json(output / "clean_gate.json", result["summary"])
    progress.message("PASS — 100 temiz kaynak kosuldu; q/k mudahalesi singleton skorlarini bit-bit koruyor.")
    progress.message(f"  Kanonik temiz etiketten fark: CPU={result['summary']['CPU_vs_canonical_clean_label_disagreements']}, GPU={result['summary']['GPU_vs_canonical_clean_label_disagreements']}.")
    return result


def padding_diagnosis(gpu, torch, functional, data, history, output, contract, progress):
    """Separate altered valid-prefix output from added padded output mass."""
    gap = history["padded_horizon"] - history["native_horizon"]
    changed = history["B1_native_prediction"] != history["B1_padding_matched_prediction"]
    controls = np.flatnonzero((gap > 0) & ~changed)
    controls = controls[np.unique(np.linspace(0, len(controls)-1, min(64, len(controls)), dtype=int))]
    ids = np.unique(np.concatenate([np.flatnonzero(changed), controls])).astype(np.int32)
    folder = output / "A_padding_decomposition"
    folder.mkdir(exist_ok=True)
    parts = []
    progress.phase("4/7 — Padding: gecerli adimlar ve ek cikti adimlari ayristiriliyor", len(ids))
    for j, sid in enumerate(ids):
        path = folder / f"source_{sid:05d}.npz"
        if path.exists():
            row = read_npz(path, contract)
            if int(row["source_index"]) != sid:
                raise ValueError("Padding source identity differs")
        else:
            x, label = data.get(sid)
            H = int(history["padded_horizon"][sid])
            padded = np.zeros((1, H, 140), dtype=np.float32)
            padded[0, :len(x)] = x
            native = scores(gpu, torch, functional, x[None], [len(x)], "cuda:0")[0]
            full, prefix, extra = scores(gpu, torch, functional, padded, [len(x)], "cuda:0", len(x))
            if np.max(np.abs(full.astype(float)-prefix.astype(float)-extra.astype(float))) > 1e-4:
                raise ValueError("Prefix/extra score decomposition failed")
            if abs(float(extra.sum())-float(H-len(x))) > 1e-3:
                raise ValueError("Added softmax mass does not equal added time bins")
            row = {"source_index": np.asarray(sid), "label": np.asarray(label),
                   "native_horizon": np.asarray(len(x)), "padded_horizon": np.asarray(H),
                   "native_scores": native, "padded_scores": full,
                   "padded_valid_prefix_scores": prefix, "padded_extra_scores": extra}
            atomic_npz(path, contract_sha256=np.asarray(contract), **row)
        parts.append(row)
        progress.update(j+1)
    ids = np.asarray([int(r["source_index"]) for r in parts], dtype=np.int32)
    native = np.stack([r["native_scores"] for r in parts])
    full = np.stack([r["padded_scores"] for r in parts])
    prefix = np.stack([r["padded_valid_prefix_scores"] for r in parts])
    extra = np.stack([r["padded_extra_scores"] for r in parts])
    pred_native, pred_full = native.argmax(axis=1), full.argmax(axis=1)
    pred_prefix = prefix.argmax(axis=1)
    # This counterfactual holds native output fixed and adds the observed extra
    # mass; it is descriptive, not an independent physical perturbation.
    pred_native_plus_extra = (native + extra).argmax(axis=1)
    original_changed = changed[ids]
    rows = []
    for j, sid in enumerate(ids):
        rows.append({"source_index": int(sid), "original_padding_changed": int(original_changed[j]),
                     "padding_bins": int(history["padded_horizon"][sid]-history["native_horizon"][sid]),
                     "native_prediction": int(pred_native[j]), "padded_prediction": int(pred_full[j]),
                     "padded_prefix_prediction": int(pred_prefix[j]),
                     "native_plus_extra_prediction": int(pred_native_plus_extra[j]),
                     "valid_prefix_score_change_max_abs": float(np.max(np.abs(prefix[j]-native[j]))),
                     "extra_score_mass": float(extra[j].sum()),
                     "extra_mass_class_range": float(np.ptp(extra[j]))})
    write_csv(output / "A_padding_decomposition.csv", rows)
    atomic_npz(output / "A_padding_decomposition.npz", contract_sha256=np.asarray(contract),
               source_index=ids, native_scores=native, padded_scores=full,
               padded_valid_prefix_scores=prefix, padded_extra_scores=extra)
    summary = {"diagnostic_sources": len(ids), "selected_from_historical_padding_changes": int(original_changed.sum()),
               "unchanged_positive_padding_controls": int((~original_changed).sum()),
               "new_padding_label_changes": int(np.sum(pred_native != pred_full)),
               "prefix_only_labels_different_from_native": int(np.sum(pred_prefix != pred_native)),
               "new_padding_changes_restored_by_excluding_extra_bins": int(np.sum((pred_full != pred_native) & (pred_prefix == pred_native))),
               "native_plus_observed_extra_changes_label": int(np.sum(pred_native_plus_extra != pred_native)),
               "historical_native_label_disagreements": int(np.sum(pred_native != history["B1_native_prediction"][ids])),
               "historical_padded_label_disagreements": int(np.sum(pred_full != history["B1_padding_matched_prediction"][ids])),
               "interpretation": "Selected diagnostic cohort. Padded-prefix changes and extra-output contributions may both matter; no single-mechanism conclusion is forced."}
    atomic_json(output / "A_padding_decomposition_summary.json", summary)
    progress.update(len(ids), force=True)
    return summary


def capture_qk(model, source_isolated=False):
    captured, handles = {}, []
    for name, module in model.named_modules():
        if not name.endswith(("q_lif2", "k_lif2")):
            continue
        key = name.replace(".", "__")
        def hook(node, inputs, result, key=key):
            raw = inputs[0].detach().cpu().numpy().copy()
            spike = result.detach().cpu().numpy().copy()
            captured[key + "__caller_input"] = raw
            captured[key + "__caller_spikes"] = spike
            T, B, N = raw.shape
            captured[key + "__LIF_sequence_input"] = raw.reshape(B, T, N).transpose(1, 0, 2).copy() if source_isolated and B > 1 else raw
            captured[key + "__LIF_sequence_spikes"] = spike.reshape(B, T, N).transpose(1, 0, 2).copy() if source_isolated and B > 1 else spike
            if hasattr(node.v, "detach"):
                captured[key + "__final_v"] = node.v.detach().cpu().numpy().copy()
            if hasattr(getattr(node, "v_seq", None), "detach"):
                captured[key + "__v_seq"] = node.v_seq.detach().cpu().numpy().copy()
        handles.append(module.register_forward_hook(hook))
    return captured, handles


def pair_diagnosis(gpu, torch, functional, data, history, output, contract, progress):
    path = output / "B_pair_controls.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["contract_sha256"] != contract:
            raise ValueError("Pair control contract differs")
        return result["results"]
    changed = np.flatnonzero(history["B256_prediction"] != history["B256_reversed_prediction"])
    groups = sorted(set((changed // 256).tolist()))[:3]
    pairs = []
    for group in groups:
        x_id = int(changed[changed // 256 == group][0])
        y_id = group*256 + ((x_id-group*256+1) % min(256, 9981-group*256))
        pairs.append((x_id, y_id, int(history["padded_horizon"][x_id])))
    rows, structural = [], []
    progress.phase("5/7 — [x,y], [y,x], [x,x]; original ve q/k mudahaleleri", len(pairs)*18)
    done = 0
    for pair_no, (x_id, y_id, H) in enumerate(pairs):
        x, _ = data.get(x_id); y, _ = data.get(y_id)
        inputs = np.zeros((2, H, 140), dtype=np.float32)
        inputs[0, :len(x)] = x; inputs[1, :len(y)] = y
        lengths = np.asarray([len(x), len(y)], dtype=np.int32)
        singleton = []
        for j in range(2):
            singleton.append(scores(gpu, torch, functional, inputs[j:j+1], lengths[j:j+1], "cuda:0")[0])
            done += 1; progress.update(done)
        singleton = np.stack(singleton)
        structural.append({"pair": pair_no, "source_ids": [x_id, y_id], "layout": layout_owners(H, 2, 256)})
        for mode in ("original", "q", "k", "qk"):
            context = qk_intervention(gpu, mode) if mode != "original" else contextlib.nullcontext()
            with context:
                values = {}
                for name, val, lens in [("xy", inputs, lengths),
                                        ("yx", inputs[::-1], lengths[::-1]),
                                        ("xx", inputs[[0, 0]], lengths[[0, 0]]),
                                        ("xy_repeat", inputs, lengths)]:
                    captured, handles = capture_qk(gpu, source_isolated=(mode == "qk")) if pair_no == 0 and mode in ("original", "qk") and name != "xy_repeat" else ({}, [])
                    try:
                        value = scores(gpu, torch, functional, val, lens, "cuda:0")
                    finally:
                        for handle in handles:
                            handle.remove()
                    values[name] = value
                    if captured:
                        atomic_npz(output / "B_qk_state_traces" / f"pair0_{mode}_{name}.npz",
                                   contract_sha256=np.asarray(contract),
                                   source_index=np.asarray([y_id, x_id] if name == "yx" else [x_id, x_id] if name == "xx" else [x_id, y_id]),
                                   scores=value, **captured)
                    done += 1; progress.update(done)
                if not np.array_equal(values["xy"], values["xy_repeat"]):
                    raise ValueError("Identical-input pair replay changed scores")
                reverse = values["yx"][::-1]
                rows.append({"pair": pair_no, "x_source": x_id, "y_source": y_id, "padded_horizon": H,
                             "mode": mode, "order_changed_labels": int(np.sum(values["xy"].argmax(1) != reverse.argmax(1))),
                             "order_score_max_abs": float(np.max(np.abs(values["xy"]-reverse))),
                             "duplicate_x_position_score_max_abs": float(np.max(np.abs(values["xx"][0]-values["xx"][1]))),
                             "versus_padded_singleton_max_abs": float(np.max(np.abs(values["xy"]-singleton)))})
    result = {"pairs": rows, "structural_source_ownership": structural,
              "interpretation": "Pairs selected from known order-sensitive groups; q-only/k-only are ablations. Full validation tests the complete paired order contrast."}
    write_csv(output / "B_pair_controls.csv", rows)
    atomic_json(path, {"contract_sha256": contract, "results": result})
    progress.update(done, force=True)
    return result


def full_batch_diagnosis(gpu, torch, functional, data, history, output, contract, progress):
    conditions = ("original", "original_reversed", "qk_isolated", "qk_isolated_reversed")
    folder = output / "B_full_validation"
    folder.mkdir(exist_ok=True)
    parts = []
    progress.phase("5/7 — 9.981 kaynakta ayni GPU, ayni batch: sira ve q/k kontrolu", 160)
    for group, lo in enumerate(range(0, 9981, 256)):
        hi = min(lo+256, 9981)
        path = folder / f"group_{lo:05d}_{hi:05d}.npz"
        if path.exists():
            row = read_npz(path, contract)
            if not np.array_equal(row["source_index"], np.arange(lo, hi)):
                raise ValueError("Batch source identities differ")
        else:
            loaded = [data.get(sid) for sid in range(lo, hi)]
            lengths = np.asarray([len(x) for x, label in loaded], dtype=np.int32)
            H = int(lengths.max())
            inputs = np.zeros((hi-lo, H, 140), dtype=np.float32)
            for j, (x, label) in enumerate(loaded):
                if hashlib.sha256(x.tobytes()).hexdigest() != history["input_sha256"][lo+j]:
                    raise ValueError("Full-validation input changed")
                inputs[j, :len(x)] = x
            row = {"source_index": np.arange(lo, hi, dtype=np.int32),
                   "label": np.asarray([label for x, label in loaded], dtype=np.int16),
                   "native_horizon": lengths, "padded_horizon": np.asarray(H)}
            row["original_scores"] = scores(gpu, torch, functional, inputs, lengths, "cuda:0")
            progress.update(group*4+1)
            row["original_reversed_scores"] = scores(gpu, torch, functional, inputs[::-1], lengths[::-1], "cuda:0")[::-1]
            progress.update(group*4+2)
            with qk_intervention(gpu):
                row["qk_isolated_scores"] = scores(gpu, torch, functional, inputs, lengths, "cuda:0")
                progress.update(group*4+3)
                row["qk_isolated_reversed_scores"] = scores(gpu, torch, functional, inputs[::-1], lengths[::-1], "cuda:0")[::-1]
            atomic_npz(path, contract_sha256=np.asarray(contract), **row)
        parts.append(row)
        progress.update((group+1)*4, force=True)
    combined = {"source_index": np.concatenate([r["source_index"] for r in parts]),
                "label": np.concatenate([r["label"] for r in parts])}
    for condition in conditions:
        combined[condition+"_scores"] = np.concatenate([r[condition+"_scores"] for r in parts])
    if not np.array_equal(combined["source_index"], np.arange(9981)):
        raise ValueError("Full validation is incomplete")
    predictions = {c: combined[c+"_scores"].argmax(1) for c in conditions}
    original_change = predictions["original"] != predictions["original_reversed"]
    isolated_change = predictions["qk_isolated"] != predictions["qk_isolated_reversed"]
    original_delta = np.max(np.abs(combined["original_scores"]-combined["original_reversed_scores"]), axis=1)
    isolated_delta = np.max(np.abs(combined["qk_isolated_scores"]-combined["qk_isolated_reversed_scores"]), axis=1)
    summary = {"sources": 9981, "original_order_changed_labels": int(original_change.sum()),
               "isolated_order_changed_labels": int(isolated_change.sum()),
               "original_order_score_difference": distribution(original_delta),
               "isolated_order_score_difference": distribution(isolated_delta),
               "historical_original_label_disagreements": int(np.sum(predictions["original"] != history["B256_prediction"])),
               "historical_reverse_label_disagreements": int(np.sum(predictions["original_reversed"] != history["B256_reversed_prediction"])),
               "conditions_correct": {c: int(np.sum(p == combined["label"])) for c, p in predictions.items()},
               "isolated_batch_vs_historical_padded_singleton_label_disagreements":
                   int(np.sum(predictions["qk_isolated"] != history["B1_padding_matched_prediction"])),
               "layout_example": layout_owners(int(history["padded_horizon"][0]), 256, 256),
               "interpretation": "Only q_lif2/k_lif2 source isolation is intervened on. Remaining order differences are retained; zero residual differences are not assumed."}
    atomic_npz(output / "B_full_validation.npz", contract_sha256=np.asarray(contract), **combined)
    atomic_json(output / "B_full_validation_summary.json", summary)
    return summary


def device_replay(gpu, cpu, torch, functional, data, geometry, clean, output, contract, progress):
    selected = np.flatnonzero(geometry["transition_code"] != 0)
    if len(selected) != 25820:
        raise ValueError("Expected exactly 25,820 originally class-changing candidates")
    ids = np.unique(geometry["source_index"][selected])
    folder = output / "D_paired_candidates"
    folder.mkdir(exist_ok=True)
    progress.phase("6/7 — 25.820 aday, CPU ve GPU singleton karsilastirmasi", 51640)
    done, parts = 0, []
    for sid in ids:
        indices = selected[geometry["source_index"][selected] == sid]
        x, label = data.get(sid)
        progress.message(f"  Kaynak {sid}: {len(indices):,} aday; CPU + GPU, agirliklar sabit.")
        for lo in range(0, len(indices), 128):
            rows = indices[lo:lo+128]
            path = folder / f"source_{sid:05d}_chunk_{lo:05d}.npz"
            if path.exists():
                part = read_npz(path, contract)
                if not np.array_equal(part["geometry_row_index"], rows):
                    raise ValueError("Saved device-replay candidate indices differ")
                if part["CPU_scores"].shape != (len(rows), 35) or part["GPU_scores"].shape != (len(rows), 35):
                    raise ValueError("Saved candidate scores are incomplete")
                done += 2*len(rows)
            else:
                cpu_scores, gpu_scores = [], []
                for row_index in rows:
                    t, u, f = (int(geometry[k][row_index]) for k in ("from_bin", "to_bin", "feature"))
                    if x[t, f] <= 0:
                        raise ValueError("Candidate count underflow")
                    candidate = x.copy()
                    candidate[t, f] -= 1
                    candidate[u, f] += 1
                    if candidate.sum() != x.sum() or np.any(candidate < 0):
                        raise ValueError("Candidate count conservation failed")
                    cpu_scores.append(scores(cpu, torch, functional, candidate[None], [len(x)], "cpu")[0])
                    done += 1; progress.update(done)
                    gpu_scores.append(scores(gpu, torch, functional, candidate[None], [len(x)], "cuda:0")[0])
                    done += 1; progress.update(done)
                part = {"geometry_row_index": rows.astype(np.int32),
                        "source_index": geometry["source_index"][rows],
                        "candidate_index": geometry["candidate_index"][rows],
                        "label": geometry["label"][rows],
                        "from_bin": geometry["from_bin"][rows], "to_bin": geometry["to_bin"][rows],
                        "feature": geometry["feature"][rows],
                        "canonical_candidate_prediction": geometry["candidate_prediction"][rows],
                        "canonical_transition_code": geometry["transition_code"][rows],
                        "CPU_scores": np.stack(cpu_scores), "GPU_scores": np.stack(gpu_scores)}
                atomic_npz(path, contract_sha256=np.asarray(contract), **part)
            parts.append(part)
            progress.update(done, force=True)
    combined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]
                if k != "contract_sha256"}
    order = np.argsort(combined["geometry_row_index"])
    combined = {k: v[order] for k, v in combined.items()}
    if not np.array_equal(combined["geometry_row_index"], selected):
        raise ValueError("Paired candidate replay is incomplete")
    sid = combined["source_index"]
    cpu_clean = np.asarray([clean[int(s)]["CPU"].argmax() for s in sid])
    gpu_clean = np.asarray([clean[int(s)]["GPU"].argmax() for s in sid])
    p_cpu, p_gpu = combined["CPU_scores"].argmax(1), combined["GPU_scores"].argmax(1)
    t_cpu = transition(combined["label"], cpu_clean, p_cpu)
    t_gpu = transition(combined["label"], gpu_clean, p_gpu)
    combined.update(CPU_prediction=p_cpu, GPU_prediction=p_gpu, CPU_transition_code=t_cpu,
                    GPU_transition_code=t_gpu, CPU_clean_prediction=cpu_clean, GPU_clean_prediction=gpu_clean)
    rows = []
    old_adverse_sources = np.unique(sid[combined["canonical_transition_code"] == 1])
    for source in np.unique(sid):
        use = sid == source
        rows.append({"source_index": int(source), "replayed_candidates": int(use.sum()),
                     "canonical_adverse": int(np.sum(use & (combined["canonical_transition_code"] == 1))),
                     "CPU_adverse": int(np.sum(use & (t_cpu == 1))),
                     "GPU_adverse": int(np.sum(use & (t_gpu == 1))),
                     "CPU_GPU_prediction_disagreements": int(np.sum(use & (p_cpu != p_gpu)))})
    by_transition = {}
    for code, name in [(1, "adverse"), (2, "corrective"), (3, "lateral")]:
        use = combined["canonical_transition_code"] == code
        by_transition[name] = {
            "canonical_candidates": int(use.sum()),
            "CPU_GPU_prediction_agreement": float(np.mean(p_cpu[use] == p_gpu[use])),
            "CPU_retains_canonical_type": float(np.mean(t_cpu[use] == code)),
            "GPU_retains_canonical_type": float(np.mean(t_gpu[use] == code)),
            "CPU_retains_canonical_prediction": float(np.mean(p_cpu[use] == combined["canonical_candidate_prediction"][use])),
            "GPU_retains_canonical_prediction": float(np.mean(p_gpu[use] == combined["canonical_candidate_prediction"][use])),
        }
    # Predetermined first archived adverse witness per source, not a new
    # witness chosen after observing the device outcome.
    witnesses = []
    for source in old_adverse_sources:
        j = int(np.flatnonzero((sid == source) & (combined["canonical_transition_code"] == 1))[0])
        witnesses.append({"source_index": int(source), "candidate_index": int(combined["candidate_index"][j]),
                          "CPU_is_adverse": bool(t_cpu[j] == 1), "GPU_is_adverse": bool(t_gpu[j] == 1),
                          "CPU_prediction": int(p_cpu[j]), "GPU_prediction": int(p_gpu[j])})
    summary = {"paired_candidates": 25820, "CPU_GPU_prediction_agreement": float(np.mean(p_cpu == p_gpu)),
               "CPU_GPU_prediction_disagreements": int(np.sum(p_cpu != p_gpu)),
               "CPU_GPU_transition_agreement": float(np.mean(t_cpu == t_gpu)),
               "CPU_GPU_score_difference": distribution(np.max(np.abs(combined["CPU_scores"]-combined["GPU_scores"]), axis=1)),
               "canonical_adverse_sources": old_adverse_sources.tolist(),
               "CPU_adverse_sources": np.unique(sid[t_cpu == 1]).tolist(),
               "GPU_adverse_sources": np.unique(sid[t_gpu == 1]).tolist(),
               "by_canonical_transition": by_transition,
               "first_canonical_adverse_witness_per_source": witnesses,
               "interpretation": "Selected canonical class-changing cohort only. Originally class-preserved candidates were not replayed, so new GPU adverse incidence outside this cohort is not estimated."}
    atomic_npz(output / "D_paired_candidate_predictions.npz", contract_sha256=np.asarray(contract), **combined)
    write_csv(output / "D_per_source.csv", rows)
    write_csv(output / "D_first_adverse_witnesses.csv", witnesses)
    atomic_json(output / "D_device_replay_summary.json", summary)
    progress.update(51640, force=True)
    return summary


if __name__ == "__main__":
    main()
