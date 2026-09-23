"""Fetch and validate the exact upstream source without redistributing it."""
from __future__ import annotations
import hashlib
import importlib
import importlib.metadata
import os
import platform
import random
import sys
import urllib.request
from pathlib import Path
from .io import read_json, write_json
from .integrity import checked_path, verify_checkpoint


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(b'blob '+str(len(raw)).encode('ascii')+b'\0'+raw).hexdigest()


def verify_source(root: Path, source: Path) -> dict:
    source=source.resolve();blobs=read_json(root/'configs/upstream_blobs.json')
    for relative,expected in blobs.items():
        path=checked_path(source,relative)
        if not path.is_file() or git_blob_sha1(path.read_bytes())!=expected:
            raise ValueError('Pinned upstream source mismatch: '+relative)
    # Prevent a local __init__.py from executing code not covered by source hashes.
    allowed=set(blobs)
    for package in ('configs','models','module'):
        for path in (source/package).rglob('*.py'):
            rel=path.relative_to(source).as_posix()
            if rel not in allowed and (path.name!='__init__.py' or path.read_bytes().strip()):
                raise ValueError('Unexpected executable upstream file: '+rel)
    return {'status':'PASS','verified_blob_count':len(blobs),'commit':read_json(root/'configs/experiment.json')['upstream_commit']}


def prepare_source(root: Path, destination: Path) -> dict:
    if destination.exists():
        return verify_source(root,destination)
    config=read_json(root/'configs/experiment.json');blobs=read_json(root/'configs/upstream_blobs.json')
    destination.mkdir(parents=True)
    try:
        for relative,expected in blobs.items():
            url=f"https://raw.githubusercontent.com/JackieWang9811/SpikeSCR/{config['upstream_commit']}/{relative}"
            request=urllib.request.Request(url,headers={'User-Agent':'spikescr-local-event-geometry/0.1'})
            with urllib.request.urlopen(request,timeout=60) as response:
                if not response.geturl().startswith('https://raw.githubusercontent.com/'):
                    raise ValueError('Unexpected download redirect')
                raw=response.read(2*1024*1024+1)
            if len(raw)>2*1024*1024 or git_blob_sha1(raw)!=expected:raise ValueError('Downloaded source hash mismatch '+relative)
            path=checked_path(destination,relative);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        for name in ('configs','models','module'):(destination/name/'__init__.py').touch()
        report=verify_source(root,destination)
        write_json(destination/'source_verification.json',report)
        return report
    except Exception:
        # Keep the failed download distinguishable. Do not accept or import it.
        write_json(destination/'INCOMPLETE.json',{'status':'FAIL','instruction':'Remove this incomplete cache and repeat prepare-upstream.'})
        raise


def load_model(root: Path, source: Path, device_name: str, allow_nonreference: bool=False):
    # Set determinism before any CUDA initialization.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    from . import _forward_kernel as forward
    cfg=read_json(root/'configs/experiment.json')
    source_report=verify_source(root,source)
    if (source/'INCOMPLETE.json').exists():raise ValueError('Upstream cache is marked incomplete')
    expected_packages={'spikingjelly':'0.0.0.0.14','rotary-embedding-torch':'0.8.4','einops':'0.8.0'}
    versions={}
    for package,version in expected_packages.items():
        installed=importlib.metadata.version(package);versions[package]=installed
        if installed!=version:raise RuntimeError(f'{package} version {installed} does not match {version}')
    device=torch.device(device_name)
    version_matches=str(torch.__version__)==cfg['recorded_torch_version']
    reference_runtime=version_matches and device.type=='cuda'
    if not reference_runtime and not allow_nonreference:
        raise RuntimeError('Reference audit requires the recorded torch build on CUDA. A CPU or other-version portability run requires --allow-nonreference-environment and is not a canonical validation.')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    if os.environ['CUBLAS_WORKSPACE_CONFIG']!=':4096:8':raise RuntimeError('Unexpected CUBLAS_WORKSPACE_CONFIG')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if device.type=='cuda':torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    checkpoint_report=verify_checkpoint(root)
    # Use a minimal verified source cache, rather than an arbitrary on-path module.
    source=source.resolve()
    for prefix in ('configs','models','module'):
        for name in list(sys.modules):
            if name==prefix or name.startswith(prefix+'.'):del sys.modules[name]
    sys.path.insert(0,str(source))
    from spikingjelly.activation_based import functional, layer
    original_bn=layer.BatchNorm1d
    if getattr(original_bn,'_ssc_3d_adapter',False):raise RuntimeError('Model loader must be called once per process')
    class BatchNorm1d3DAdapter(original_bn):
        _ssc_3d_adapter=True
        def forward(self,x):
            if getattr(self,'step_mode','s')=='m' and x.ndim==3:
                return super().forward(x.unsqueeze(-1)).squeeze(-1)
            return super().forward(x)
    layer.BatchNorm1d=BatchNorm1d3DAdapter
    forward.SJ_FUNCTIONAL=functional
    config=importlib.import_module('configs.best_config_SSC_former').Config()
    for key,value in cfg['public_config_expected'].items():
        if getattr(config,key,None)!=value:raise ValueError('Upstream config mismatch '+key)
    if hasattr(config,'use_ln'):raise ValueError('Unexpected use_ln field')
    config.use_ln=False;config.time_step=5;config.epochs=300;config.seed=312;config.backend='torch'
    config.n_hidden_neurons=int(config.n_hidden_neurons_list[0]);config.hidden_dims=int(config.mlp_ratio*config.n_hidden_neurons)
    config.n_inputs=140;config.n_outputs=35
    model=importlib.import_module('models.spikescr').SpikeDrivenTransformer(config).to(device)
    blob=torch.load(root/'data/model/frozen_checkpoint.pt',weights_only=True,map_location='cpu')
    model.load_state_dict(blob['model_state_dict'],strict=True);model.eval();functional.reset_net(model)
    parameter_counts = {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'nontrainable_parameters': sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    for name, observed in parameter_counts.items():
        if observed != cfg[name]:
            raise ValueError(f'Model {name} mismatch: expected {cfg[name]}, observed {observed}')
    named=dict(model.named_modules());stages=[]
    for name,path,cls in zip(cfg['stage_names'],cfg['stage_paths'],cfg['stage_classes']):
        module=named.get(path)
        if module is None or module.__class__.__name__!=cls:raise ValueError('Internal boundary mismatch '+name)
        stages.append({'stage':name,'module':module,'module_name':path,'module_class':cls})
    environment={'torch':str(torch.__version__),'numpy':np.__version__,'python':sys.version.split()[0],
        'device_type':device.type,'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        'recorded_torch_and_cuda_mode_matched':reference_runtime,'packages':versions,'upstream':source_report,
        'checkpoint':checkpoint_report,'parameter_counts':parameter_counts,'full_original_environment_known':False,
        'runtime':{'platform':platform.platform(),'python_full':sys.version,
            'cuda_build':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'torchvision':importlib.metadata.version('torchvision'),
            'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
            'mkldnn_enabled':torch.backends.mkldnn.enabled,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG']}}
    return model,device,stages,environment
