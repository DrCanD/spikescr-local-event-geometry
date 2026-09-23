"""Cryptographic integrity and safe reference-file access."""
from __future__ import annotations
import json
import re
from pathlib import Path, PurePosixPath
from .io import sha256_file, read_json


def checked_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if not relative or '\\' in relative or path.is_absolute() or any(p in ('..','.') for p in path.parts):
        raise ValueError('Unsafe manifest path')
    if ':' in relative or '\x00' in relative or '\n' in relative:
        raise ValueError('Unsafe manifest path')
    target=root.joinpath(*path.parts)
    if any(p.is_symlink() for p in (target, *target.parents) if p==root or root in p.parents):
        raise ValueError('Symlinks are not accepted in the immutable release')
    if root.resolve() not in target.resolve().parents:
        raise ValueError('Manifest path escapes repository')
    return target


def verify_integrity(root: Path) -> dict:
    manifest=read_json(root/'checksums/manifest.json')
    if manifest.get('schema_version')!=1 or not isinstance(manifest.get('files'),dict):
        raise ValueError('Invalid integrity manifest')
    missing=[];modified=[]
    for relative, expected in manifest['files'].items():
        if not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('Invalid SHA256 in manifest')
        path=checked_path(root,relative)
        if not path.is_file():missing.append(relative)
        elif sha256_file(path)!=expected:modified.append(relative)
    # Untracked files in executable/config/reference directories must not be imported unnoticed.
    unexpected=[]
    for directory in ('src','scripts','configs','data','tests','.github'):
        for path in (root/directory).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc' or any(part.endswith('.egg-info') for part in path.parts):continue
            relative=path.relative_to(root).as_posix()
            if relative not in manifest['files']:unexpected.append(relative)
    report=dict(status='PASS' if not(missing or modified or unexpected) else 'FAIL',
                checked_files=len(manifest['files']),missing=missing,modified=modified,unexpected=unexpected,
                trust_note='Checksums detect corruption against the included manifest, not authenticity against an independent trusted publisher.')
    if report['status']!='PASS':raise ValueError(json.dumps(report,indent=2))
    return report


def verify_checkpoint(root: Path) -> dict:
    import hashlib
    import numpy as np
    import torch
    config=read_json(root/'configs/experiment.json')
    path=root/'data/model/frozen_checkpoint.pt'
    if sha256_file(path)!=config['checkpoint_sha256']:raise ValueError('Checkpoint file hash mismatch')
    blob=torch.load(path,map_location='cpu',weights_only=True)
    if blob['epoch']!=config['checkpoint_epoch_zero_based']:raise ValueError('Checkpoint epoch mismatch')
    digest=hashlib.sha256()
    for key in sorted(blob['model_state_dict']):
        t=blob['model_state_dict'][key].detach().cpu().contiguous()
        if not torch.isfinite(t).all():raise ValueError('Nonfinite checkpoint tensor')
        digest.update(key.encode('utf-8'));digest.update(str(t.dtype).encode('ascii'))
        digest.update(np.asarray(t.shape,dtype=np.int64).tobytes());digest.update(t.numpy().tobytes())
    if digest.hexdigest()!=config['state_dict_sha256']:raise ValueError('Checkpoint tensor-state digest mismatch')
    return dict(status='PASS',file_sha256=config['checkpoint_sha256'],state_dict_sha256=digest.hexdigest(),
                stored_epoch_zero_based=blob['epoch'],selected_epoch_one_based=blob['epoch']+1,
                tensors=len(blob['model_state_dict']),unsafe_pickle_enabled=False,
                model_forward_executed=False)


def verify_raw_validation(root: Path, path: Path) -> dict:
    import h5py
    import numpy as np
    from .io import load_npz, read_numeric_csv
    from .core import transform_events
    from ._kernel import literal_event_transform
    config=read_json(root/'configs/experiment.json')
    if sha256_file(path)!=config['validation_h5_sha256']:
        raise ValueError('Not the hash-verified official validation HDF5. Nothing was read as HDF5.')
    panel=load_npz(root/'data/panel/inputs.npz'); rows=read_numeric_csv(root/'data/panel/manifest.csv')
    with h5py.File(path,'r') as h:
        if len(h['labels'])!=9981:raise ValueError('Wrong validation cardinality')
        for row in rows:
            sid=row['source_index'];t=h['spikes/times'][sid];u=h['spikes/units'][sid]
            a=transform_events(t,u);b=literal_event_transform(t,u,700,5,5)
            if not np.array_equal(a,b) or not np.array_equal(a,panel[f'source_{sid:05d}']):raise ValueError('Panel transform mismatch')
            if int(h['labels'][sid])!=row['label']:raise ValueError('Panel label mismatch')
    return dict(status='PASS',sources=100,two_implementations_agreed=True,raw_h5_sha256=config['validation_h5_sha256'],raw_official_test_opened=False)
