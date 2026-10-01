"""The official SSC validation split, read from the hash-checked HDF5 and transformed on demand,
or the 100 released panel inputs when only the audit panel is needed."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from . import paths
from .core import transform_events, validate_input
from .io import load_npz, read_json, read_numeric_csv, sha256_file


class ValidationData:
    """``data.get(source_index)`` -> (count tensor [horizon,140] uint16, label). Nothing else is read."""

    def __init__(self, root: Path, h5_path: Path):
        import h5py
        config = read_json(root / paths.EXPERIMENT_CONFIG)
        self.sha256 = sha256_file(h5_path)
        if self.sha256 != config['validation_h5_sha256']:
            raise ValueError('Not the hash-verified official validation HDF5 (ssc_valid.h5); nothing was read.')
        self.path = Path(h5_path)
        self.file = h5py.File(self.path, 'r')
        if len(self.file['labels']) != 9981:
            raise ValueError('Wrong validation cardinality')
        self.cache: dict[int, tuple[np.ndarray, int]] = {}
        self.provenance = {'file': self.path.name, 'sha256': self.sha256, 'sources': 9981, 'split': 'official validation',
                           'transform': '5 ms bins, 5-channel units, source-specific horizon, first-event-relative time'}

    def get(self, source_index: int) -> tuple[np.ndarray, int]:
        if source_index not in self.cache:
            if self.file is None:
                raise KeyError(f'source {source_index} is not one of the 100 released panel inputs; supply --h5 for the full validation set')
            x = transform_events(self.file['spikes/times'][source_index], self.file['spikes/units'][source_index])
            self.cache[source_index] = (x, int(self.file['labels'][source_index]))
        return self.cache[source_index]

    @classmethod
    def from_panel(cls, root: Path) -> 'ValidationData':
        """The 100 released panel inputs (data/panel/inputs.npz); enough for the panel-level steps."""
        self = cls.__new__(cls)
        inputs = load_npz(root / paths.PANEL_INPUTS); rows = read_numeric_csv(root / paths.PANEL_SOURCES)
        self.file = None; self.path = root / paths.PANEL_INPUTS; self.sha256 = sha256_file(self.path)
        self.cache = {row['source_index']: (validate_input(inputs[f"source_{row['source_index']:05d}"]), int(row['label'])) for row in rows}
        self.provenance = {'file': paths.PANEL_INPUTS, 'sha256': self.sha256, 'sources': len(self.cache), 'split': 'official validation (released panel only)',
                           'transform': '5 ms bins, 5-channel units, source-specific horizon, first-event-relative time'}
        return self

    @staticmethod
    def input_hash(x: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
