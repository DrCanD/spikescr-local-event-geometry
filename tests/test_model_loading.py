"""Load the real pinned model; the lightweight analysis job may skip this test."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from ssc_geometry.io import repository_root

ROOT = repository_root()


class RealModelLoadingTests(unittest.TestCase):
    def test_real_checkpoint_parameter_inventory_and_boundaries(self):
        missing = []
        for package in ('torch', 'torchvision', 'spikingjelly', 'rotary-embedding-torch', 'einops'):
            try:
                importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                missing.append(package)
        if not (ROOT/'.cache/upstream/source_verification.json').is_file():
            missing.append('verified upstream source (run scripts/prepare_upstream.py)')
        if missing:
            reason = 'Real-model prerequisites missing: ' + ', '.join(missing)
            if os.environ.get('SSC_REQUIRE_REAL_MODEL') == '1':
                self.fail(reason)
            self.skipTest(reason)
        # The loader deliberately modifies the process-local BatchNorm adapter.
        # Keep this real-model test independent of the toy forward-kernel tests.
        code = '''
import json
from pathlib import Path
from ssc_geometry.upstream import load_model
root=Path.cwd()
model,device,stages,environment=load_model(root,root/'.cache/upstream','cpu',True)
fixed={name:p.numel() for name,p in model.named_parameters() if not p.requires_grad}
print(json.dumps({'counts':environment['parameter_counts'],
                  'fixed':fixed,'stages':[s['stage'] for s in stages]}))
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        observed = json.loads(result.stdout)
        self.assertEqual(observed['counts'], {
            'total_parameters': 3302416,
            'trainable_parameters': 3302400,
            'nontrainable_parameters': 16,
        })
        self.assertEqual(observed['fixed'], {
            'blocks.0.attn.rotary_emb.freqs': 8,
            'blocks.1.attn.rotary_emb.freqs': 8,
        })
        self.assertEqual(observed['stages'], [
            'stem', 'attention_1', 'local_1', 'block_1',
            'attention_2', 'local_2', 'block_2',
        ])


if __name__ == '__main__':
    unittest.main()
