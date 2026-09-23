import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from ssc_geometry.io import repository_root,write_json
from ssc_geometry.upstream import git_blob_sha1,verify_source
from ssc_geometry.inference import execution_digest
ROOT=repository_root()


class EntrypointTests(unittest.TestCase):
    def test_all_help_commands(self):
        for name in ('prepare_upstream.py','run_singleton_audit.py','replay_benchmark.py'):
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/name),'--help'],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,(name,result.stderr))
    def test_official_test_requires_explicit_authorization(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/replay_benchmark.py'),'--split','test','--h5','/sentinel/DO_NOT_OPEN.h5','--out','/sentinel/DO_NOT_WRITE'],text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Official test replay requires',result.stderr)
    def test_git_blob_sha(self):
        self.assertEqual(git_blob_sha1(b'hello\n'),'ce013625030ba8dba906f756967f9e9ca394464a')
    def test_execution_digest_ignores_install_metadata_but_tracks_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'src').mkdir();(root/'pyproject.toml').write_text('[project]\n')
            code=root/'src/example.py';code.write_text('x=1\n')
            before=execution_digest(root)
            metadata=root/'src/example.egg-info';metadata.mkdir();(metadata/'PKG-INFO').write_text('generated\n')
            cache=root/'src/__pycache__';cache.mkdir();(cache/'example.pyc').write_bytes(b'compiled')
            self.assertEqual(execution_digest(root),before)
            code.write_text('x=2\n')
            self.assertNotEqual(execution_digest(root),before)
    def test_unverified_upstream_cannot_be_imported(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)/'repo';s=Path(tmp)/'source';(r/'configs').mkdir(parents=True);(s/'models').mkdir(parents=True)
            write_json(r/'configs/upstream_blobs.json',{'models/a.py':git_blob_sha1(b'original\n')})
            write_json(r/'configs/experiment.json',{'upstream_commit':'a'*40})
            (s/'models/a.py').write_bytes(b'changed\n')
            with self.assertRaises(ValueError):verify_source(r,s)

if __name__=='__main__':unittest.main()
