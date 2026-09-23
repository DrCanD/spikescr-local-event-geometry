import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from ssc_geometry.io import repository_root,write_json
from ssc_geometry.upstream import git_blob_sha1,verify_source
ROOT=repository_root()


class EntrypointTests(unittest.TestCase):
    def test_all_help_commands(self):
        for name in ('prepare_upstream.py','run_singleton_audit.py','replay_benchmark.py','publish_repository.py'):
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/name),'--help'],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,(name,result.stderr))
    def test_official_test_requires_explicit_authorization(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/replay_benchmark.py'),'--split','test','--h5','/sentinel/DO_NOT_OPEN.h5','--out','/sentinel/DO_NOT_WRITE'],text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Official test replay requires',result.stderr)
    def test_publication_dry_run_does_not_need_credentials(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/publish_repository.py')],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('No remote changes made',result.stdout)
    def test_git_blob_sha(self):
        self.assertEqual(git_blob_sha1(b'hello\n'),'ce013625030ba8dba906f756967f9e9ca394464a')
    def test_unverified_upstream_cannot_be_imported(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)/'repo';s=Path(tmp)/'source';(r/'configs').mkdir(parents=True);(s/'models').mkdir(parents=True)
            write_json(r/'configs/upstream_blobs.json',{'models/a.py':git_blob_sha1(b'original\n')})
            write_json(r/'configs/experiment.json',{'upstream_commit':'a'*40})
            (s/'models/a.py').write_bytes(b'changed\n')
            with self.assertRaises(ValueError):verify_source(r,s)

if __name__=='__main__':unittest.main()
