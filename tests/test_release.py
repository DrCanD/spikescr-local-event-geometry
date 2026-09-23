import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from ssc_geometry.io import repository_root,read_json,read_numeric_csv,compare_tree,write_json
from ssc_geometry.integrity import verify_integrity
from ssc_geometry.analysis import reproduce

ROOT=repository_root()


class ReleaseTests(unittest.TestCase):
    def test_release_integrity(self):
        self.assertEqual(verify_integrity(ROOT)['status'],'PASS')
    def test_tampered_file_is_rejected(self):
        # Minimal self-contained fixture; no real data are changed.
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);(r/'checksums').mkdir();(r/'data').mkdir();p=r/'data/item';p.write_bytes(b'original')
            from ssc_geometry.io import sha256_file
            write_json(r/'checksums/manifest.json',{'schema_version':1,'files':{'data/item':sha256_file(p)}})
            self.assertEqual(verify_integrity(r)['status'],'PASS')
            p.write_bytes(b'modified')
            with self.assertRaises(ValueError):verify_integrity(r)
    def test_untracked_executable_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);(r/'checksums').mkdir();(r/'src').mkdir();write_json(r/'checksums/manifest.json',{'schema_version':1,'files':{}})
            (r/'src/extra.py').write_text('x=1')
            with self.assertRaises(ValueError):verify_integrity(r)
    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);(r/'checksums').mkdir();write_json(r/'checksums/manifest.json',{'schema_version':1,'files':{'data/absent':'0'*64}})
            with self.assertRaises(ValueError):verify_integrity(r)
    def test_complete_result_reproduction_and_figure_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);report=reproduce(ROOT,out)
            self.assertEqual(report['status'],'PASS')
            self.assertEqual(report['candidates_checked'],725070)
            # Publication CSVs must not silently substitute weights for percentages or reorder outcomes.
            for name in ('fig2_outcome_partition.csv','fig4_trace_contrasts.csv','fig4_adverse_patching.csv'):
                actual=read_numeric_csv(out/'matlab'/name);expected=read_numeric_csv(ROOT/'matlab'/name)
                self.assertEqual(compare_tree(actual,expected),[],name)
            actual=read_numeric_csv(out/'matlab/fig3_source_geometry.csv');expected=read_numeric_csv(ROOT/'matlab/fig3_source_geometry.csv')
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):
                self.assertEqual(compare_tree({k:a[k] for k in b},b),[])
            panel=read_json(out/'statistics/panel.json')
            self.assertAlmostEqual(panel['adverse_rate_ci95'][0]*100,.2532,places=4)
            self.assertAlmostEqual(panel['adverse_rate_ci95'][1]*100,1.6978,places=4)
            self.assertEqual(report['new_model_inference_performed'],False)

if __name__=='__main__':unittest.main()
