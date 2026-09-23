import tempfile
import unittest
from pathlib import Path
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
    def test_complete_result_reproduction_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);report=reproduce(ROOT,out)
            self.assertEqual(report['status'],'PASS')
            self.assertEqual(report['candidates_checked'],725070)
            # Check the exported scientific tables against the recorded numerical results.
            outcomes={r['outcome']:r for r in read_numeric_csv(out/'tables/outcome_partition.csv')}
            expected=read_json(ROOT/'data/reference/neighborhood_totals.json')
            for name,row in outcomes.items():
                self.assertEqual(row['unique_n'],expected[name+'_unique'])
                self.assertEqual(row['event_weighted_n'],expected[name+'_event_weighted'])
                self.assertAlmostEqual(row['unique_percent'],100*row['unique_n']/expected['unique_candidates'],places=12)
                self.assertAlmostEqual(row['event_weighted_percent'],100*row['event_weighted_n']/expected['event_weighted_candidates'],places=12)
            actual=read_numeric_csv(out/'tables/source_geometry.csv');expected=read_numeric_csv(ROOT/'data/neighborhood/source_geometry.csv')
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):
                self.assertEqual(compare_tree({k:a[k] for k in b},b),[])
            internal=read_json(out/'statistics/internal.json')
            for row in read_numeric_csv(out/'tables/trace_contrasts.csv'):
                ref=internal['source_level_inference']['transition_vs_preserved'][row['transition']][row['boundary']]
                self.assertEqual(row['mean_contrast'],ref['mean_contrast'])
                self.assertEqual(row['q'],ref['fdr_bh_q_within_transition_vs_preserved_family'])
            for row in read_numeric_csv(out/'tables/adverse_patching.csv'):
                ref=internal['clean_activation_patch']['adverse'][row['boundary']]
                self.assertEqual(row['candidate_pooled_percent'],100*ref['restored_clean_prediction_rate_unique'])
            panel=read_json(out/'statistics/panel.json')
            self.assertAlmostEqual(panel['adverse_rate_ci95'][0]*100,.2532,places=4)
            self.assertAlmostEqual(panel['adverse_rate_ci95'][1]*100,1.6978,places=4)
            self.assertEqual(report['new_model_inference_performed'],False)

if __name__=='__main__':unittest.main()
