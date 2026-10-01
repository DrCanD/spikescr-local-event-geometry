"""The complete census: regenerated tables against the recorded values (Tables 3, 5-8)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from ssc_geometry import paths
from ssc_geometry.census import run, final_block_descriptives, internal_output_correlations, accuracy_decomposition, derive_transition_masks
from ssc_geometry.io import repository_root, read_json, read_numeric_csv, compare_tree

ROOT = repository_root()


class CensusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = tempfile.TemporaryDirectory()
        cls.report = run(ROOT, Path(cls.out.name), progress=lambda *args, **kwargs: None)

    @classmethod
    def tearDownClass(cls):
        cls.out.cleanup()

    def test_report_status_and_scope(self):
        self.assertEqual(self.report['status'], 'PASS')
        self.assertEqual(self.report['candidates_checked'], 725070)
        self.assertFalse(self.report['new_model_inference_performed'])

    def test_table_03_matches_recorded_totals(self):
        rows = {r['outcome']: r for r in read_numeric_csv(Path(self.out.name) / 'table_03.csv')}
        expected = read_json(ROOT / paths.REFERENCE['neighborhood'])
        for label, key in [('Class-preserved', 'class_preserved'), ('Any class change', 'class_changed'), ('Adverse', 'adverse'),
                           ('Corrective', 'corrective'), ('Lateral', 'lateral')]:
            self.assertEqual(rows[label]['unique_n'], expected[key + '_unique'])
            self.assertEqual(rows[label]['event_weighted_n'], expected[key + '_event_weighted'])
            self.assertAlmostEqual(rows[label]['unique_percent'], 100 * rows[label]['unique_n'] / expected['unique_candidates'], places=12)

    def test_source_table_is_reproduced(self):
        actual = read_numeric_csv(Path(self.out.name) / 'sources.csv'); expected = read_numeric_csv(ROOT / paths.SOURCES)
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertEqual(compare_tree({k: a[k] for k in b}, b), [])

    def test_tables_06_and_08_come_from_the_internal_statistics(self):
        internal = self.report['internal']
        for row in read_numeric_csv(Path(self.out.name) / 'table_06.csv'):
            stage = row['boundary'].lower().replace(' ', '_')
            for group in ('adverse', 'corrective', 'lateral'):
                ref = internal['source_level_inference']['transition_vs_preserved'][group][stage]
                self.assertEqual(row[f'{group}_mean_contrast'], ref['mean_contrast'])
                self.assertEqual(row[f'{group}_q'], ref['fdr_bh_q_within_transition_vs_preserved_family'])
        for row in read_numeric_csv(Path(self.out.name) / 'table_08.csv'):
            stage = row['boundary'].lower().replace(' ', '_')
            ref = internal['clean_activation_patch']['adverse'][stage]
            self.assertEqual(row['candidate_pooled_percent'], 100 * ref['restored_clean_prediction_rate_unique'])

    def test_table_05_and_07_values(self):
        fb = {r['outcome']: r for r in self.report['final_block']['table']}
        self.assertEqual(fb['class_preserved']['candidates'], 699250)
        self.assertAlmostEqual(fb['class_preserved']['pooled_median'], 0.6464, places=4)
        self.assertAlmostEqual(fb['adverse']['equal_source_mean'], 0.7007, places=4)
        self.assertEqual(self.report['final_block']['preserved_nonzero'], 617941)
        self.assertEqual(self.report['final_block']['preserved_exact_zero'], 81309)
        t7 = {r['candidates']: r for r in self.report['internal_output_correlations']['table']}
        self.assertAlmostEqual(t7['all_neighbors']['score_l2'], 0.4410, places=4)
        self.assertAlmostEqual(t7['nonzero_internal_response']['absolute_margin'], 0.0544, places=4)

    def test_panel_statistics(self):
        panel = self.report['panel']
        self.assertEqual((panel['clean_correct_sources'], panel['adverse_sensitive_sources'], panel['any_change_sources']), (84, 13, 25))
        self.assertAlmostEqual(panel['adverse_rate_ci95'][0] * 100, .2532, places=4)
        self.assertAlmostEqual(panel['adverse_rate_ci95'][1] * 100, 1.6978, places=4)
        self.assertAlmostEqual(panel['spearman_rho'], -0.5688, places=4)
        self.assertAlmostEqual(self.report['accuracy_decomposition']['unique_candidate_gain_percentage_points'], 0.5343, places=4)


class SyntheticCensusTests(unittest.TestCase):
    def test_transition_masks_partition(self):
        masks = derive_transition_masks(1, 1, np.array([1, 2, 1, 3]))
        self.assertEqual(masks['code'].tolist(), [0, 1, 0, 1])
        masks = derive_transition_masks(1, 2, np.array([2, 1, 3]))
        self.assertEqual(masks['code'].tolist(), [0, 2, 3])

    def _bundle(self):
        # two sources, three candidates each; block_2 relative L2 is metric index 2 at stage index 6
        geometry = {'source_index': np.array([7, 7, 7, 9, 9, 9]), 'transition_code': np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8),
                    'label': np.array([1, 1, 1, 2, 2, 2]), 'candidate_prediction': np.array([1, 1, 1, 2, 2, 3]),
                    'candidate_true_class_margin': np.array([1.0, 0.5, -1.0, 2.0, 1.0, 0.0], dtype=np.float32),
                    'score_l2_change': np.array([0.1, 0.2, 0.9, 0.3, 0.2, 0.1], dtype=np.float32)}
        trace = np.zeros((6, 7, 6), dtype=np.float32); trace[:, 6, 2] = [0.0, 0.5, 0.7, 0.2, 0.4, 0.6]
        clean = {7: np.array([0, 3, 0, 1] + [0] * 31, dtype=np.float32), 9: np.array([0, 0, 2, 0] + [0] * 31, dtype=np.float32)}
        config = {'stage_names': ['stem', 'attention_1', 'local_1', 'block_1', 'attention_2', 'local_2', 'block_2'],
                  'metric_names': ['mean_abs_change', 'rms_change', 'relative_l2_change', 'cosine_distance', 'activity_flip_rate', 'mean_signed_change']}
        return dict(geometry=geometry, internal={'trace_metrics': trace}, clean_scores=clean, config=config)

    def test_final_block_descriptives_counts_exact_zero(self):
        result = final_block_descriptives(self._bundle())
        self.assertEqual(result['preserved_exact_zero'], 1); self.assertEqual(result['preserved_nonzero'], 4)
        row = next(r for r in result['table'] if r['outcome'] == 'class_preserved')
        self.assertEqual(row['candidates'], 5); self.assertEqual(row['sources'], 2)
        self.assertAlmostEqual(row['equal_source_mean'], (np.mean([0.0, 0.5, 0.7]) + np.mean([0.2, 0.4])) / 2)
        self.assertEqual(next(r for r in result['table'] if r['outcome'] == 'corrective')['candidates'], 0)

    def test_internal_output_correlations_use_clean_margin(self):
        result = internal_output_correlations(self._bundle())
        self.assertEqual(result['excluded_exact_zero_candidates'], 1)
        row = next(r for r in result['table'] if r['candidates'] == 'all_neighbors')
        self.assertEqual(row['sources'], 2); self.assertTrue(-1 <= row['signed_margin'] <= 1)

    def test_accuracy_decomposition(self):
        b = self._bundle()
        rows = [dict(source_index=7, unique_candidates=3, clean_correct=True), dict(source_index=9, unique_candidates=3, clean_correct=True)]
        result = accuracy_decomposition(b['geometry'], rows)
        self.assertEqual(result['equal_source_clean_accuracy'], 1.0)
        self.assertAlmostEqual(result['equal_source_neighbor_accuracy'], (1 + 2 / 3) / 2)
        self.assertEqual(result['corrective_minus_adverse'], -1)


if __name__ == '__main__':
    unittest.main()
