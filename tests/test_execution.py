"""Execution controls: alignment checks and the appendix conclusions (Tables 2, A.1-A.3)."""
import copy
import unittest
import numpy as np
from ssc_geometry import paths
from ssc_geometry.execution import analyze, score_predictions, validation_alignment, label_contrast
from ssc_geometry.io import load_npz, repository_root

ROOT = repository_root()


class ExecutionControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = analyze(ROOT, progress=lambda *args, **kwargs: None)
        cls.gpu = load_npz(ROOT / paths.EXECUTION_GPU_CONDITIONS)
        cls.cpu = load_npz(ROOT / paths.EXECUTION_CPU_SINGLETON)
        cls.isolated = load_npz(ROOT / paths.EXECUTION_QK_ISOLATION)

    def test_table_02(self):
        rows = {r['split']: r for r in self.report['table_02']}
        self.assertEqual(rows['Validation, batch 256']['correct'], 8617)
        self.assertEqual(rows['Validation, GPU singleton (audit path)']['correct'], 8592)
        self.assertEqual(rows['Validation, CPU singleton']['correct'], 8591)
        self.assertEqual((rows['Official test, batch 256']['correct'], rows['Official test, batch 256']['samples']), (17247, 20382))

    def test_isolation_removes_order_dependence_without_claiming_score_equivalence(self):
        contrasts = {(row['a'], row['b']): row for row in self.report['paired_validation_contrasts']}
        original = contrasts[('GPU_batch256', 'GPU_batch256_reversed')]
        isolated = contrasts[('GPU_qk_isolated_batch256', 'GPU_qk_isolated_batch256_reversed')]
        residual = contrasts[('GPU_padding_matched_singleton', 'GPU_qk_isolated_batch256')]
        self.assertEqual((original['changed_labels'], original['correct_to_wrong'], original['wrong_to_correct'], original['wrong_to_different_wrong']), (531, 205, 158, 168))
        self.assertEqual(isolated['changed_labels'], 0); self.assertEqual(isolated['score_difference']['exact_zero'], 9981)
        self.assertEqual(residual['changed_source_ids'], [7942]); self.assertEqual(residual['changed_classes'], {'7942': [27, 25]})
        self.assertGreater(residual['score_difference']['maximum'], 3.7); self.assertEqual(residual['score_difference']['above_1e_4'], 54)
        self.assertEqual(self.report['conditions']['GPU_singleton']['correct'], 8592)

    def test_device_replay_preserves_sources_but_has_explicit_cohort_limit(self):
        replay = self.report['class_changing_device_replay']
        self.assertEqual((replay['candidates'], replay['CPU_GPU_label_agreements'], replay['CPU_GPU_transition_agreements']), (25820, 25794, 25795))
        self.assertEqual(len(replay['canonical_adverse_sources']), 13)
        self.assertEqual(replay['adverse_sources']['CPU'], replay['canonical_adverse_sources'])
        self.assertEqual(replay['adverse_sources']['GPU'], replay['canonical_adverse_sources'])
        self.assertTrue(all(w['CPU_is_adverse'] and w['GPU_is_adverse'] and w['CPU_same_class'] for w in replay['first_canonical_adverse_witnesses']))
        self.assertEqual(replay['by_canonical_transition']['adverse']['CPU_same_type'], 5152)
        self.assertFalse(replay['originally_preserved_candidates_replayed']); self.assertFalse(self.report['new_model_inference'])

    def test_padding_gap_counts(self):
        bins = {b['padding_bins_min']: b for b in self.report['padding_gap_counts']}
        self.assertEqual((bins[0]['sources'], bins[0]['changed_labels']), (4057, 0))
        self.assertEqual((bins[1]['sources'], bins[1]['changed_labels']), (5004, 236))

    def test_reordered_source_metadata_is_rejected(self):
        cpu = copy.copy(self.cpu); cpu['source_index'] = cpu['source_index'][::-1]
        with self.assertRaisesRegex(ValueError, 'source order changed'):
            validation_alignment(self.gpu, cpu, self.isolated)

    def test_input_substitution_is_rejected(self):
        cpu = copy.copy(self.cpu); cpu['input_sha256'] = cpu['input_sha256'].copy(); cpu['input_sha256'][0] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Transformed inputs differ'):
            validation_alignment(self.gpu, cpu, self.isolated)

    def test_invalid_score_shapes_and_nonfinite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'expected'):
            score_predictions(np.zeros((2, 34)), 2, 'probe')
        scores = np.zeros((2, 35), dtype=np.float32); scores[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            score_predictions(scores, 2, 'probe')

    def test_label_contrast_partition(self):
        labels = np.array([0, 0, 0, 1]); a = np.array([0, 1, 1, 1]); b = np.array([1, 0, 2, 1])
        c = label_contrast(labels, a, b)
        self.assertEqual((c['changed_labels'], c['correct_to_wrong'], c['wrong_to_correct'], c['wrong_to_different_wrong']), (3, 1, 1, 1))


if __name__ == '__main__':
    unittest.main()
