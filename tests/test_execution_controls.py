"""Check evidence alignment and the retained execution-control conclusions."""
import copy
import unittest

import numpy as np

from ssc_geometry.execution_controls import analyze, score_predictions, validation_alignment
from ssc_geometry.io import load_npz, repository_root

ROOT = repository_root()


class ExecutionControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = analyze(ROOT, progress=lambda *args, **kwargs: None)
        folder = ROOT / 'data/execution_controls'
        cls.gpu = load_npz(folder / 'gpu_validation_predictions.npz')
        cls.cpu = load_npz(folder / 'cpu_validation_predictions.npz')
        cls.isolated = load_npz(folder / 'qk_isolation_validation.npz')

    def test_isolation_removes_order_dependence_without_claiming_score_equivalence(self):
        contrasts = {(row['a'], row['b']): row for row in self.report['paired_validation_contrasts']}
        original = contrasts[('GPU_batch256', 'GPU_batch256_reversed')]
        isolated = contrasts[('GPU_qk_isolated_batch256', 'GPU_qk_isolated_batch256_reversed')]
        residual = contrasts[('GPU_padding_matched_singleton', 'GPU_qk_isolated_batch256')]
        self.assertEqual(original['changed_labels'], 531)
        self.assertEqual(isolated['changed_labels'], 0)
        self.assertEqual(isolated['score_difference']['exact_zero'], 9981)
        self.assertEqual(residual['changed_source_ids'], [7942])
        self.assertGreater(residual['score_difference']['maximum'], 3.7)
        self.assertEqual(self.report['conditions']['GPU_singleton']['correct'], 8592)

    def test_device_replay_preserves_sources_but_has_explicit_cohort_limit(self):
        replay = self.report['class_changing_device_replay']
        self.assertEqual(replay['candidates'], 25820)
        self.assertEqual(replay['CPU_GPU_label_agreements'], 25794)
        self.assertEqual(replay['CPU_GPU_transition_agreements'], 25795)
        self.assertEqual(len(replay['canonical_adverse_sources']), 13)
        self.assertEqual(replay['adverse_sources']['CPU'], replay['canonical_adverse_sources'])
        self.assertEqual(replay['adverse_sources']['GPU'], replay['canonical_adverse_sources'])
        self.assertTrue(all(w['CPU_is_adverse'] and w['GPU_is_adverse']
                            for w in replay['first_canonical_adverse_witnesses']))
        self.assertFalse(replay['originally_preserved_candidates_replayed'])
        self.assertFalse(self.report['new_model_inference'])

    def test_reordered_source_metadata_is_rejected(self):
        cpu = copy.copy(self.cpu)
        cpu['source_index'] = cpu['source_index'][::-1]
        with self.assertRaisesRegex(ValueError, 'source order changed'):
            validation_alignment(self.gpu, cpu, self.isolated)

    def test_input_substitution_is_rejected(self):
        cpu = copy.copy(self.cpu)
        cpu['input_sha256'] = cpu['input_sha256'].copy()
        cpu['input_sha256'][0] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Transformed inputs differ'):
            validation_alignment(self.gpu, cpu, self.isolated)

    def test_invalid_score_shapes_and_nonfinite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'expected'):
            score_predictions(np.zeros((2, 34)), 2, 'probe')
        scores = np.zeros((2, 35), dtype=np.float32)
        scores[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            score_predictions(scores, 2, 'probe')


if __name__ == '__main__':
    unittest.main()
