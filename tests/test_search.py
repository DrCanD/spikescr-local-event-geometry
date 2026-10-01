"""Finite-budget coverage and the frozen gradient rankings (Table 4, §4.3)."""
import itertools
import unittest
import numpy as np
from ssc_geometry.io import repository_root
from ssc_geometry.search import allocate_exact, detection, analyze, table_04


class AllocationTests(unittest.TestCase):
    def test_capped_focus_redistributes_budget_after_exhaustion(self):
        queries, overflow = allocate_exact(7, [3, 8], [1, 0], [10, 20])
        np.testing.assert_array_equal(queries, [3, 4]); self.assertTrue(overflow)

    def test_equal_remainders_use_source_id_tie_break(self):
        queries, overflow = allocate_exact(4, [10, 10, 10], [1, 1, 1], [9, 2, 5])
        np.testing.assert_array_equal(queries, [1, 2, 1]); self.assertFalse(overflow)

    def test_finite_detection_matches_literal_subset_enumeration(self):
        subsets = list(itertools.combinations(range(7), 3))
        expected = sum(bool({0, 1}.intersection(s)) for s in subsets) / len(subsets)
        self.assertAlmostEqual(detection(np.array([7]), np.array([2]), np.array([3]))[0], expected, places=14)


class SearchPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze(repository_root())

    def test_table_04(self):
        rows = {r['candidate_queries']: r for r in table_04(self.result)}
        self.assertAlmostEqual(rows[8400]['uniform_expected'], 7.61, places=2)
        self.assertAlmostEqual(rows[8400]['lowest_25_percent_expected'], 9.42, places=2)
        self.assertEqual((rows[8400]['runner_up_discovered'], rows[8400]['max_directional_gain_discovered'], rows[8400]['max_predicted_target_margin_discovered']), (7, 7, 8))
        self.assertEqual((rows[84000]['runner_up_discovered'], rows[84000]['max_directional_gain_discovered'], rows[84000]['max_predicted_target_margin_discovered']), (11, 10, 11))

    def test_frozen_rankings_have_capped_full_discovery_cost(self):
        prefix = self.result['posthoc_full_discovery_prefixes'][0]
        self.assertEqual(prefix['policy'], 'runner_up')
        self.assertEqual((prefix['posthoc_prefix_per_source'], prefix['capped_queries'], prefix['sources_with_shorter_neighborhood']), (5612, 461380, 14))
        self.assertTrue(self.result['retrospective_policy_comparison']); self.assertFalse(self.result['new_model_inference'])

    def test_sparse_tail_source(self):
        c = next(c for c in self.result['finite_sampling_coverage'] if c['ranking'] == 'margin_per_horizon' and c['policy'] == 'lowest_50_percent' and c['total_candidate_queries'] == 84000)
        self.assertEqual(c['vulnerable_sources_in_lowest_half'], 13); self.assertEqual(c['source1627_rank'], 42)
        self.assertAlmostEqual(c['expected_detected_sources'], 10.83, places=2)

    def test_witnesses_agree_with_rankings(self):
        self.assertEqual(self.result['witness_verification']['runner_up']['rows'], 13)
        self.assertEqual(self.result['witness_verification']['all_target']['rows'], 26)


if __name__ == '__main__':
    unittest.main()
