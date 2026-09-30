"""Test finite-budget counting and the source composition behind replica results."""
import itertools
import unittest

import numpy as np

from ssc_geometry.io import repository_root
from ssc_geometry.retained_map_analyses import allocate_exact, detection, replica_analysis, search_analysis


class RetainedMapAnalysisTests(unittest.TestCase):
    def test_capped_focus_redistributes_budget_after_exhaustion(self):
        queries, overflow = allocate_exact(7, [3, 8], [1, 0], [10, 20])
        np.testing.assert_array_equal(queries, [3, 4])
        self.assertTrue(overflow)

    def test_equal_remainders_use_source_id_tie_break(self):
        queries, overflow = allocate_exact(4, [10, 10, 10], [1, 1, 1], [9, 2, 5])
        np.testing.assert_array_equal(queries, [1, 2, 1])
        self.assertFalse(overflow)

    def test_finite_detection_matches_literal_subset_enumeration(self):
        subsets = list(itertools.combinations(range(7), 3))
        expected = sum(bool({0, 1}.intersection(s)) for s in subsets) / len(subsets)
        result = detection(np.array([7]), np.array([2]), np.array([3]))
        self.assertAlmostEqual(result[0], expected, places=14)

    def test_frozen_rankings_have_capped_full_discovery_cost(self):
        result = search_analysis(repository_root())
        prefix = result['posthoc_full_discovery_prefixes'][0]
        self.assertEqual(prefix['policy'], 'runner_up')
        self.assertEqual(prefix['posthoc_prefix_per_source'], 5612)
        self.assertEqual(prefix['capped_queries'], 461380)
        self.assertEqual(prefix['sources_with_shorter_neighborhood'], 14)
        at100 = [r['actual_discovered_sources'] for r in result['deterministic_discovery']
                 if r['budget_per_source'] == 100]
        self.assertEqual(at100, [7, 7, 8])
        self.assertTrue(result['retrospective_policy_comparison'])

    def test_replica_gap_is_concentrated_and_reverses_in_common_correct_cohort(self):
        result = replica_analysis(repository_root())
        self.assertEqual(result['four_source_ids'], [914, 4499, 7289, 8134])
        self.assertEqual(result['four_sources_B_adverse'], 15760)
        self.assertEqual(result['replicas']['B']['adverse'], 15773)
        self.assertEqual(result['common_clean_correct_sources'], 67)
        self.assertLess(result['common_clean_correct_difference_percentage_points'], 0)


if __name__ == '__main__':
    unittest.main()
