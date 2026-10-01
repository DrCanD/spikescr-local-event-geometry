"""Two count-readout replicas: Table 9 and the paired comparisons (§4.7)."""
import unittest
import numpy as np
from ssc_geometry.io import repository_root
from ssc_geometry.replicas import analyze, paired_bootstrap


class ReplicaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = analyze(repository_root())

    def test_table_09(self):
        rows = {r['outcome']: r for r in self.result['table_09']}
        self.assertEqual((rows['T3: class changed']['replica_A'], rows['T3: class changed']['replica_B']), (3564, 18566))
        self.assertEqual((rows['Adverse neighbors (subset of T3)']['replica_A'], rows['Adverse neighbors (subset of T3)']['replica_B']), (790, 15773))
        self.assertEqual((rows['Correct and robust sources (of 100)']['replica_A'], rows['Correct and robust sources (of 100)']['replica_B']), (67, 70))

    def test_gap_is_concentrated_in_four_sources_and_reverses_on_the_common_cohort(self):
        c = self.result['comparison']
        self.assertEqual(c['four_source_ids'], [914, 4499, 7289, 8134])
        self.assertEqual((c['four_sources_T3']['A'], c['four_sources_T3']['B']), (369, 15760))
        self.assertEqual((c['other_sources_T3']['A'], c['other_sources_T3']['B']), (3195, 2806))
        self.assertEqual((c['common_clean_correct_adverse']['A'], c['common_clean_correct_adverse']['B']), (248, 13))
        self.assertLess(c['common_clean_correct_difference_percentage_points'], 0)

    def test_paired_bootstrap_is_seeded(self):
        b = self.result['comparison']['paired_bootstrap']['full_panel']
        self.assertAlmostEqual(b['T3_rate_difference_B_minus_A_percentage_points'], 1.1896, places=4)
        self.assertAlmostEqual(b['T3_rate_difference_ci95'][0], 0.0230, places=4)
        self.assertAlmostEqual(b['T3_rate_difference_ci95'][1], 2.4507, places=4)

    def test_bootstrap_respects_label_quotas(self):
        rows = [dict(source_index=i, label=i % 2, A_T3_source_rate=0.0, B_T3_source_rate=float(i % 2), A_adverse_source_rate=0.0, B_adverse_source_rate=0.0) for i in range(6)]
        cohorts = dict(cohort_order=['all'], cohorts={'all': dict(source_ids=list(range(6)), sources=6)})
        result = paired_bootstrap(dict(paired=rows), cohorts, 50, 1)
        # with fixed per-label quotas the resampled mean of a label-constant rate is constant
        self.assertEqual(result['all']['T3_rate_difference_ci95'], [50.0, 50.0])
        self.assertEqual(result['all']['label_strata'], 2)


if __name__ == '__main__':
    unittest.main()
