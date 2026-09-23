"""Synthetic contract tests for comparison logic, not new model experiments."""
import unittest
import numpy as np
from ssc_geometry.inference import compare_point


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        clean=np.zeros(35,dtype=np.float32);clean[1]=5
        self.scores=np.zeros(35,dtype=np.float32);self.scores[1]=3;self.scores[2]=6
        d=self.scores-clean;cos=(self.scores@clean)/(np.linalg.norm(self.scores)*np.linalg.norm(clean))
        g={k:np.array([v],dtype=np.float32) for k,v in dict(candidate_top1_margin=3,candidate_clean_class_margin=-3,candidate_true_class_margin=-3,
            delta_clean_class_score=-2,delta_true_class_score=-2,score_l2_change=np.linalg.norm(d),score_linf_change=6,score_cosine_distance=1-cos).items()}
        g.update(label=np.array([1]),clean_prediction=np.array([1]),candidate_prediction=np.array([2]),candidate_index=np.array([0]))
        self.traces=np.zeros((7,6),dtype=np.float32);self.patches=np.tile(clean,(7,1))
        self.b={'geometry':g,'internal':{'trace_metrics':self.traces[None], 'patch_scores':self.patches[None],'patch_prediction':self.patches.argmax(axis=1)[None]},
            'config':{'source_score_atol':1e-5,'replay_trace_atol':1e-5},'clean_scores':{7:clean}}
    def check(self,score=None,trace=None,patch=None):
        return compare_point(self.scores if score is None else score,self.traces if trace is None else trace,self.patches if patch is None else patch,7,0,self.b,{(7,0):0})
    def test_valid_record(self):self.assertEqual(self.check()['status'],'PASS')
    def test_wrong_prediction_rejected(self):
        score=self.scores.copy();score[0]=7
        self.assertIn('candidate_prediction',self.check(score=score)['errors'])
    def test_wrong_trace_rejected(self):
        trace=self.traces.copy();trace[0,0]=.001
        self.assertIn('trace_metrics',self.check(trace=trace)['errors'])
    def test_wrong_patch_rejected(self):
        patch=self.patches.copy();patch[0,0]=.1
        self.assertIn('patch_scores',self.check(patch=patch)['errors'])

if __name__=='__main__':unittest.main()
