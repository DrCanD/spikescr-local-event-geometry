"""Synthetic contract tests for comparison logic, not new model experiments."""
import unittest
import numpy as np
from ssc_geometry.inference import compare_point, validate_cached_source


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
    def test_missing_required_patch_rejected(self):
        result=compare_point(self.scores,self.traces,None,7,0,self.b,{(7,0):0})
        self.assertIn('missing_patch_scores',result['errors'])
    def test_trace_broadcast_rejected(self):
        with self.assertRaisesRegex(ValueError,'Invalid observed trace'):
            self.check(trace=np.zeros(6,dtype=np.float32))


class ResumeCoverageTests(unittest.TestCase):
    def setUp(self):
        self.clean=np.zeros(35,dtype=np.float32);self.clean[1]=5
        self.cached={'source_index':np.array(7),'candidate_index':np.array([0,1]),
            'patch_candidate_index':np.array([1]),'scores':np.tile(self.clean,(2,1)),
            'trace_metrics':np.zeros((2,7,6),dtype=np.float32),
            'patch_scores':np.tile(self.clean,(1,7,1)),'clean_scores':self.clean.copy()}
        self.summary={'source_index':7,'candidates':2,'patched_candidates':1}
    def validate(self):
        return validate_cached_source(self.cached,self.summary,7,[0,1],[1],self.clean,1e-5)
    def test_complete_source(self):self.assertEqual(list(self.validate()),[1])
    def test_missing_replacement_rejected(self):
        self.cached['patch_candidate_index']=np.array([],dtype=np.int32)
        self.cached['patch_scores']=np.empty((0,7,35),dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'replacement coverage'):self.validate()
    def test_summary_count_mismatch_rejected(self):
        self.summary['candidates']=725070
        with self.assertRaisesRegex(ValueError,'summary coverage'):self.validate()
    def test_changed_clean_scores_rejected(self):
        self.cached['clean_scores'][1]=6
        with self.assertRaisesRegex(ValueError,'clean score parity'):self.validate()
    def test_truncated_trace_rejected(self):
        self.cached['trace_metrics']=self.cached['trace_metrics'][:1]
        with self.assertRaisesRegex(ValueError,'cached output'):self.validate()

if __name__=='__main__':unittest.main()
