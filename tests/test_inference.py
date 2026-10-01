"""Model replay helpers: forward-pass capture on a toy module (not SpikeSCR) and the output comparison contract."""
import unittest
import numpy as np
try:
    import torch
    from torch import nn
except ImportError:
    torch=None
from ssc_geometry.io import repository_root
from ssc_geometry.inference import compare_point, validate_cached_source


@unittest.skipIf(torch is None,'Optional PyTorch is not installed')
class ForwardKernelTests(unittest.TestCase):
    def setUp(self):
        from ssc_geometry import forward as fk
        self.fk=fk;self.previous=fk.SJ_FUNCTIONAL
        class Reset:
            calls=0
            @classmethod
            def reset_net(cls,model):cls.calls+=1
        self.reset=Reset;fk.SJ_FUNCTIONAL=Reset
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.stem=nn.Identity();self.boundary=nn.Identity()
            def forward(self,x,mask):
                x=self.stem(x.permute(1,0,2));x=self.boundary(x*2)
                return x
        self.model=Toy();self.stages=[{'stage':'stem','module':self.model.stem},{'stage':'boundary','module':self.model.boundary}]
    def tearDown(self):self.fk.SJ_FUNCTIONAL=self.previous
    def test_score_definition(self):
        x=torch.arange(24,dtype=torch.float32).reshape(4,1,6)
        np.testing.assert_allclose(self.fk.upstream_softmax_sum(x).numpy(),torch.softmax(x.float(),dim=2).sum(dim=0).numpy(),rtol=0,atol=0)
    def test_trace_and_replacement(self):
        x=torch.zeros((1,4,35));x[:,:,1]=2
        score,trace=self.fk.captured_model_forward(self.model,x,torch.device('cpu'),self.stages)
        z=x.clone();z[:,:,1]=0;z[:,:,2]=3
        patched=self.fk.patched_model_scores(self.model,z,torch.device('cpu'),self.stages[0],trace['stem'],1)
        np.testing.assert_array_equal(score,patched)
        self.assertEqual(len(self.model.stem._forward_hooks),0);self.assertGreaterEqual(self.reset.calls,4)
    def test_hooks_cleaned_after_failure(self):
        x=torch.full((1,4,35),float('nan'))
        with self.assertRaises(FloatingPointError):self.fk.captured_model_forward(self.model,x,torch.device('cpu'),self.stages)
        self.assertEqual(len(self.model.stem._forward_hooks),0);self.assertEqual(len(self.model.boundary._forward_hooks),0)
    def test_trace_metrics(self):
        clean=torch.tensor([[[1.,0.,2.]]]);changed=torch.tensor([[[0.,1.,2.]]])
        m=self.fk.singleton_trace_metrics(changed,clean)
        self.assertAlmostEqual(m[0],2/3,places=6);self.assertAlmostEqual(m[2],np.sqrt(2)/np.sqrt(5),places=6)
        self.assertAlmostEqual(m[4],2/3,places=6);self.assertAlmostEqual(m[5],0,places=6)
    def test_shape_mismatch(self):
        with self.assertRaises(RuntimeError):self.fk.singleton_trace_metrics(torch.zeros(2),torch.zeros(3))
    def test_ambiguous_batch_axis_rejected(self):
        with self.assertRaises(RuntimeError):self.fk.infer_trace_batch_axes({'test':torch.zeros(1,1,5)},1)
    def test_real_checkpoint_safely_loads(self):
        from ssc_geometry.integrity import verify_checkpoint
        report=verify_checkpoint(repository_root())
        self.assertEqual(report['status'],'PASS');self.assertEqual(report['selected_epoch_one_based'],282)
        self.assertFalse(report['model_forward_executed'])


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
