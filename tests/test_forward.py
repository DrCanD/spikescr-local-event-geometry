"""These CPU tests use a small explicitly labelled test double, not SpikeSCR."""
import unittest
import numpy as np
try:
    import torch
    from torch import nn
except ImportError:
    torch=None
from ssc_geometry.io import repository_root


@unittest.skipIf(torch is None,'Optional PyTorch is not installed')
class ForwardKernelTests(unittest.TestCase):
    def setUp(self):
        from ssc_geometry import _forward_kernel as fk
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

if __name__=='__main__':unittest.main()
