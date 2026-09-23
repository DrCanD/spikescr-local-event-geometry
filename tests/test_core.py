import tempfile
import unittest
from pathlib import Path
import numpy as np
from ssc_geometry.core import transform_events, enumerate_moves, move_one_count, classify_transitions, validate_input
from ssc_geometry._kernel import literal_event_transform, rank_average, benjamini_hochberg, source_signflip_test
from ssc_geometry.io import fresh_output, read_numeric_csv, compare_internal_statistics
from ssc_geometry.integrity import checked_path


class OperatorTests(unittest.TestCase):
    def test_terminal_boundary_rule(self):
        x=transform_events(np.array([0.,.005,.010]),np.array([0,0,0]))
        self.assertEqual(x.shape,(2,140));self.assertEqual(x[:,0].tolist(),[1,2])
    def test_spatial_integration(self):
        x=transform_events(np.zeros(4),np.array([0,4,5,699]))
        self.assertEqual(x[0,0],2);self.assertEqual(x[0,1],1);self.assertEqual(x[0,139],1)
    def test_literal_transform(self):
        rng=np.random.default_rng(86);t=np.sort(rng.uniform(0,1,350));u=rng.integers(0,700,350)
        np.testing.assert_array_equal(transform_events(t,u),literal_event_transform(t,u,700,5,5))
    def test_empty_events(self):
        x=transform_events(np.array([]),np.array([],dtype=int));self.assertEqual(x.shape,(1,140));self.assertEqual(int(x.sum()),0)
    def test_invalid_events(self):
        for t,u in [([0,np.nan],[1,2]),([1,0],[1,2]),([0],[700]),([0],[1.5]),([0],[np.inf]),([[0]],[1])]:
            with self.subTest(t=t,u=u),self.assertRaises((ValueError,RuntimeError)):transform_events(np.array(t),np.array(u))
    def test_overflow(self):
        with self.assertRaises(OverflowError):transform_events(np.zeros(65536),np.zeros(65536,dtype=int))
    def test_candidate_order_and_multiplicity(self):
        x=np.zeros((3,140),dtype=np.uint16);x[0,2]=3;x[1,2]=2
        a,b,f,m=enumerate_moves(x)
        self.assertEqual(list(zip(a,b,f,m)),[(0,1,2,3),(1,0,2,2),(1,2,2,2)])
    def test_collision_allowed_and_source_unchanged(self):
        x=np.zeros((3,140),dtype=np.uint16);x[0,2]=3;x[1,2]=2;original=x.copy()
        y=move_one_count(x,0,1,2)
        self.assertEqual(y[:,2].tolist(),[2,3,0]);np.testing.assert_array_equal(x,original)
        self.assertEqual(x.sum(),y.sum());self.assertEqual(np.count_nonzero(x!=y),2)
    def test_distinct_candidate_tensors(self):
        x=np.zeros((4,140),dtype=np.uint16);x[:,0]=[1,3,2,4]
        a,b,f,_=enumerate_moves(x);images={move_one_count(x,int(t),int(v),int(c)).tobytes() for t,v,c in zip(a,b,f)}
        self.assertEqual(len(images),len(a))
    def test_invalid_moves(self):
        x=np.zeros((3,140),dtype=np.uint16);x[0,0]=1
        for args in [(0,2,0),(0,-1,0),(0,1,140),(1,2,0),(0.0,1,0),(True,2,0)]:
            with self.subTest(args=args),self.assertRaises(ValueError):move_one_count(x,*args)
    def test_invalid_count_tensor(self):
        for x in [np.zeros((0,140)),np.zeros((2,139)),np.full((2,140),-.1),np.full((2,140),.1),np.full((2,140),np.nan),np.ones((2,140),dtype=bool)]:
            with self.assertRaises(ValueError):validate_input(x)
    def test_transitions_all_four_types(self):
        label=np.array([1,1,1,1]);clean=np.array([1,1,2,2]);changed=np.array([1,2,1,3])
        np.testing.assert_array_equal(classify_transitions(label,clean,changed),[0,1,2,3])
    def test_invalid_class_ids(self):
        with self.assertRaises(ValueError):classify_transitions(np.array([35]),np.array([0]),np.array([1]))
        with self.assertRaises(ValueError):classify_transitions(np.array([1.5]),np.array([0]),np.array([1]))
    def test_ties_and_fdr(self):
        np.testing.assert_array_equal(rank_average(np.array([3,1,1,5])),[3,1.5,1.5,4])
        np.testing.assert_allclose(benjamini_hochberg([.01,.04,.03]),[.03,.04,.04])
    def test_random_seed_repeatability(self):
        a=source_signflip_test(np.array([.1,.3,-.2]),1000,33);b=source_signflip_test(np.array([.1,.3,-.2]),1000,33)
        self.assertEqual(a,b)


class IOTests(unittest.TestCase):
    def test_output_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)/'repo';r.mkdir()
            for name in ('data','configs','docs','src'): (r/name).mkdir()
            for path in [r,r.parent,r/'data/new',r/'src/new']:
                with self.assertRaises(ValueError):fresh_output(r,path)
            out=fresh_output(r,r/'outputs/new');self.assertTrue(out.is_dir())
            with self.assertRaises(FileExistsError):fresh_output(r,out)
    def test_manifest_path_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)
            for name in ('../x','/tmp/x','a\\b','C:x','a\nfoo'):
                with self.assertRaises(ValueError):checked_path(r,name)
            self.assertEqual(checked_path(r,'data/a.npz'),r/'data/a.npz')
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);(r/'real').write_text('x')
            try:(r/'alias').symlink_to(r/'real')
            except OSError:self.skipTest('No symlink support')
            with self.assertRaises(ValueError):checked_path(r,'alias')
    def test_boolean_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.csv';p.write_text('clean_correct,value\nTrue,1\nFalse,2\n1,3\n0,4\n')
            self.assertEqual([v['clean_correct'] for v in read_numeric_csv(p)],[True,False,True,False])
    def test_invalid_boolean_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.csv';p.write_text('clean_correct\n2\n')
            with self.assertRaises(ValueError):read_numeric_csv(p)
    def test_float_tolerance_does_not_relax_pvalues(self):
        a={'source_level_inference':{'transition_vs_preserved':{'adverse':{'stem':{'two_sided_signflip_p':.001}}}}}
        b={'source_level_inference':{'transition_vs_preserved':{'adverse':{'stem':{'two_sided_signflip_p':.0010001}}}}}
        self.assertEqual(compare_internal_statistics(a,b)['status'],'FAIL')
    def test_float_tolerance_reports_rounding(self):
        a={'trace_group_means_unique':{'a':.5+1e-7}};b={'trace_group_means_unique':{'a':.5}}
        result=compare_internal_statistics(a,b);self.assertEqual(result['status'],'PASS');self.assertEqual(len(result['differences']),1)

if __name__=='__main__':unittest.main()
