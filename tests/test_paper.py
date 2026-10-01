"""The manuscript cross-check: path resolution, the rounding rule, and the expected-values file."""
import unittest
from ssc_geometry import paths
from ssc_geometry.io import repository_root, read_json
from ssc_geometry.tables import resolve, check_cell

ROOT = repository_root()


class CrossCheckTests(unittest.TestCase):
    def test_resolve_indexes_and_selectors(self):
        data = {'a': {'rows': [{'k': 'x', 'v': 1}, {'k': 'y', 'v': 2}], 'list': [10, 20]}}
        self.assertEqual(resolve(data, 'a.rows[k=y].v'), 2)
        self.assertEqual(resolve(data, 'a.list[1]'), 20)
        with self.assertRaises(KeyError):
            resolve(data, 'a.rows[k=z].v')

    def test_rounding_rule(self):
        self.assertEqual(check_cell({'x': 0.86334034}, dict(id='a', location='t', path='x', value=86.3340, decimals=4, scale=100))['status'], 'PASS')
        self.assertEqual(check_cell({'x': 0.86335034}, dict(id='a', location='t', path='x', value=86.3340, decimals=4, scale=100))['status'], 'FAIL')
        self.assertEqual(check_cell({'x': 8617}, dict(id='a', location='t', path='x', value=8617))['status'], 'PASS')
        self.assertEqual(check_cell({'x': 8617.4}, dict(id='a', location='t', path='x', value=8617))['status'], 'FAIL')
        self.assertEqual(check_cell({'x': [0.02301, 2.45068]}, dict(id='a', location='t', path='x', value=[0.0230, 2.4507], decimals=4))['status'], 'PASS')
        self.assertEqual(check_cell({}, dict(id='a', location='t', path='x', value=1))['status'], 'FAIL')

    def test_expected_values_file_is_well_formed(self):
        spec = read_json(ROOT / paths.EXPECTED_VALUES)
        ids = [c['id'] for c in spec['cells']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(len(ids), 400)
        for cell in spec['cells']:
            self.assertTrue(cell['path'].split('.')[0] in ('census', 'search', 'replicas', 'execution'), cell['id'])
            self.assertIn('location', cell)


if __name__ == '__main__':
    unittest.main()
