import unittest
import numpy as np

from networkmodifier import EdgeDropper, ModificationType


# noinspection PyTypeChecker
class TestEdgeDropper(unittest.TestCase):

    def test_dropping_random(self):
        # Test 1
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM)
        ed.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 1.5
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM)
        ed.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 1]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM)
        ed.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [1, 0, 1]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM)
        ed.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 1]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM)
        ed.fit(adj)
        ed.transform(m=1, types=types)
        ed.transform(m=1, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

    def test_dropping_mono(self):
        # Test 1
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_MONO)
        ed.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 1]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_MONO)
        ed.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [1, 0, 1]
        adj_te = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_MONO)
        ed.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_MONO)
        ed.fit(adj)
        ed.transform(m=1, types=types)
        ed.transform(m=1, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

    def test_dropping_bi(self):
        # Test 1
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_BI)
        ed.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 1]
        adj_te = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_BI)
        ed.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [1, 0, 1]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_BI)
        ed.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 1, 0]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_BI)
        ed.fit(adj)
        ed.transform(m=1, types=types)
        ed.transform(m=1, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

    def test_dropping_random_biased(self):
        # Test 1
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_BIASED)
        ed.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 1]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_BIASED)
        ed.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 2.5
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 1]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_BIASED)
        ed.fit_transform(adj, m=2, types=types, p_s=1)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 2.75
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 1]
        adj_te = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_BIASED)
        ed.fit_transform(adj, m=2, types=types, p_s=0)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [1, 0, 1]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_BIASED)
        ed.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 1, 0]
        adj_te = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

        ed = EdgeDropper(modification_type=ModificationType.RANDOM_BIASED)
        ed.fit(adj)
        ed.transform(m=1, types=types)
        ed.transform(m=1, types=types)

        self.assertListEqual(adj_te.tolist(), ed.adj_mat.tolist())


if __name__ == '__main__':
    unittest.main()
