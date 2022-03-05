import unittest
import numpy as np

from networkmodifier import EdgeCloser, ModificationType


# noinspection PyTypeChecker
class TestEdgeCloser(unittest.TestCase):

    def test_closing_random(self):
        # Test 1
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM)
        ec.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 1.5
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM)
        ec.fit_transform(adj, m=4, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM)
        ec.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 1, 1, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM)
        ec.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 1, 1, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM)
        ec.fit(adj)
        ec.transform(m=1, types=types)
        ec.transform(m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

    def test_closing_mono(self):
        # Test 1
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_MONO)
        ec.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 1, 0],
                           [1, 0, 1, 1],
                           [1, 1, 0, 0],
                           [0, 1, 0, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_MONO)
        ec.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 1, 1, 1]
        adj_te = np.array([[0, 1, 0, 0],
                           [1, 0, 1, 1],
                           [0, 1, 0, 1],
                           [0, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_MONO)
        ec.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [1, 0, 1, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_MONO)
        ec.fit(adj)
        ec.transform(m=1, types=types)
        ec.transform(m=2, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

    def test_closing_bi(self):
        # Test 1
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 0, 0],
                           [1, 0, 1, 1],
                           [0, 1, 0, 0],
                           [0, 1, 0, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_BI)
        ec.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 0, 1],
                           [1, 0, 1, 1],
                           [0, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_BI)
        ec.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 1, 1, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 0],
                           [1, 1, 0, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_BI)
        ec.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 0, 1],
                           [1, 0, 1, 1],
                           [0, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_BI)
        ec.fit(adj)
        ec.transform(m=1, types=types)
        ec.transform(m=1, types=types)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

    def test_closing_random_biased(self):
        # Test 1
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        ec.fit_transform(adj, m=3, types=types, p_s=0.5)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        ec.fit_transform(adj, m=3, types=types, p_s=0.5)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 2.5
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 1, 0],
                           [1, 0, 1, 1],
                           [1, 1, 0, 0],
                           [0, 1, 0, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        ec.fit_transform(adj, m=3, types=types, p_s=1)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 2.75
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 0, 1],
                           [1, 0, 1, 1],
                           [0, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        ec.fit_transform(adj, m=3, types=types, p_s=0)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 1, 1, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        ec.fit_transform(adj, m=3, types=types, p_s=0.5)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 1, 1, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        ec = EdgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        ec.fit(adj)
        ec.transform(m=1, types=types, p_s=0.5)
        ec.transform(m=2, types=types, p_s=0.5)

        self.assertListEqual(adj_te.tolist(), ec.adj_mat.tolist())


if __name__ == '__main__':
    unittest.main()
