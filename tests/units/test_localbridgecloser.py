import unittest
import numpy as np

from networkmodifier import LocalBridgeCloser, ModificationType


# noinspection PyTypeChecker
class TestLocalBridgeCloser(unittest.TestCase):

    def test_closing_random(self):
        # Test 1
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        lbc = LocalBridgeCloser(modification_type=ModificationType.RANDOM)
        lbc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]])

        lbc = LocalBridgeCloser(modification_type=ModificationType.RANDOM)
        lbc.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]])

        lbc = LocalBridgeCloser(modification_type=ModificationType.RANDOM)
        lbc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())

    def test_closing_mono(self):
        # Test 1
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        lbc = LocalBridgeCloser(modification_type=ModificationType.RANDOM_MONO)
        lbc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]])

        lbc = LocalBridgeCloser(modification_type=ModificationType.RANDOM_MONO)
        lbc.fit_transform(adj, m=2, types=types)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 0, 0],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0]])

        lbc = LocalBridgeCloser(modification_type=ModificationType.RANDOM_MONO)
        lbc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())

    def test_closing_bi(self):
        # Test 1
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 0, 0],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0]])

        lbc = LocalBridgeCloser(modification_type=ModificationType.RANDOM_BI)
        lbc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]])

        lbc = LocalBridgeCloser(modification_type=ModificationType.RANDOM_BI)
        lbc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())

    def test_closing_random_biased(self):
        # Test 1
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]])

        lbc = LocalBridgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        lbc.fit_transform(adj, m=1, types=types, p_s=0.5)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 0, 1],
                           [1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [1, 0, 1, 0]])

        lbc = LocalBridgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        lbc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 1],
                        [0, 0, 0, 1, 0]])
        types = [0, 0, 0, 0, 1]
        adj_te = np.array([[0, 1, 0, 1, 1],
                           [1, 0, 1, 0, 1],
                           [0, 1, 0, 1, 0],
                           [1, 0, 1, 0, 1],
                           [1, 1, 0, 1, 0]])

        lbc = LocalBridgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        lbc.fit_transform(adj, m=3, types=types, p_s=0.5)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 1],
                        [0, 0, 0, 1, 0]])
        types = [0, 0, 0, 0, 1]
        adj_te = np.array([[0, 1, 0, 1, 0],
                           [1, 0, 1, 0, 0],
                           [0, 1, 0, 1, 0],
                           [1, 0, 1, 0, 1],
                           [0, 0, 0, 1, 0]])

        lbc = LocalBridgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        lbc.fit(adj)

        lbc.transform(m=1, types=types, p_s=1)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())

        adj_te = np.array([[0, 1, 0, 1, 0],
                           [1, 0, 1, 0, 1],
                           [0, 1, 0, 1, 0],
                           [1, 0, 1, 0, 1],
                           [0, 1, 0, 1, 0]])

        lbc.transform(m=1, types=types, p_s=0)

        self.assertListEqual(adj_te.tolist(), lbc.adj_mat.tolist())


if __name__ == '__main__':
    unittest.main()
