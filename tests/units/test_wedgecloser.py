import unittest
import numpy as np

from networkmodifier import WedgeCloser, ModificationType


# noinspection PyTypeChecker
class TestWedgeCloser(unittest.TestCase):

    def test_fit(self):
        # Test 1
        adj_mat = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]])
        wedges_mat = np.array([[0, 0, 1],
                               [0, 0, 0],
                               [1, 0, 0]])
        wedge_closer = WedgeCloser(ModificationType.RANDOM)
        wedge_closer.fit(adj_mat)
        self.assertListEqual(wedges_mat.tolist(), wedge_closer.wedges_mat.tolist())

        # Test 2
        adj_mat = np.array([[0, 1, 1, 1],
                            [1, 0, 1, 1],
                            [1, 1, 0, 0],
                            [1, 1, 0, 0]])
        wedges_mat = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 2],
                               [0, 0, 2, 0]])
        wedge_closer.fit(adj_mat)
        self.assertListEqual(wedges_mat.tolist(), wedge_closer.wedges_mat.tolist())

    def test_closing_random(self):
        # Test 1
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM)
        wc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 1, 0]
        adj_te = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM)
        wc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 1]
        adj_te = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM)
        wc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM)
        wc.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 5
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM)
        wc.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 6
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM)
        wc.fit(adj)
        wc.transform(m=1, types=types)
        wc.transform(m=2, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

    def test_closing_mono(self):
        # Test 1
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_MONO)
        wc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 1, 0]
        adj_te = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_MONO)
        wc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 1]
        adj_te = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_MONO)
        wc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_MONO)
        wc.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 5
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 1, 0],
                           [1, 0, 1, 1],
                           [1, 1, 0, 0],
                           [0, 1, 0, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_MONO)
        wc.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 6
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_MONO)
        wc.fit(adj)
        wc.transform(m=1, types=types)
        wc.transform(m=2, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

    def test_closing_bi(self):
        # Test 1
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 1]
        adj_te = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BI)
        wc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 1, 1]
        adj_te = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BI)
        wc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BI)
        wc.fit_transform(adj, m=1, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 0, 0],
                           [1, 0, 1, 1],
                           [0, 1, 0, 0],
                           [0, 1, 0, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BI)
        wc.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 5
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 0, 1],
                           [1, 0, 1, 1],
                           [0, 1, 0, 1],
                           [1, 1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BI)
        wc.fit_transform(adj, m=3, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 6
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 0, 1],
                           [1, 0, 1, 1],
                           [0, 1, 0, 1],
                           [1, 1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BI)
        wc.fit(adj)
        wc.transform(m=1, types=types)
        wc.transform(m=1, types=types)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

    def test_closing_random_biased(self):
        # Test 1
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        wc.fit_transform(adj, m=1, types=types, p_s=0.5)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 2
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 1]
        adj_te = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        wc.fit_transform(adj, m=1, types=types, p_s=0.5)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 3
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 1]
        adj_te = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        wc.fit_transform(adj, m=1, types=types, p_s=1)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 4
        adj = np.array([[0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
        types = [0, 0, 0]
        adj_te = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        wc.fit_transform(adj, m=1, types=types, p_s=0)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 5
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 0]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        wc.fit_transform(adj, m=3, types=types, p_s=0.5)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 6
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        wc.fit_transform(adj, m=3, types=types, p_s=0.5)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 7
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0]])
        types = [0, 0, 0, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        wc.fit(adj)
        wc.transform(m=1, types=types, gamma=0.5)
        wc.transform(m=2, types=types, gamma=0.5)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())

        # Test 8 (test for being refitted)
        adj = np.array([[0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0]])
        types = [0, 1, 0, 1]
        adj_te = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 1],
                           [1, 1, 0, 1],
                           [1, 1, 1, 0]])

        wc = WedgeCloser(modification_type=ModificationType.RANDOM_BIASED)
        wc.fit(adj)
        wc.transform(m=1, types=types, gamma=0.5)
        wc.transform(m=2, types=types, gamma=0.5)

        self.assertListEqual(adj_te.tolist(), wc.adj_mat.tolist())


if __name__ == '__main__':
    unittest.main()
