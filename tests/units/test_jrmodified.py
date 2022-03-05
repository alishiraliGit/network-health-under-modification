import unittest
import numpy as np

from networksimulator import JacksonRogersModifiedModel
from networkmeasure import integration


# noinspection PyTypeChecker
class TestJRModified(unittest.TestCase):

    def test_forward(self):
        # Test 1
        jr = JacksonRogersModifiedModel(n_s=1, n_d=0, n_f=0, alpha=1, ps=(1, 0))
        jr.forward(steps=1)

        adj_te = np.array([[0, 1, 1],
                           [0, 0, 0],
                           [0, 0, 0]])

        self.assertListEqual(adj_te.tolist(), jr.adj_mat.tolist())

        # Test 1.5
        jr.n_s = 2
        jr.forward(steps=1)

        adj_te = np.array([[0, 1, 1, 1],
                           [0, 0, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]])

        self.assertListEqual(adj_te.tolist(), jr.adj_mat.tolist())

        # Test 2
        jr = JacksonRogersModifiedModel(n_s=1, n_d=0, n_f=0, alpha=1, ps=(0, 1))
        jr.forward(steps=1)

        adj_te = np.array([[0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]])

        self.assertListEqual(adj_te.tolist(), jr.adj_mat.tolist())

        # Test 3
        jr = JacksonRogersModifiedModel(n_s=0, n_d=1, n_f=0, alpha=0.49, ps=(1, 0))  # Should work for alpha < 0.5
        jr.n_f = 1
        jr.forward(1)

        adj_te = np.array([[0, 1, 1],
                           [0, 0, 1],
                           [0, 0, 0]])

        self.assertListEqual(adj_te.tolist(), jr.adj_mat.tolist())


if __name__ == '__main__':
    unittest.main()
