import unittest
import numpy as np

from networksimulator import StochasticBlockModel


class TestSBM(unittest.TestCase):

    def test_types(self):
        sbm = StochasticBlockModel(p=0.8, q=0.1, k=4, ns=(1, 3, 2, 5))
        self.assertListEqual([0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3], sbm.types.tolist())

    def test_init_probability(self):
        sbm = StochasticBlockModel(p=0.7, q=0.1, k=2, ns=(2, 1))
        p_mat = np.array([[0.7, 0.7, 0.1],
                          [0.7, 0.7, 0.1],
                          [0.1, 0.1, 0.7]])
        self.assertListEqual(p_mat.tolist(), sbm.p_mat.tolist())


if __name__ == '__main__':
    unittest.main()
