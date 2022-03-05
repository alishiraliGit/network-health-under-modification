import unittest
import numpy as np

from utils.math_tools import freq2vec, vec2freq, one_hot


class TestMathTools(unittest.TestCase):

    def test_freq2vec(self):
        self.assertListEqual([0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3], freq2vec([1, 3, 2, 5]).tolist())

    def test_vec2freq(self):
        items, freqs = vec2freq([0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3])
        self.assertListEqual([0, 1, 2, 3], items.tolist())
        self.assertListEqual([1, 3, 2, 5], freqs.tolist())

    # noinspection PyTypeChecker
    def test_one_hot(self):
        types = [0, 0, 1, 2, 0]
        u_mat = np.array([[1, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]])
        self.assertListEqual(u_mat.tolist(), one_hot(types).tolist())


if __name__ == '__main__':
    unittest.main()
