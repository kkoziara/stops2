__author__ = 'Kamil Koziara'

import unittest
import numpy
from utils import HexGrid

class TestStopUtils(unittest.TestCase):

    def test_hex_grid(self):

        xctrs = [(0.0, 0.0), (2.0, 0.0), (1.0, 1.7320508075688774), (3.0, 1.7320508075688774)]
        xarr = numpy.array([[ 0.        ,  2.        ,  2.        ,  3.46410162],
                            [ 2.        ,  0.        ,  2.        ,  2.        ],
                            [ 2.        ,  2.        ,  0.        ,  2.        ],
                            [ 3.46410162,  2.        ,  2.        ,  0.        ]])

        grid = HexGrid(2, 2, 2)
        self.assertEqual([2, 2], grid.shape)
        self.assertEqual(xctrs, grid.xy)
        self.assertTrue(numpy.allclose(xarr, grid.adj_mat), "Distance arrays are not equal.")

if __name__ == '__main__':
    unittest.main()
