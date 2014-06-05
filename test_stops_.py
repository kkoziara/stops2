__author__ = 'Kamil Koziara'

import unittest
import numpy
import pyopencl as cl
from utils import generate_pop, HexGrid, draw_hex_grid
from stops_ import Stops2

secretion = numpy.array([5, 6, 7])
reception = numpy.array([3, 4, 2])
receptors = numpy.array([1, -1, -1])
bound=numpy.array([1,1,1,1,1,1,1,1,0])

base1=numpy.array([0,0,1,0,0,0,0,0,0])
base2=numpy.array([0,0,0,0,0,0,0,0,0])


trans_mat = numpy.array([[0,-10,0,0,0,0,10,0,0], #notch
                         [0,0,0,0,0,1,0,0,0], #Delta
                         [0,1,0,0,0,0,0,0.1,0], #basal
                         [0.005,0,0,0,0,0,0,0,0], #delta receptor
                         [-10,0,0,0,0,0,0,0,0], #notch receptor
                         [0,0,0,0,0,0,0,0,0], #ligand_delta
                         [0,0,0,0,0,0,0,0,0], #ligand_notch
                         [0,0,0,0,0,0,0,0,0], #ligand_basal
                         [0,0,0,0,0,0,0,0,0] #divdie dummy
                        ])

init_pop = generate_pop([(3, base1), (6, base2)])
grid = HexGrid(3, 3, 1)

class NumpyStopsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(NumpyStopsTest, self).__init__(*args, **kwargs)

    def test_each_step_with_constant_probs(self):
        s2 = Stops2(trans_mat, init_pop, grid.adj_mat, range(9),
               bound, secretion, reception, receptors, secr_amount=6, leak=0, max_con=6, opencl=True,
               asym_id=-1, div_id=-1, die_id=-1)
        s2.__random = lambda x: numpy.zeros(x) + 0.5


class PartialKernelsTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(PartialKernelsTest, self).__init__(*args, **kwargs)
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags

    def test_mat_mul(self):
        with open("mat_mul_kernel.c") as f:
            r_kernel = f.read()
            shape = (5, 10, 15)
            A = numpy.random.random(shape[:2]).astype(numpy.float32)
            B = numpy.random.random(shape[1:]).astype(numpy.float32)
            C = numpy.zeros((shape[0], shape[2])).astype(numpy.float32)
            bA = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=A)
            bB = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=B)
            bC = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=C)
            prog = cl.Program(self.ctx, r_kernel).build()
            prog.mat_mul(self.queue, (shape[0], shape[2]), None, bC, bA, bB, numpy.int32(shape[1]))
            cl.enqueue_copy(self.queue, C, bC)
            rC = A.dot(B)
            self.assertTrue(numpy.allclose(C, rC), "Multiplication failed")

    def test_ranlux(self):
        with open("ranlux_random.c") as f:
            r_kernel = f.read()
        N = 5
        state_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = N * 112)
        prog2 = cl.Program(self.ctx, r_kernel).build()
        prog2.init_ranlux(self.queue, (N, 1), None, numpy.int32(17), state_buf)

        for bN in range(2, 17, 1):
            res = numpy.zeros(bN).astype(numpy.float32)
            res_buf = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=res)
            prog2.get_random_vector(self.queue, (N, 1), None, res_buf, numpy.int32(bN), state_buf)
            cl.enqueue_copy(self.queue, res, res_buf)
            # Just running



if __name__ == '__main__':
    unittest.main()
