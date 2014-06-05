__author__ = 'Kamil Koziara'

from stops_ import Stops2
from utils import generate_pop
import numpy


SECR=4.
POP_SIZE=1000
LEAK=5
INIT_ENV=POP_SIZE/10

# g1,g2 gp,ga, l,r, divide,die
trans_mat= numpy.array([
                       [-1,1, 0,0, 0,0, 0,0], #g1
                       [0,-1, 0,0, 1,0, 0,0], #g2
                       [0,0, 0,0, 0,0, 0,0], #gp
                       [0,0, 0,0, 0,0, 0,0], #ga
                       [0,0, 0,0, 0,0, 0,0], #l
                       [1,0, 0,0, 0,-1, 0,0], # r
                       [0,0, 0,0, 0,0, 0,0], #div
                       [0,0, 0,0, 0,0, 0,0], #die
                       ])


secretion=numpy.array([4])
reception=numpy.array([5])
receptors = numpy.array([-1])
base=numpy.array([0,0, 0,0, 0,0, 0,0])
bound=numpy.array( [1,1, 1,1, 1,1, 1,1])
init_env = numpy.array([INIT_ENV])
init_pop = generate_pop([(POP_SIZE, base)])

def run():
    x=Stops2(trans_mat,init_pop,numpy.array([1.0]),[0.0]*POP_SIZE, bound,secretion,reception, receptors,
            secr_amount=SECR, leak=LEAK, init_env=init_env, opencl=True)
    for i in range(30):
        x.step()
    print x.pop.mean(axis=0)
    print x.env

run()