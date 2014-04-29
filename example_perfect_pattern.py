__author__ = 'Kamil Koziara'

import cProfile

import numpy
from utils import generate_pop, HexGrid, draw_hex_grid
from stops_ import Stops2

secretion = numpy.array([5, 6, 7])
reception = numpy.array([3, 4, 2])
receptors = numpy.array([1, -1, -1])
bound=numpy.array([1,1,1,1,1,1,1,1])

base1=numpy.array([0,0,1,0,0,0,0,0])
base2=numpy.array([0,0,0,0,0,0,0,0])


trans_mat = numpy.array([[0,-10,0,0,0,0,10,0], #notch
                         [0,0,0,0,0,1,0,0], #Delta
                         [0,1,0,0,0,0,0,0.1], #basal
                         [0.005,0,0,0,0,0,0,0], #delta receptor
                         [-10,0,0,0,0,0,0,0], #notch receptor
                         [0,0,0,0,0,0,0,0], #ligand_delta
                         [0,0,0,0,0,0,0,0], #ligand_notch
                         [0,0,0,0,0,0,0,0] #ligand_basal
                        ])

init_pop = generate_pop([(50, base1), (2450, base2)])
grid = HexGrid(50, 50, 1)

def color_fun(row):
    if row[0]==1:
        return 1
    elif row[4]==1:
        return 0.75
    elif row[1]==1:
        if row[3]==1:
            return 0.55
        else:
            return 0.3
    else:
        if row[2]==1:
            return 0.2
        else:
            return 0.



def run():
    x = Stops2(trans_mat, init_pop, grid.adj_mat, bound, secretion, reception, receptors, secr_amount=6, leak=0, max_con=6, max_dist=1.5, opencl=True)
    for i in range(130):
        x.step()
        if i%10 == 0:
            print i
            draw_hex_grid("pics/f%04d.png"%i, x.pop, grid, color_fun)


cProfile.run("run()")