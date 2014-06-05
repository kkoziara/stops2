__author__ = 'Kamil Koziara'

import cProfile

import numpy
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

init_pop = generate_pop([(20, base1), (380, base2)])
grid = HexGrid(20, 20, 1)

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
    x = Stops2(trans_mat, init_pop, grid.adj_mat, range(400),
               bound, secretion, reception, receptors, secr_amount=6, leak=0, max_con=6, max_dist=1.5, opencl=False,
                asym_id=-1, div_id=-1, die_id=-1)
    for i in range(300):
        x.step()
        if i%10 == 0:
            print i
            draw_hex_grid("pics/f%04d.png"%i, x.pop, grid, color_fun)

run()
#cProfile.run("run()")
