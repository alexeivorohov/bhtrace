
import matplotlib.pyplot as plt
from typing import Iterable

# todo:
# - Add pre-set line coloring routines
# 

def opt_mosaic(ax_ids: Iterable, fill_None=True, filler=None, rot=False):
    '''
    Function that composes graphs from list to a visually optimal mosaic

    Returns: 
    - shape: tuple(h_n, w_n)
    - mosaic: nested list 
    '''
    shape_cases = {
        1: (1, 1),
        2: (2, 1),
        3: (3, 1),
        4: (2, 2),
        5: (3, 2),
        6: (3, 2),
        7: (4, 2),
        8: (4, 2),
        9: (3, 3)
    }

    n_graphs = len(ax_ids)

    shape = shape_cases[n_graphs]

    mosaic = []

    for h in range(shape[1]):
        row = []
        for w in range(shape[0]):
            
            i = w + h*shape[0]
        
            if i < n_graphs:
                row.append(ax_ids[i])
            elif fill_None:
                row.append(filler)
        mosaic.append(row)

    return shape, mosaic


class Coloring:

    @classmethod
    def lensing_colors():

        pass

    @classmethod
    def trajectory_colors():

        pass

    @classmethod
    def image_colors():

        pass

class LensingPlot:

    @classmethod
    def plot(cls, traj, dphi, layout='joined'):

        pass


    def subplot_lf(b, dphi):

        pass