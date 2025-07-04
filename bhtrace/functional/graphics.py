
import matplotlib.pyplot as plt


# todo:
# - Add pre-set line coloring routines
# 

def opt_mosaic(ax_ids, fill_None=True, filler=None, rot=False):
    '''
    Function that composes graphs from list to an optimal mosaic

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