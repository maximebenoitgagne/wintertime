#!/usr/bin/env python
from matplotlib.colors.import LinearSegmentedColormap

def cmjet(n):
    cdict = {'red':   [(0./8,  0.0, 0.0),
                       (3./8,  0.0, 0.0),
                       (5./8,  1.0, 1.0),
                       (7./8,  1.0, 1.0),
                       (8./8,  0.5, 0.5)],
             'green': [(0./8,  0.0, 0.0),
                       (1./8,  0.0, 0.0),
                       (3./8,  1.0, 1.0),
                       (5./8,  1.0, 1.0),
                       (7./8,  0.0, 0.0),
                       (8./8,  0.0, 0.0)],
             'blue':  [(0./8,  0.5, 0.5),
                       (1./8,  1.0, 1.0),
                       (3./8,  1.0, 1.0),
                       (5./8,  0.0, 0.0),
                       (8./8,  0.0, 0.0)]}
    return LinearSegmentedColormap('jet', cdict, n)

