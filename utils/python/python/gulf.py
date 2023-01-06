import numpy as np
from haloarray import Exchange, FacetArray
from gridfacet import GridFacet, MITGridFacet

exch = Exchange([[ None, None, None, None ]])

n1,n2 = 640,864
dims = (n1,n2)
#grid48 = GridFacet('/scratch/jahn/grid/ap0003.48', 50, dims, exch)
mitgrid = MITGridFacet('/scratch/jahn/grid/gulf/cs2040/bathy0/tile{0:03d}.mitgrid', dims, exch)


