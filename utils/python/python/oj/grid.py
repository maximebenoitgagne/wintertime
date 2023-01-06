import sys
import numpy.core.numeric as _nx
from numpy.core.numeric import ( asarray, ScalarType, array, alltrue, cumprod,
                                 arange, mod )
import math

class my_nd_grid(object):
    def __init__(self, sparse=False, centered=False):
        self.sparse = sparse
        self.centered = centered
    def __getitem__(self,key):
        try:
            size = []
            typ = int
            for k in range(len(key)):
                step = key[k].step
                start = key[k].start
                if start is None: start=0
                if step is None: step=1
                if isinstance(step, complex):
                    if self.centered:
                        size.append(int(abs(step)))
                    else:
                        size.append(int(abs(step)))
                    typ = float
                else:
                    if self.centered:
                        # end point included, but one less
                        size.append(math.floor((key[k].stop - start)/(step*1.0)))
                    else:
                        size.append(math.ceil((key[k].stop - start)/(step*1.0)))
                if isinstance(step, float) or \
                    isinstance(start, float) or \
                    isinstance(key[k].stop, float) or \
                    (self.centered and not isinstance(step,complex) and mod(step,2)):
                    typ = float
            if self.sparse:
                nn = map(lambda x,t: _nx.arange(x, dtype=t), size, \
                                     (typ,)*len(size))
            else:
                nn = _nx.indices(size, typ)
            for k in range(len(size)):
                step = key[k].step
                start = key[k].start
                if start is None: start=0
                if step is None: step=1
                if isinstance(step, complex):
                    step = int(abs(step))
                    if self.centered:
                        if step != 0:
                            step = (key[k].stop - start)/float(step)
                    else:
                        if step != 1:
                            step = (key[k].stop - start)/float(step-1)
                if self.centered:
                    start += .5*step
                nn[k] = (nn[k]*step+start)
            if self.sparse:
                slobj = [_nx.newaxis]*len(size)
                for k in range(len(size)):
                    slobj[k] = slice(None,None)
                    nn[k] = nn[k][slobj]
                    slobj[k] = _nx.newaxis
            return nn
        except (IndexError, TypeError):
            step = key.step
            stop = key.stop
            start = key.start
            if start is None: start = 0
            if isinstance(step, complex):
                step = abs(step)
                length = int(step)
                if self.centered:
                    if step != 0:
                        step = (key.stop-start)/float(step)
                else:
                    if step != 1:
                        step = (key.stop-start)/float(step-1)
                stop = key.stop+step
                if self.centered:
                    start += .5*step
                return _nx.arange(0, length,1, float)*step + start
            else:
                if self.centered:
                    start += .5*step
                return _nx.arange(start, stop, step)

    def __getslice__(self,i,j):
        if self.centered:
            return _nx.arange(i+.5,j)
        else:
            return _nx.arange(i,j)

    def __len__(self):
        return 0

cgrid = my_nd_grid(sparse=False, centered=True)
mymgrid = my_nd_grid(sparse=False, centered=False)

