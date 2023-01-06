import numpy as np
import oj.misc

fieldnames = ('n','t','x','y','z','i','j','k','P','U','V','T','S')

def readflt(filename):
    a = oj.misc.fromraw(filename)
    a = a.view({'names':fieldnames, 'formats':13*(a.dtype,)})
    a = a[...,0]
    a = a.view(np.recarray)
    return a

