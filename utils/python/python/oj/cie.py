'''
RGB/sRGB values are in the range (0,1)
'''
import numpy as np

_gamma = 2.4
_igamma = 1./_gamma
_lRGBcut = 0.0031306684425005567
_sRGBcut = 0.04044823627710719
_XYZ2RGB_mat = np.matrix([[ 3.2406, -1.5372, -0.4986],
                          [-0.9689,  1.8758,  0.0415],
                          [ 0.0557, -0.2040,  1.0570]])
_RGB2XYZ_mat = _XYZ2RGB_mat**-1
#_RGB2XYZ_mat = np.matrix([[0.4124, 0.2126, 0.0193],
#                          [0.3576, 0.7152, 0.1192],
#                          [0.1805, 0.0722, 0.9505]])
_D65 = np.r_[0.95047, 1.0000, 1.08883]
_fXYZcut = (6./29)**3
_fXYZfac = (29./6)**2/3.
_fXYZoff = 4./29

def rollaxis(a, axis):
    return np.rollaxis(np.asfarray(a), axis)

def arollaxis(a, axis):
    b = rollaxis(a, axis)
    if b.ndim == 1:
        return (np.ndarray((), b.dtype, b[i:].data) for i in range(len(b)))
    else:
        return b

def XYZ2sRGB(XYZ, axis=-1):
    XYZ = np.asfarray(XYZ)
    out = np.zeros_like(XYZ)
    sRGB = np.rollaxis(out, axis)
    XYZ = np.rollaxis(XYZ, axis)
    lRGB = np.zeros_like(XYZ)
    for i in range(3):
        for j in range(3):
            lRGB[i] += _XYZ2RGB_mat[i,j]*XYZ[j]
    sRGB[...] = np.where(lRGB > _lRGBcut, 1.055*lRGB**_igamma - .055,
                                          12.92*lRGB)
    return out
    
def sRGB2XYZ(sRGB, axis=-1):
    sRGB = np.asfarray(sRGB)
    out = np.zeros_like(sRGB)
    XYZ = np.rollaxis(out, axis)
    sRGB = np.rollaxis(sRGB, axis)
    lRGB = np.where(sRGB > _sRGBcut, ((sRGB+.055)/1.055)**_gamma,
                                     sRGB/12.92)
    for i in range(3):
        for j in range(3):
            XYZ[i] += _RGB2XYZ_mat[i,j]*lRGB[j]
    return out

def XYZ2Lab(XYZ, axis=-1, white=_D65):
    XYZ = np.asfarray(XYZ)
    out = np.zeros_like(XYZ)
    L,a,b = arollaxis(out, axis)
    XYZ = np.rollaxis(XYZ, axis)
    fXYZ = np.zeros_like(XYZ)
    for i in range(3):
        fXYZ[i] = XYZ[i] / white[i]
    fX,fY,fZ = np.where(fXYZ > _fXYZcut, fXYZ**(1./3),
                                         _fXYZfac*fXYZ + _fXYZoff)
    L[...] = 116.*(fY-_fXYZoff)
    a[...] = 500.*(fX-fY)
    b[...] = 200.*(fY-fZ)
    return out

def Lab2XYZ(Lab, axis=-1, white=_D65):
    Lab = np.asfarray(Lab)
    out = np.zeros_like(Lab)
    X,Y,Z = arollaxis(out, axis)
    L,a,b = np.rollaxis(Lab, axis)
    Y[...] = L/116. + _fXYZoff
    X[...] = Y + a/500.
    Z[...] = Y - b/200.
    out[...] = np.where(out > 6./29, out**3.,
                                     (out-_fXYZoff)/_fXYZfac)
    X *= white[0]
    Y *= white[1]
    Z *= white[2]
    return out

def Lab2Msh(Lab, axis=-1):
    Lab = np.asfarray(Lab)
    out = np.zeros_like(Lab)
    M,s,h = arollaxis(out, axis)
    L,a,b = np.rollaxis(Lab, axis)
    M[...] = np.sqrt(L*L+a*a+b*b)
    s[...] = np.where(M > 0., np.arccos(L/M), 0.)
    h[...] = np.arctan2(b, a)
    return out

def Msh2Lab(Msh, axis=-1):
    Msh = np.asfarray(Msh)
    out = np.zeros_like(Msh)
    L,a,b = arollaxis(out, axis)
    M,s,h = np.rollaxis(Msh, axis)
    L[...] = M*np.cos(s)
    r = M*np.sin(s)
    a[...] = r*np.cos(h)
    b[...] = r*np.sin(h)
    return out

def sRGB2Lab(sRGB, axis=-1, white=_D65):
    return XYZ2Lab(sRGB2XYZ(sRGB, axis), axis, white)

def Lab2sRGB(Lab, axis=-1, white=_D65):
    return XYZ2sRGB(Lab2XYZ(Lab, axis, white), axis)

def sRGB2Msh(sRGB, axis=-1, white=_D65):
    return Lab2Msh(XYZ2Lab(sRGB2XYZ(sRGB, axis), axis, white), axis)

def Msh2sRGB(Msh, axis=-1, white=_D65):
    return XYZ2sRGB(Lab2XYZ(Msh2Lab(Msh, axis), axis, white), axis)

