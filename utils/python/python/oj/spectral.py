#!/usr/bin/env python
import sys
import numpy as np
import matplotlib as mpl
import colorpy.ciexyz as cie
import colorpy.colormodels as cmod

cmod.init_clipping(clip_method=cmod.CLIP_CLAMP_TO_ZERO)

deflams = np.arange(400,701,25)

# SyncMaster 275T warm1 contr .65 bright .59
SM275T_Red   = cmod.xyz_color(0.66 , 0.33 )
SM275T_Green = cmod.xyz_color(0.205, 0.68 )
SM275T_Blue  = cmod.xyz_color(0.15 , 0.064)
SM275T_White = cmod.xyz_color(0.32 , 0.33 )
SM275T_gamma = 2.07

def recompute():
    global rgb_from_oasim_matrix
    rgb_from_oasim_matrix = rgb_from_spec_mat(deflams)


def SM275T(clip_method=cmod.CLIP_CLAMP_TO_ZERO):
    """ initialize color model and gamma for SyncMaster 275T
        default clipping method is CLIP_CLAMP_TO_ZERO """
    cmod.init(SM275T_Red,SM275T_Green,SM275T_Blue,SM275T_White)
    cmod.init_clipping(clip_method)
    cmod.init_gamma_correction(cmod.simple_gamma_invert, 
                               cmod.simple_gamma_correct,
                               SM275T_gamma)
    recompute()
    sys.stderr.write('Initialized phosphors and gamma for SyncMaster 275T\n')


def init_simple(gamma=2.2, clip_method=cmod.CLIP_CLAMP_TO_ZERO):
    """ initialize color model and gamma for default phosphors
        and power-law gamma correction
        default clipping method is CLIP_CLAMP_TO_ZERO
    """
    cmod.init()
    cmod.init_clipping(clip_method)
    cmod.init_gamma_correction(cmod.simple_gamma_invert, 
                               cmod.simple_gamma_correct,
                               gamma)
    recompute()
    sys.stderr.write('Initialized default phosphors and gamma = %g\n'%gamma)


def init_srgb(clip_method=cmod.CLIP_CLAMP_TO_ZERO):
    """ initialize color model and gamma for sRGB phosphors
        and gamma correction; default clipping method is CLIP_CLAMP_TO_ZERO
    """
    cmod.init()
    cmod.init_clipping(clip_method)
    cmod.init_gamma_correction()
    recompute()
    sys.stderr.write('Initialized default phosphors and sRGB gamma correction\n')


_c0s = [.9,0.,0.,.4,1.]
_c1s = [.9,0.,1.,.4,1.]
_names = ['R','G','B']

def make_cmaps(limlo=-0.25,limhi=2.0):
    rng = limhi - limlo
    x0 = -limlo/rng
    x1 = (1-limlo)/rng
    xs = [0.,x0,x1,x1,1.]
    cmaps = [ mpl.colors.LinearSegmentedColormap(_names[i],
                  [ (x, i*[c0] + [c1] + (2-i)*[c0])
                    for x,c0,c1 in zip(xs,_c0s,_c1s) ])
              for i in range(3) ]
    return cmaps


cmaps = make_cmaps(0.2222222222222222, 2.)

#_x0 = .1
#_x1 = .55
#cmapR = mpl.colors.LinearSegmentedColormap.from_list('R',[
#            (0.,[.9,.9,.9]),
##            (x0,[.1,.1,.1]),
#            (x0,[0.,0.,0.]),
#            (x1,[1.,0.,0.]),
#            (x1,[1.,.4,.4]),
#            (1.,[1.,1.,1.])])
#cmapG = mpl.colors.LinearSegmentedColormap.from_list('G',[
#            (0.,[.9,.9,.9]),
##            (x0,[.1,.1,.1]),
#            (x0,[0.,0.,0.]),
#            (x1,[0.,1.,0.]),
#            (x1,[.4,1.,.4]),
#            (1.,[1.,1.,1.])])
#cmapB = mpl.colors.LinearSegmentedColormap.from_list('B',[
#            (0.,[.9,.9,.9]),
##            (x0,[.1,.1,.1]),
#            (x0,[0.,0.,0.]),
#            (x1,[0.,0.,1.]),
#            (x1,[.4,.4,1.]),
#            (1.,[1.,1.,1.])])
#cmaps = [cmapR,cmapG,cmapB]

def rgb_from_spec_mat(lams=deflams):
    n = len(lams)
    m = np.empty((3,n))
    for i in range(n):
        m[:,i] = cie.xyz_from_wavelength(lams[i])
    return np.dot(cmod.rgb_from_xyz_matrix, m)


rgb_from_oasim_matrix = rgb_from_spec_mat(deflams)

def rgb_from_oasim(oasim):
    return np.tensordot(rgb_from_oasim_matrix, oasim, 1)


def rgb_from_spec(a,lams=deflams):
    """ return unclipped, not gamma-corrected rgb array for spectrum a """
    if a.shape[-1] != len(lams):
        # first axis becomes last
        a = np.rollaxis(a,0,len(a.shape))
    nl = a.shape[-1]
    a1 = a.reshape((-1,nl))
    dims,_ = a1.shape
    spec = np.c_[lams,lams*0.]
    rgb = np.ones((dims,3))
    for i in range(dims):
        spec[:,1] = a1[i,:]
        rgb[i,:] = cmod.rgb_from_xyz(cie.xyz_from_spectrum(spec))

    return rgb.reshape(a.shape[:-1] + (3,))


def rgba_from_spec(a,lams=deflams,norm=False):
    """ return [0,1]-clipped and gamma-corrected rgba array
        for spectrum a (opacity = 1) """
    rgb = rgb_from_spec(a,lams).reshape((-1,3))
    if norm:
        mx = np.amax(rgb)
        print 'dividing by',mx
        rgb /= mx
    dims = rgb.shape[0]
    rgba = np.ones((dims,4))
    for i in range(dims):
        rgba[i,:3] = cmod.irgb_from_rgb(rgb[i,:])/255.

    return rgba.reshape(a.shape[:-1] + (4,))


def rgba_from_rgb(a,norm=False):
    """ return [0,1]-clipped and gamma-corrected rgba array
        for spectrum a (opacity = 1) """
    if a.shape[-1] != 3:
        # move first axis to last
        a = np.rollaxis(a,0,len(a.shape))

    rgb = a.reshape((-1,3))
    if norm:
        mx = np.amax(rgb)
        print 'dividing by',mx
        rgb = rgb/mx
    dims = rgb.shape[0]
    rgba = np.ones((dims,4))
    for i in range(dims):
        rgba[i,:3] = cmod.irgb_from_rgb(rgb[i,:])/255.

    return rgba.reshape(a.shape[:-1] + (4,))


def clip_rgb(rgb_color, norm=False, gamma=cmod.gamma_exponent):
    rgb = rgb_color.copy()

    rgb[rgb<0] = 0

    # clip intensity if needed (rgb values > 1.0) by scaling
    if norm:
        # scale everything the same
        mx = np.amax(rgb)
        print 'dividing by', mx
        rgb *= 1./mx
    else:
        # scale individual colors if > 1
        mx = np.amax(rgb, axis=0)
        ind = np.where(mx > 1.)
        rgb[np.s_[:,] + ind] /= mx[ind]

    # gamma correction
    rgb = np.power(rgb, 1./gamma)

    rgb[rgb>1] = 1

    return np.rollaxis(rgb,0,len(rgb.shape))


######################################################################
def rmud_from_solz(rad,solz,rn=1.341):
    rsza = solz/rad
    sinszaw = np.sin(rsza)/rn
    szaw = np.arcsin(sinszaw)
    rmudl = 1.0/np.cos(szaw)   # avg cosine direct (1 over)
    rmudl = np.minimum(rmudl,1.5)
    rmud = np.maximum(rmudl,0.0)
    return rmud


rad = 180./np.pi

rd = 1.5
ru = 3.0
rmus = 1.0/0.83
rmuu = 1.0/0.4
radmod_thresh = 1e-4

def radmod(zd,Edtop,Estop,rmud,rmus,rmuu,a,bt,bb):
    """ Edbot,Esbot,Eubot,Eutop = radmod(zd,Edtop,Estop,a,bt,bb,rmud,rmus,rmuu)

        radiative transfer model for one layer of thinkness zd
        all arguments may be scalar or 1d arrays
    """
    cd = (a+bt)*rmud
    Edz = Edtop*np.exp(-cd*zd)

    au = a*rmuu
    Bu = ru*bb*rmuu
    Cu = au+Bu
    As = a*rmus
    Bs = rd*bb*rmus
    Cs = As+Bs
    Bd = bb*rmud
    Fd = (bt-bb)*rmud
    bquad = Cs - Cu
    cquad = Bs*Bu - Cs*Cu
    rt = np.sqrt(bquad*bquad - 4.0*cquad)
    a1 = 0.5*(-bquad + rt)  # K of Aas
    a2 = 0.5*(-bquad - rt)
    S = -(Bu*Bd + Cu*Fd)
    SEdz = S*Edz
    a2ma1 = a2 - a1
    rM = SEdz/(a1*a2ma1)
    rN = SEdz/(a2*a2ma1)
#    ea2Dmax = exp(a2ma1*Dmax)
#    c1 = (rN-rM)*exp(-a1*Dmax) - (Estop-rM+rN)*ea2Dmax
#   *                             /(1.0-ea2Dmax)
#    c2 = Estop - rM + rN - c1
    c2 = Estop - rM + rN
#    a1arg = a1*zd
#    a1arg = min(a1arg,82.8)
#    Ta1z = exp(a1arg)
    Ta2z = np.exp(a2*zd)
#    Esz = c1*Ta1z + c2*Ta2z + rM - rN
    Esz = c2*Ta2z + rM - rN
    Esz = np.maximum(Esz,0.0)
#    Eutmp = ((a1+Cs)*c1)*Ta1z + ((a2+Cs)*c2)*Ta2z + Cs*rM
#   *             - Cs*rN - Fd*Edz
    Eutmp = ((a2+Cs)*c2)*Ta2z + Cs*rM - Cs*rN - Fd*Edz
    Euz = Eutmp/Bu
    Euz = np.maximum(Euz,0.0)

# Eut top of layer
    rM = S*Edtop/(a1*a2ma1)
    rN = S*Edtop/(a2*a2ma1)
    Eutmp = (a2+Cs)*c2 + Cs*rM - Cs*rN - Fd*Edtop
    Eutop = Eutmp/Bu
    Eutop = np.maximum(Eutop,0.0)

    return Edz,Esz,Euz,Eutop


def radtrans(drf,hFacC,Edsf,Essf,a,bt,bb,rmud,rmus=rmus,rmuu=rmuu,thresh=radmod_thresh):
    """
        Edz,Esz,Euz,Eut = radtrans(drf,hFacC,Edsf,Essf,a,bt,bb,rmud,rmus,rmuu,thresh)

        drf[k], hFacC[k,j,i]
        Edsf[l,j,i]
        a[l,k,j,i]
        rmud[j,i]
        rmus, rmuu

        Edz[l,k,j,i]
    """
    nl,nr,ny,nx = a.shape
    drf = drf.reshape([nr,1,1])

    Edz = np.zeros([nl,nr,ny,nx])
    Esz = np.zeros([nl,nr,ny,nx])
    Euz = np.zeros([nl,nr,ny,nx])
    Eut = np.zeros([nl,nr,ny,nx])

    Edtop = Edsf
    Estop = Essf
    for k in range(nr):
        zd = drf[k,:,:]*hFacC[k,:,:]
        for l in range(nl):
            Edz[l,k,:,:] = 0.
            Esz[l,k,:,:] = 0.
            Euz[l,k,:,:] = 0.
            Eut[l,k,:,:] = 0.
            jj,ii = np.where((Edtop[l,:,:] >= thresh) &
                             (Estop[l,:,:] >= thresh))
            for j,i in zip(jj,ii):
              Edz[l,k,j,i],Esz[l,k,j,i],Euz[l,k,j,i],Eut[l,k,j,i] = radmod(
                    zd[j,i],Edtop[l,j,i],Estop[l,j,i],rmud[j,i],rmus,rmuu,
                    a[l,k,j,i],bt[l,k,j,i],bb[l,k,j,i]
                )

        Edtop = Edz[:,k,:,:]
        Estop = Esz[:,k,:,:]

    return Edz,Esz,Euz,Eut

