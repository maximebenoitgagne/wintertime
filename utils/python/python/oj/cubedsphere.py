import numpy as np
import matplotlib.pyplot as plt
from oj.axes import pcolormesh

def plotfaces(x,y,c,faces,slices,mask=None,axes=None,**kwargs):
    if 'norm' not in kwargs:
        vmin = kwargs.get('vmin',None)
        vmax = kwargs.get('vmax',None)
        if vmin is None or vmax is None:
            cglob = c.toglobal()
            if vmin is None:
                kwargs['vmin'] = cglob.min()
            if vmax is None:
                kwargs['vmax'] = cglob.max()
            del cglob

    if axes is None:
        axes = plt.gca()
    ims = []
    for f,s in zip(faces,slices):
        if mask is None:
            data = c.face(f).z
        else:
            data = np.ma.MaskedArray(c.face(f).z, mask.face(f).z)

        if len(data.shape) > 2 and data.shape[0] <= 4:
            # make last axis color
            data = np.rollaxis(data, 0, len(data.shape))

        ims.append( pcolormesh(axes, x.face(f).z[s], y.face(f).z[s], data[s], **kwargs) )

    return ims
        

def setfaces(ims,c,faces,slices,mask=None,cmap=None,norm=None):
    # flat plus rgba (if any)
    fcsh = (-1,) + c.shape[:-2]
    for im,f,s in zip(ims,faces,slices):
        if mask is None:
            data = c.face(f).z
        else:
            data = np.ma.MaskedArray(c.face(f).z, mask.face(f).z)

        # make last axis color
        if data.ndim > 2:
            data = np.rollaxis(data, 0, data.ndim)

        datas = data[s]
        if datas.shape[0]*datas.shape[1] > im.get_facecolors().shape[0]:
            datas = datas[:-1,:-1]

        datas = datas.reshape(fcsh)

        if len(fcsh) == 1:
            datas = cmap(norm(datas))
        im.set_facecolor(datas)


def plotll(x,y,c,mask=None,**kwargs):
    faces = [0,1,2,3,5]
    slices =  [np.s_[:,:],
               np.s_[:,:],
               np.s_[:256,:],
               np.s_[:256,:],
               np.s_[:,255:],
              ]
    ims1 = plotfaces(np.mod(x+90,360)-90,y,c,faces,slices,mask,**kwargs)
    faces = [2,3,4,5]
    slices =  [
               np.s_[255:,:],
               np.s_[255:,:],
               np.s_[:,:],
               np.s_[:,:256],
              ]
    ims2 = plotfaces(np.mod(x+270,360)-270,y,c,faces,slices,mask,**kwargs)
    return ims1,ims2


def setll(imss,c,mask=None):
    faces = [0,1,2,3,5]
    slices =  [np.s_[:,:],
               np.s_[:,:],
               np.s_[:256,:],
               np.s_[:256,:],
               np.s_[:,255:],
              ]
    setfaces(imss[0],c,faces,slices,mask)
    faces = [2,3,4,5]
    slices =  [
               np.s_[255:,:],
               np.s_[255:,:],
               np.s_[:,:],
               np.s_[:,:256],
              ]
    setfaces(imss[1],c,faces,slices,mask)


class PcolorCS(object):
    def __init__(self,x,y,c,faces,slices,offx,mask=None,**kwargs):
        self.faces = faces
        self.slices = slices
        self.offxs = offx
        self.ims = [ plotfaces(np.mod(x-x0,360)+x0,y,c,f,s,mask,**kwargs) for f,s,x0 in zip(faces,slices,offx) ]

    def set(self,c,mask=None):
        if len(c.shape) < 3:
            cmap = self.ims[0][0].cmap
            norm = self.ims[0][0].norm
        else:
            cmap = None
            norm = None
        for im,f,s in zip(self.ims,self.faces,self.slices):
            setfaces(im,c,f,s,mask,cmap,norm)


def pcolor_ll_0_360(x,y,c,mask=None,**kwargs):
    ny,nx = c.face(0).i.shape
    nyh = ny//2
    nxh = nx//2
    faces = [[0,1,2,3,5], [0,2,4,5]]
    slices =  [[
               np.s_[:,nxh:],
               np.s_[:,:],
               np.s_[:-nyh,:],
               np.s_[:,:],
               np.s_[:,nxh:],
              ],[
               np.s_[:,:-nxh],
               np.s_[nyh:,:],
               np.s_[:,:],
               np.s_[:,:-nxh],
              ]]
    offx = [-90,90]
    return PcolorCS(x,y,c,faces,slices,offx,mask,**kwargs)


def pcolor_ll_180_180(x,y,c,mask=None,**kwargs):
    ny,nx = c.face(0).i.shape
    nyh = ny//2
    nxh = nx//2
    faces = [[0,1,2,3,5], [2,3,4,5]]
    slices = [[np.s_[:,:],
               np.s_[:,:],
               np.s_[:-nyh,:],
               np.s_[:-nyh,:],
               np.s_[:,nxh:],
              ],[
               np.s_[nyh:,:],
               np.s_[nyh:,:],
               np.s_[:,:],
               np.s_[:,:-nxh],
              ]]
    offx = [-90,-270]
    return PcolorCS(x,y,c,faces,slices,offx,mask,**kwargs)


