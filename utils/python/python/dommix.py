#!/usr/bin/env python
'''
Usage:
  dommix.py save
  dommix.py venn <pngoutname>
  dommix.py [options] <indices> <pngoutname>

Options:
  -p <float>     power [default: 1.0]
'''
import numpy as np
import matplotlib as mpl
import oj.num
from modis import enhance

def mkCBYR():
    hsv = np.zeros((2,2,2,2,3))

    #   C B Y R
    hsv[0,0,0,1] = np.r_[  0.  ,  12.,  12.  ]  # r
    hsv[0,0,1,0] = np.r_[  2.  ,  12.,  12.  ]  # y
    hsv[0,1,0,0] = np.r_[  8.  ,  12.,  12.  ]  # b
    hsv[1,0,0,0] = np.r_[  6.  ,  12.,  12.  ]  # c
    hsv[0,0,1,1] = np.r_[  1.25,   8.,  12.  ]  # o
    hsv[0,1,1,0] = np.r_[  5.0 ,   8.,  12.  ]  # g(y)
    hsv[1,1,0,0] = np.r_[  7.  ,  10.,  12.  ]  # c-b
    hsv[0,1,0,1] = np.r_[ 10.75,   9.,  12.  ]  # m
    hsv[1,0,1,0] = np.r_[  2.75,   8.,  12.  ]  # y-g
    hsv[1,0,0,1] = np.r_[  9.  ,   8.,  12.  ]  # l-b
    hsv[1,1,1,0] = np.r_[  6.  ,   4.,  12.  ]
    hsv[1,1,0,1] = np.r_[  8.  ,  3.5,  12.  ]
    hsv[1,0,1,1] = np.r_[  2.  ,   4.,  12.  ]
    hsv[0,1,1,1] = np.r_[  0.  ,   4.,  12.  ]
    hsv[1,1,1,1] = np.r_[  0.  ,   0.,  12.  ]
    hsv[0,0,0,0] = np.r_[  0.  ,   0.,   0.  ]

    rgb = mpl.colors.hsv_to_rgb(hsv/12.)
    return rgb


_order = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (1, 1, 0, 0),
        (1, 0, 1, 0),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (0, 1, 0, 1),
        (0, 0, 1, 1),
        (1, 1, 1, 0),
        (1, 1, 0, 1),
        (1, 0, 1, 1),
        (0, 1, 1, 1),
        (1, 1, 1, 1),
        ]

class DommixMapper(object):
    def __init__(self, lut, vmax=1., p1=1., p2=1., frac=.5, thresh=0., indices=[0,1,2,3]):
        self.lut = lut
        self.vmax = vmax
        self.p1 = p1
        self.p2 = p2
        self.frac = frac
        self.thresh = thresh
        self.indices = indices

    @classmethod
    def fromfile(cls, fname, vmax=1., p1=1., p2=1., frac=.5, thresh=0., indices=[0,1,2,3]):
        lut = oj.num.loadbin(fname)
        obj = cls(lut, vmax, p1, p2, frac, thresh, indices)
        return obj

    def __call__(self, a, axis=-1):
        a = np.moveaxis(a, axis, -1)
        x = a.clip(0.)/self.vmax

        s = x.sum(-1, keepdims=True).clip(0., 1.)
        if self.p1 == 0:
            f = enhance(s)
        elif self.p1 < 0:
            f = enhance(s*abs(self.p1))/enhance(abs(self.p1))
        else:
            f = s**self.p1

        amx = a.max(-1, keepdims=True)
        msk = (a >= self.frac*amx).astype(int)
        low = a.sum(-1) < self.thresh
        msk[low] = 0
        idx = tuple(msk[...,c] for c in self.indices)
        rgb = (f*self.lut[idx])**self.p2
        return rgb

    def Venn_scale(self, fig, pos, names, **kw):
        import descartes
        import shapely.geometry as sg
        import shapely.affinity as sa
        import shapely.ops as so

        r = np.sqrt(.5)
        c = np.cos(np.pi*np.linspace(0.,2.,63))
        s = np.sin(np.pi*np.linspace(0.,2.,63))
        x = np.r_[c[:32],c[31:],c[:1]]
        y = np.r_[s[:32]+1,s[31:]-1,s[:1]+1]
        p = sg.Polygon(zip(x, y))
        p1 = sa.affine_transform(p, [r,-r,r,r,-r-.2,-r+.2])  # LL
        p2 = sa.affine_transform(p, [r,-r,r,r,-.15,.15])     # UL
        p3 = sa.affine_transform(p, [r,r,-r,r,.15,.15])      # UR
        p4 = sa.affine_transform(p, [r,r,-r,r,r+.2,-r+.2])   # LR
        p0 = sg.Polygon([(3,3),(-3,3),(-3,-3),(3,-3),(3,3)])
        ps = [p1, p2, p3, p4]

        a = ['difference', 'intersection']

        ax = fig.add_axes(pos)
        w = 3.5
        ax.set_aspect(1.)
        ax.set_xlim(-w, w)
        ax.set_ylim(-.75*w, .75*w)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])

        for idx in np.ndindex((2, 2, 2, 2)):
            pp = p0
            for i in range(4):
                pp = getattr(pp, a[idx[i]])(ps[i])
            if idx != (0, 0, 0, 0):
                ax.add_patch(descartes.PolygonPatch(pp, fc=self.lut[idx], ec='k', alpha=1.))

        ax.text(-2.35, .95, names[0], ha='right', **kw)
        ax.text(-1, 1.95, names[1], ha='center', **kw)
        ax.text(1, 1.95, names[2], ha='center', **kw)
        ax.text(2.35, .95, names[3], **kw)

        return ax

    def scales(self, fig, rect, names):
        import matplotlib.pyplot as plt

        ax = fig.add_axes(rect)
        nx = int(w*W*fig.dpi)

        tg = np.linspace(0., 1., nx+1)
        tc = .5*(tg[:-1] + tg[1:])
        if self.p1 == 0:
            tc = enhance(tc)
        elif self.p1 < 0:
            tc = enhance(tc*abs(self.p1))/enhance(abs(self.p1))
        else:
            tc = tc**self.p1

        l = []
        rgb = np.zeros((15, nx, 3))
        for i, idx in enumerate(_order):
            rgbaidx = tuple(idx[j] for j in self.indices)
            rgb[i] = (tc[:, None]*self.lut[rgbaidx])**self.p2
            label = '+'.join(s for j, s in enumerate(names) if idx[j])
            l.append(label)

        ax.imshow(rgb, origin='lower', interpolation='none', aspect='auto')

        ax.set_xlabel('total biomass')
        ax.set_xticks([])
        ax.set_yticks(range(15))
        ax.set_yticklabels(l)
        ax.tick_params(left=False, labelleft=False, right=False, labelright=True)

        return ax


if __name__ == '__main__':
    import sys
    from docopt import docopt

    names = 'Pico Cocco Diatom Dino'.split()

    lut = mkBCYR()

    args = docopt(__doc__)
    if args['save']:
        oj.num.savebin('mixBCYR', lut)
    else:
        import matlpotlib.pyplot as plt
        plt.style.use('dark')

        oname = args['<pngoutname>']

        p1 = float(args.get('-p', 1.))
        indices = args.get('<indices>', '0,1,2,3')
        indices = map(int, indices.split(','))
        cmap = DommixMapper(lut, p1=p1, indices=indices)

        W, H = 6.4, 4.8
        fig = plt.figure(figsize=(W, H))
        if args['venn']:
            ax = cmap.Venn_scale(fig, [0,0,1,1], names)
        else:
            w = .58
            b = .05*H
            bb = 1.7*b
            pos = [b/W, bb/H, w, 1.-(b+bb)/H]
            ax = cmap.scales(fig, pos, names)

        fig.savefig(oname, dpi=fig.dpi)

