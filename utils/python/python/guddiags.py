import numpy as np
from modis import enhance

def mkmix(a, mx, p1, p2, cmap, frac=.5, thresh=0., indices=[0, 1, 2, 3]):
    x = a.clip(0.)/mx

    s = x.sum(-1, keepdims=True).clip(0., 1.)
    if p1 == 0:
        f = enhance(s)
    elif p1 < 0:
        f = enhance(s*abs(p1))/enhance(abs(p1))
    else:
        f = s**p1

    amx = a.max(-1, keepdims=True)
    msk = (a >= frac*amx).astype(int)
    low = a.sum(-1) < thresh
    msk[low] = 0
    idx = tuple(msk[..., c] for c in indices)

    rgb = (f*cmap[idx])**p2

    return rgb


def mixcolorscale(fig, pos, names, rgba, **kw):
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
            ax.add_patch(descartes.PolygonPatch(pp, fc=rgba[idx], ec='k', alpha=1.))

    ax.text(-2.35, .95, names[0], ha='right', **kw)
    ax.text(-1, 1.95, names[1], ha='center', **kw)
    ax.text(1, 1.92, names[2], ha='center', **kw)
    ax.text(2.35, .95, names[3], **kw)

    return ax

