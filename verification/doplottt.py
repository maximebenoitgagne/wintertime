#!/usr/bin/env python
'''Usage: doplottt.py tt-a-file [SEC0]
'''
import sys
from plt import *
import sobol

def sbar(a, cols, labels=None):
    cla()
    ns,n = a.shape
    if labels is None:
        labels = (ns-1)*[None]
    for i in range(1, ns)[::-1]:
        bar(r_[.6:n], a[i]-a[i-1], bottom=a[i-1], color=cols[i-1:i], label=labels[i-1])


args = sys.argv[1:]
fname = args.pop(0)

tt = genfromtxt(fname, names=True)
ta = array(tt.tolist()).T
a = cumsum(array(tt.tolist()).T, 0)
a = r_[a[:1]*0,a]
names = tt.dtype.names
lennames = max(len(name) for name in names)

sob2 = sobol.sobol(2, 0)
sob2cols = array([sob2.next() for _ in range(100)])
sob = sobol.sobol(3, 0)
sobcols = array([sob.next() for _ in range(100)])
gudinds = [i for i,name in enumerate(tt.dtype.names) if name[:3] == 'GUD']
cols = array(sobcols[:len(tt.dtype.names)])
#cols[:,2] = .5 + .5*cols[:,2]
#for i,ind in enumerate(gudinds):
##    cols[ind] = r_[1., 1., 1.]*(i+1)/(len(gudinds)+1)
#    cols[ind] = 0
#    cols[ind,0] = sob2cols[i+1,0]
#    cols[ind,1] = sob2cols[i+1,1]

if args:
    i0name = args.pop(0).upper()
    i0 = tt.dtype.names.index(i0name[:lennames])
else:
    i0 = 0
a1 = a - a[i0] # + a[i0].max()

atot = a[-1].max()
labels = list(names)
for i in range(len(labels)):
    if ta[i].max() < .005*atot:
        labels[i] = None

import gapgrid

fig = figure(1)
clf()
gg = gapgrid.gapgrid([1.], [1.], left=.8j, right=2.2j)
ax = gg.axes_all[0]
sca(ax)

sbar(a1, cols, labels)
legend(fontsize='medium', loc=2, bbox_to_anchor=(1,1), borderaxespad=0)
xticks(r_[1:len(tt)+1])
xlim(.5,len(tt)+.5)
ylim(a1.min(), a1.max())
title(fname)
draw()

fw,fh = fig.get_size_inches()
fig.set_size_inches((23.9, 10.))
comps = fname.split(os.sep)
comps[-3:] = ['_'.join(comps[-2:])+'.png']
pngname = os.sep.join(comps)
savefig(pngname, dpi=80)

fig.set_size_inches((fw,fh))

