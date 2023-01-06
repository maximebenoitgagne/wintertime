from pylab import *

def ffep(fet):
    return .5*(fet-LT-1/b+sqrt((1/b-(LT-fet))**2+4*LT/b))

LT = 1.         # nM
b = 1./(5e-3)   # 1/nM
freefemax = .4  # nM
scav = .4       # 1/y

fet = linspace(0., 2.2, 501)
fep = ffep(fet)
fepp = minimum(fep, freefemax)
fetp = fet - fep + fepp

rc('lines', lw=2.)

figure(1).clf()
fig, axs = subplots(2, 1, sharex=True, num=1)
fig.set_size_inches((8., 8.), forward=True)
subplots_adjust(left=.11, bottom=.08, top=.94, right=.97, hspace=.1)

sca(axs[1])
cla()
plot(fet, fetp, label=r'Fe${}_{\sf T}^{\sf out}$')
plot(fet, fepp, label=r"Fe'")
#plot(fet, fet, 'k:', scaley=False)
xlabel(r'Fe$_{\sf T}$   (nM)')
ylabel('nM', rotation=0)
legend(loc=0)
#xlm = xlim()
#tick_params('x', labelbottom=False)

rscav = scav*fepp/fet
rmx = rscav[fep<=freefemax][1:].max()
rscav[fep>freefemax] *= 1e3

sca(axs[0])
cla()
title('scavenging')
plot(fet, rscav, label=r'(d/dt) log Fe$_{\sf T}$')
legend(loc=0)
#xlabel(r'FeT    (nM)')
ylabel(r'y$^{-1}$', rotation=0)
#xlim(xlm)
ylim(0, .14)
tick_params('x', labelbottom=False)

xlim(0, fet.max())

draw()

savefig('../scavenging.pdf', dpi=80)
