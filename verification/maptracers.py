#!/usr/bin/env python
import numpy as np
from darwin import iofmt

RCP = 120.0

def mapmonod(namesmonod, namesdarwin, log=None):
    names0 = np.array(namesmonod, object)
    names = np.array(namesdarwin, object)
    namel0 = names0.tolist()
    namel = names.tolist()

    nphy = 0

    ii = []
    ff = []
    for name in namel:
        nameorig = name
        if name in namel0:
            ii.append(namel0.index(name))
            ff.append(1.)
            continue
        if name[-1] == 'C':
            name = name[:-1] + 'P'
            if name in namel0:
                ii.append(namel0.index(name))
                ff.append(RCP)
                continue
        if name.startswith('c'):
            iphy = int(name[1:])
            name0 = 'Phy' + str(iphy)
            if name0 in namel0:
                ii.append(namel0.index(name0))
                ff.append(RCP)
                nphy = max(nphy, iphy)
                continue
            name0 = 'Phy{:02d}'.format(iphy)
            if name0 in namel0:
                ii.append(namel0.index(name0))
                ff.append(RCP)
                nphy = max(nphy, iphy)
                continue
            name0 = 'ZOO' + str(int(name[1:])-nphy) + 'P'
            if name0 in namel0:
                ii.append(namel0.index(name0))
                ff.append(RCP)
                continue
        if name.startswith('Phy'):
            name = 'Phy{:02d}'.format(int(name[3:]))
            if name in namel0:
                ii.append(namel0.index(name))
                ff.append(RCP)
                continue
        if name.startswith('Chl'):
            name = 'Chl{:02d}'.format(int(name[3:]))
            if name in namel0:
                ii.append(namel0.index(name))
                ff.append(1.0)
                continue
        if name.startswith('zc'):
            name = 'ZOO' + str(int(name[2:])) + 'P'
            if name in namel0:
                ii.append(namel0.index(name))
                ff.append(RCP)
                continue
        ii.append(0)
        ff.append(np.nan)
        if log is not None:
            log.write('Tracer not found: {}\n'.format(nameorig))

    for i,name in enumerate(namel):
        if log is not None:
            log.write('{i} {0:<6s} <- {1:6.2f}*{2}\n'.format(name, ff[i], ff[i] and namel0[ii[i]] or '', i=iofmt(i+1)))

    return ii,ff


def mapquota(namesquota, namesdarwin, log=None):
    names0 = np.array(namesquota, object)
    names = np.array(namesdarwin, object)
    namel0 = names0.tolist()
    namel = names.tolist()

    ii = []
    ff = []
    for name in namel:
        nameorig = name
        f = 1.
        if name in namel0:
            ii.append(namel0.index(name))
            ff.append(f)
            continue
    #    if name[-1] == 'P':
    #        name = name[:-1] + 'C'
    #        f = 1./120.
        if name.startswith('c'):
            name = 'BIO_1_{:03d}'.format(int(name[1:]))
            if name in namel0:
                ii.append(namel0.index(name))
                ff.append(f)
                continue
        if name.startswith('n'):
            name = 'BIO_2_{:03d}'.format(int(name[1:]))
            if name in namel0:
                ii.append(namel0.index(name))
                ff.append(f)
                continue
        if name.startswith('fe'):
            name = 'BIO_3_{:03d}'.format(int(name[2:]))
            if name in namel0:
                ii.append(namel0.index(name))
                ff.append(f)
                continue
        if name.startswith('Chl'):
            name = 'BIO_4_{:03d}'.format(int(name[3:]))
            if name in namel0:
                ii.append(namel0.index(name))
                ff.append(f)
                continue
        if name.startswith('DO') and name[2] in 'CNF':
            name = 'OM_{:d}_1'.format('CNF'.index(name[2])+1)
            if name in namel0:
                ii.append(namel0.index(name))
                ff.append(f)
                continue
        if name.startswith('PO') and name[2] in 'CNF':
            name = 'OM_{:d}_2'.format('CNF'.index(name[2])+1)
            if name in namel0:
                ii.append(namel0.index(name))
                ff.append(f)
                continue
        ii.append(0)
        ff.append(np.nan)
        if log is not None:
            log.write('Tracer not found: {}\n'.format(nameorig))

    for i,name in enumerate(namel):
        if log is not None:
            log.write('{i} {0:<6s} <- {1:6.2f}*{2}\n'.format(name, ff[i], ff[i] and namel0[ii[i]] or '', i=iofmt(i+1)))

    return ii, ff


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    exp = args[0]
    f0 = '../../../Darwin2/darwin2/verification/{}/names'.format(exp)
    f1 = '{}/names'.format(exp)
    names0 = np.genfromtxt(f0, object)
    names = np.genfromtxt(f1, object)
    if '-q' in args:
        mapquota(names0, names, sys.stdout)
    else:
        mapmonod(names0, names, sys.stdout)

