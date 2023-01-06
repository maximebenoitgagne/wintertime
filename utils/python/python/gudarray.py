#!/usr/bin/env python
import sys
from os.path import join as pjoin, split as psplit, exists
from collections import OrderedDict
import numpy as np
import xarray as xr
import nml

class RunInfo(object):
    def __init__(self, rundir=None):
        if rundir:
            self.read(rundir)

        self.nphoto = 0
        for i in range(self.nplank):
            if self.tr.isphoto[i]:
                self.nphoto = i + 1
        self.nphy = self.nphoto
        self.nzoo = self.nplank - self.nphy

        self.grpnames = self.tr.get('grp_names', None)
        self.grp = self.tr['mygroup'] - 1
        self.vol = self.tr['biovol']
        # vol = (4*pi/3)*(dm/2)**3 = (pi/6)*dm**3
        # dm = (6*vol/np.pi)**(1./3)
        self.esd = (6*self.vol/np.pi)**(1./3)
        if len(self.vol) > 1:
            self.dlv = np.median(np.diff(sorted(set(np.array(np.log(self.vol))))))
            self.ivol = np.round(np.log(self.vol/self.vol.min())/self.dlv).astype(int)
        else:
            self.ivol = len(self.vol)*[0]
        self.ngrp = int(np.max(self.grp)) + 1
        self.nvol = int(np.max(self.ivol)) + 1
        self.esds = np.zeros((self.nvol,))
        self.esds.put(self.ivol, self.esd)

    def __getattr__(self, a):
        if a in self.tr:
            return self.tr[a]
        elif hasattr(self, a):
            return getattr(self, a)
        else:
            raise AttributeError

    def write(self, oname):
        self.tr.to_netcdf(oname)

    def read(self, rundir='.'):
        params = nml.NmlFile(pjoin(rundir, 'gud_params.nml')).merge()
        traits = nml.NmlFile(pjoin(rundir, 'gud_traits.nml')).merge()
        self.params = params
        self.traits = traits
        nmls = [params, traits]

        nplank = len(traits['pcmax'])
        ngroup = len(params['grp_nplank'])
        dimsize = {
            'plankton': nplank,
            'group': ngroup,
            }

        fname = pjoin(rundir, 'radtrans_params.nml')
        if not exists(fname):
            fname = pjoin(rundir, 'gud_radtrans.nml')
        if exists(fname):
            self.radtrans = radtrans = nml.NmlFile(fname).merge()
            nmls.append(radtrans)
            try:
                asize = radtrans['asize']
            except:
                asize = params['asize']
            nopt = len(asize)
            dimsize['opttype'] = nopt
            wb = radtrans.pop('rt_wbcenters', None)
            wbe = radtrans.pop('rt_wbedges', None)

        if wb is None:
            wb = params.pop('gud_waveband_centers', np.r_[0.])
            wbe = params.pop('gud_waveband_edges', np.r_[350., 700.])

        nlam = len(wb)
        dimsize['waveband'] = nlam

        self.nplank = nplank
        self.ngroup = ngroup
        self.nlam = nlam

        # try these size in order
        dimlist = [
            (1, ()),
            (ngroup, ('group',)),
            (nplank, ('plankton',)),
            (ngroup*ngroup, ('group', 'group')),
            (ngroup*nplank, ('group', 'plankton')),
            (nplank*nplank, ('plankton', 'plankton')),
            (nlam*nplank, ('plankton', 'waveband')),
            (nlam, ('waveband',)),
            ]

        if 'opttype' in dimsize:
            dimlist.extend([
                (nopt, ('opttype',)),
                (nopt*nlam, ('opttype', 'waveband')),
                ])

        data = OrderedDict()
        unmatched = OrderedDict()
        for nmld in nmls:
            for k in sorted(nmld):
                v = nmld[k]
                if v.dtype.kind == 'S':
                    v = np.array([s.rstrip() for s in v])

                for n, dims in dimlist:
                    if v.size == n:
                        break
                else:
                    dims = None
                    sys.stderr.write('Dimension not matched: {}\n'.format(v.size))
                    unmatched[k] = v
                    continue

                if dims is not None:
                    sh = tuple(dimsize[d] for d in dims)
                    v = v.reshape(sh)

                if v.dtype.kind == 'S':
                    print k,v
                    n = max(128, v.dtype.itemsize)
                    v = v.astype( '|S'+str(n))

                if k in dimsize:
                    k = 'my' + k
                data[k] = (dims, v)

        coords = {
            'waveband': wb,
            }
        ndig = int(np.log10(nplank)) + 1
        if 'grp_names' in params:
            coords['group'] = data['grp_names'][1]
            if 'igroup' in traits:
                l = []
                for i in range(nplank):
                    g = traits['group'][i]
                    j = traits['igroup'][i]
                    name = data['grp_names'][1][g-1] + '{0:0{d}d}'.format(j, d=ndig)
                    l.append(name)
                coords['plankton'] = l

        self.tr = xr.Dataset(data, coords)
        for k in unmatched:
            self.tr[k] = unmatched[k]

        fname = pjoin(rundir, 'gud_indices.nml')
        if exists(fname):
            self.indices = nml.NmlFile(fname).merge()
            attrs = OrderedDict()
            for k in self.indices:
                i = int(self.indices[k])
#                attrs[k] = i
                if k != 'n_gud':
                     i -= 1
                setattr(self, k.replace('_', ''), i)
                attrs[k.replace('_', '')] =  i

            self.names = self.ngud*['']
            for k in self.indices:
                if k[:2] == 'i_':
                    i = int(self.indices[k]) - 1
                    name = k[2:]
                    if 'e_'+name in self.indices:
                        e = int(self.indices['e_'+name])
                        for ii in range(i, e):
                            self.names[ii] = name + '{0:0{d}d}'.format(ii-i+1, d=ndig)
                    else:
                        self.names[i] = name

            self.tr['tracers'] = self.names
            self.tr.tracers.attrs.update(attrs)

        return self.tr


if __name__ == '__main__':
    rundir, ncname = sys.argv[1:]
    info = RunInfo(rundir)
    info.write(ncname)

