
def fillslices(index, active, lens):
    if type(index) != type(()): index = (index,)
    length = len(index)
    dims = len(active)
    fixed = []
    newactive = []
    i = 0
    for ind in index:
        if ind is Ellipsis:
            fixed += (dims-length+1)*[slice(None)]
            newactive += active[i:i+dims-length+1]
            i += dims-length+1
            length = len(fixed)
        elif isinstance(ind, slice):
            fixed.append(ind)
            newactive.append(active[i])
            i += 1
        else:
            if ind < 0: ind += lens[active[i]]
            if ind >= lens[active[i]]:
                raise IndexError
            fixed.append(slice(ind,ind+1,1))
            i += 1

    if i < dims:
        fixed += (dims-i) * [slice(None)]
        newactive += active[i:]

    return fixed,newactive


def slicelen(s):
    return -((s.start-s.stop)//s.step)


def subscribe(s,i,n):
    if isinstance(i,slice):
        istart,istop,istep = i.indices(n)
        start = s.start+s.step*istart
        stop  = s.start+s.step*istop
        step  = s.step*istep
    else:
        if i < 0: i += n
        start = s.start+s.step*i
        if start > s.stop-1:
            raise IndexError
        stop = start + 1
        step = 1

    return slice(start,stop,step)


class MSlice(object):
    def __init__(self, shape=None, slices=None, active=None):
        if slices is not None:
            self.s = tuple(slices)
            self.ndim = len(slices)
            self.active = active
        else:
            self.s = tuple(slice(0,d,1) for d in shape)
            self.ndim = len(shape)
            self.active = range(self.ndim)

        self.lens = [ slicelen(s) for s in self.s ]

    def __str__(self):
        #return 'MSlice(' + ', '.join( str(s) for s in self.s ) + '; ' + repr(self.active) + ')'
        #return 'MSlice(' + ', '.join( str(s) for s in self.s ) + ')'
        l = [ str(s.start) for s in self.s ]
        for a in self.active:
            s = self.s[a]
            l[a] = ':%d' % (s.stop)
            if s.start != 0:
                l[a] = '%d%s' % (s.start,l[a])

            if s.step != 1:
                l[a] += ':%d' % s.step

        return ', '.join(l)

    def __repr__(self):
        return 'MSlice[' + str(self) + ']'

    def fill(self, inds):
        indices,active = fillslices(inds, self.active, self.lens)
        return indices, active

    def __getitem__(self, inds):
        #indices,active = self.fill(inds)  # fillslices(inds, self.active, self.lens)
        indices,active = fillslices(inds, self.active, self.lens)
        slices = list(self.s)
        for a,ind in zip(self.active,indices):
            slices[a] = subscribe(self.s[a], ind, self.lens[a])

        return MSlice(slices=slices,active=active)

    @property
    def shape(self):
        return tuple( self.lens[a] for a in self.active )

    @property
    def augshape(self):
        return tuple(self.lens)

