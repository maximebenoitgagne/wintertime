def int_or_None(s):
    if len(s):
        res = int(s)
    else:
        res = None
    return res

def str2slc(s, forceslice=False):
    if s is None: return slice(None)
    v = map(int_or_None, s.split(':'))
    if len(v) == 1:
        if forceslice:
            v[:0] = [None]
        else:
            return v[0]
    return slice(*v)

def str2slcs(s):
    return tuple(map(str2slc, s.split(',')))

def _slc2str(s):
    if s is Ellipsis:
        return '...'
    elif isinstance(s, slice):
        sss = (s.start, s.stop, s.step)
        if sss[2] is None:
            sss = sss[:2]
        return ':'.join(s is not None and str(s) or '' for s in sss)
    else:
        return str(s)

def slc2str(s, shape=None, delim=None):
    if shape is not None:
        s = compute(s, shape)
    else:
        try:
            iter(s)
        except ValueError:
            s = [s]
    res = ','.join(_slc2str(s1) for s1 in s)
    if delim is not None:
        res = delim[0] + res + delim[1]
    return res

def fixndim(ss, ndim):
    # turn into list
    try:
        iter(ss)
    except:
        ss = [ss]
    else:
        ss = list(ss)

    # maatxh ndim
    if Ellipsis in ss:
        i = ss.index(Ellipsis)
        ss[i:i+1] = (ndim + 1 - len(ss))*[slice(None)]
    elif len(ss) < ndim:
        ss.extend((ndim - len(ss))*[slice(None)])

    while Ellipsis in ss:
        ss[ss.index(Ellipsis)] = slice(None)

    return tuple(ss)

def indices(ss, dims):
    # turn into list
    try:
        iter(ss)
    except:
        ss = [ss]
    else:
        ss = list(ss)

    # maatxh ndim
    if Ellipsis in ss:
        i = ss.index(Ellipsis)
        ss[i:i+1] = (len(dims) + 1 - len(ss))*[slice(None)]
    elif len(ss) < len(dims):
        ss.extend((len(dims) - len(ss))*[slice(None)])

    while Ellipsis in ss:
        ss[ss.index(Ellipsis)] = slice(None)

    return tuple(s.indices(d) for s,d in zip(ss, dims))

def compute(ss, dims):
    inds = indices(ss, dims)
    slc = []
    for i in inds:
        if i[2] == 1:
            i = i[:2]
        slc.append(slice(*i))
    return tuple(slc)

def scale(slices, num, denom=1):
    l = []
    for s in slices:
        if isinstance(s, slice):
            start = None if s.start is None else s.start*num//denom
            stop = None if s.stop is None else s.stop*num//denom
            step = None if s.step is None else s.step*num//denom
            s = slice(start, stop, step)
        elif s is Ellipsis:
            pass
        else:
            s = s*num//denom
        l.append(s)
    return tuple(l)

