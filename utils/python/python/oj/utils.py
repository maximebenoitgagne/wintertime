
def _fmtslice1(s):
    try:
        v = s.start, s.stop, s.step
    except:
        return str(s)
    if v[2] is None:
        v = v[:2]
    return ':'.join('' if x is None else str(x) for x in v)

def fmtslice(s, sep=',', bracket=False):
    try:
        iter(s)
    except:
        res = _fmtslice1(s)
    else:
        res = sep.join(map(_fmtslice1, s))

    if bracket:
        res = '[' + res + ']'

    return res
