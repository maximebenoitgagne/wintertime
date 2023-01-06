import string
import re
from collections import OrderedDict

class CurrentCondTracker(object):
    '''keeps track of current preprocessor conditional
    '''
    def __init__(self):
        self.l = []

    def put(self, line):
        lines = line.strip(' #')
        if lines[:2] == 'if':
            self.l.append(line+'\n')
        elif lines[:4] == 'else':
            self.l[-1] += '#else\n'
        elif lines[:3] == 'end':
            self.l.pop()
        else:
            print 'ERROR cond', line

    def get(self):
        cond1 = ''.join(self.l)
        cond2 = len(self.l)*'#endif\n'
        return cond1,cond2


class CondTracker(object):
    '''keeps track of preprocessor conditional level.
    Stores incomplete conditionals for retrieval.
    '''
    def __init__(self):
        self.reset()

    def put(self, line):
        lines = line.strip(' #')
        if lines[:2] == 'if':
            self.l.append(line)
            self.n += 1
        elif lines[:4] == 'else':
            if self.l:
                self.l[-1] += '\n' + line
            else:
                self.l = [line]
        elif lines[:3] == 'end':
            if self.n:
                if self.l and self.l[-1][:3] == '#if':
                    self.l.pop()
                else:
                    self.l.append(line)
                self.n -= 1
        else:
            print 'ERROR cond', line

    def popl(self):
        res = list(self.l)
        self.l = []
        return res

    def pop(self):
        res = ''.join(s+'\n' for s in self.l)
        self.l = []
        return res

    def popendl(self):
        res = []
        while self.l:
            if self.l[0][:3] == '#if':
                break
            res.append(self.l.pop(0))
        return res

    def popend(self):
        return ''.join(s+'\n' for s in self.popendl())

    def reset(self):
        self.l = []
        self.n = 0

        
noread = ['locals']

typemap = OrderedDict([
    ('logical', '_l'),
    ('character', '_c'),
    ('integer', '_i'),
    ('_rl', '_r'),
])

uninitmap = OrderedDict([
    ('logical', '_L'),
    ('character', '_C'),
    ('integer', '_I'),
    ('_rl', '_RL'),
])

dimre = re.compile(r'^([^(]*)(\([^)]*\))$')

def parse(varstring):
    # we need to fill these
    nmls = OrderedDict()
    coms = OrderedDict()
    code = OrderedDict()
    uninits = OrderedDict()
    allparams = []

    # these track progress
    nmlname = None
    allcond = CurrentCondTracker()
    nmlcond = CondTracker()
    comconds = dict((k, CondTracker()) for k in typemap.values())
    codeconds = CondTracker()
    undefcond = []
    codecomm = []
    for line in varstring.split('\n'):
        line = line.rstrip()
        lines = line.lstrip()

        # preprocessor directive
        if lines[:1] == '#':
            allcond.put(line)
            nmlcond.put(line)
            for k in comconds:
                comconds[k].put(line)
            codeconds.put(line)
            undefcond.append(line)
            continue

        # new namelist
        if lines[:1] == '&':
            # finish previous
            if nmlname:
                if nml:
                    nml[-1] = nml[-1][:-1] + (nmlcond.popend(),)
                for sfx in coml:
                    if coml[sfx]:
                        coml[sfx][-1] = coml[sfx][-1][:-1] + (comconds[sfx].popend(),)
                codecomm = []
                codel.extend(codeconds.popendl())
                while undefcond:
                    if undefcond[0][:3] == '#if':
                        break
                    uninits[nmlname].append(undefcond.pop(0))

            # trailing '-' signals namelist not read in
            if lines[-1] == '-':
                nmlname = lines[1:-1]
                noread.append(nmlname.lower())
            else:
                nmlname = lines[1:]

            # set up new lists
            cond1,cond2 = allcond.get()
            nml = []
            nmls[nmlname] = (nml, cond1, cond2)
            coml = OrderedDict([(tp,[]) for tp in typemap.values()])
            coms[nmlname] = (coml, cond1, cond2)
            codel = []
            code[nmlname] = (codel, cond1, cond2)
            uninits[nmlname] = undefcond
            undefcond = []

            # and reset in-namelist conditions
            nmlcond.reset()
            for sfx in coml:
                comconds[sfx].reset()
            codeconds.reset()

            continue

        # split off comment
        if '!' in line:
            i = line.index('!')
            l = line[:i].rstrip()
            comment = line[len(l):]
            line = l
            i = lines.index('!')
            lines = lines[:i].rstrip()
        else:
            comment = ''

        # type
        v = string.split(lines, maxsplit=1)
        if len(v) == 0:
            # just a comment, indent appropriately
            if nmlname:
                if comment and comment[:6] != 6*' ':
                    comment = 6*' ' + comment
                if comment or codel:
                    codecomm.append(comment)
            continue
        elif len(v) == 1:
            # cannot have just a type on a line
            print 'ERROR', line
        tp,var = v
        m = dimre.match(tp)
        if m:
            tp,dims = m.groups()
        else:
            dims = ''

        # variable name and possibly value
        v = string.split(var, '=', maxsplit=1)
        name = v.pop(0).strip()
        assert ' ' not in name
        allparams.append(name)
        nml.append((tp, dims, name, nmlcond.pop(), ''))
        sfx = typemap[tp.lower().rstrip('*0123456789')]
        coml[sfx].append((tp, dims, name, comconds[sfx].pop(), ''))
        if v:
            # it is an assignment, just use original line
            indent = max(6, len(line) - len(line.lstrip()))
            v = string.split(lines, maxsplit=1)
            l = indent*' ' + v[1] + comment
            codel.extend(codecomm+codeconds.popl())
            codecomm = []
            codel.append(l)
        else:
            # declaration without assignment
            # find appropriate UNINIT parameter
            tpk = tp.lower().rstrip('*0123456789')
            if tpk in uninitmap:
                usfx = uninitmap[tpk]
                indent = max(6, len(line) - len(line.lstrip()))
                v = string.split(lines, maxsplit=1)
                l = indent*' ' + v[1] + ' = UNINIT' + usfx #+ ' ' + comment
                uninits[nmlname].extend(undefcond)
                undefcond = []
                uninits[nmlname].append(l)

            # remove empty lines from comments
            while codecomm and codecomm[-1] == '':
                codecomm.pop()

    # finish last
    if nml:
        nml[-1] = nml[-1][:-1] + (nmlcond.popend(),)

    for sfx in coml:
        if coml[sfx]:
            coml[sfx][-1] = coml[sfx][-1][:-1] + (comconds[sfx].popend(),)

    while undefcond:
        if undefcond[0][:3] == '#if':
            break
        uninits[nmlname].append(undefcond.pop(0))

    # clean up
    for k in nmls:
        if k.lower() in noread:
            del uninits[k]

        coml,_,_ = coms[k]
        for tp in typemap.values():
            if tp in coml and len(coml[tp]) == 0:
                del coml[tp]

    locals = nmls.pop('locals', ([], '', ''))
    localdecls = coms.pop('locals', ([], '', ''))

    # find variables that have denominators
    denomparams = []
    for param in allparams:
        if param[-6:] == '_denom':
            if param[:-6] in allparams:
                denomparams.append(param[:-6])
            else:
                print 'DENOM without PARAM:', param

    ns = dict(
        nmls=nmls,
        coms=coms,
        code=code,
        uninits=uninits,
        locals=locals,
        localdecls=localdecls,
        denomparams=denomparams,
        noread=noread,
        )

    return ns

if __name__ == '__main__':
    for k in nmls:
        print '&'+k, nml
        for line in code[k]:
            print line
    for k,comm in coms.items():
        print k,comm
