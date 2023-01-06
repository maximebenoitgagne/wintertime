import sys
import re
import glob
import numpy as np
from collections import OrderedDict
from dicts import CaselessOrderedDict

# for python2.5
try: next
except NameError:
    def next ( obj ): return obj.next()

debug = True
_currentline = ''

class ParseError(ValueError):
    def __str__(self):
        metafile = self.args[0]
        lines = self.args[1:]
        try:
            name = metafile.name
        except AttributeError:
            name = metafile

        return '\n'.join(('in metafile: '+name,)
                         + lines
                         + ('in: ' + _currentline,))


# these deal with comments in the metafile

_comment_pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )

def _comment_replacer(match):
    s = match.group(0)
    if s.startswith('/'):
        return ""
    else:
        return s

def strip_comments(text):
    """ strips C and C++ style comments from text """
    return re.sub(_comment_pattern, _comment_replacer, text)


_logical_pattern = re.compile(r"\.?([tTfF])[a-zA-Z]*\.?$")
_string_pattern = re.compile(r"""['"]([^'"]*)['"]$""")

def parse1(s):
    """ convert one item to appropriate type """
#    m = _logical_pattern.match(s)
#    if m:
#        s = m.group(1)
#        assert s.lower() in ['t', 'f']
#        return s.lower() == 't'
    if s.lower() in ['t', 'f', '.true.', '.false.', 'true', 'false']:
        return s.lower().strip('.') == 't'

    m = _string_pattern.match(s)
    if m:
        s = m.group(1)
        # unquote quotes
        s = re.sub(r"''","'",s)
        return s

    if '.' in s or 'e' in s.lower():
        try:
            s = float(s)
        except ValueError:
            pass
        return s
    else:
        try:
            return int(s)
        except ValueError:
            return s
#            raise ParseError("Cannot parse value: " + s)


def readnml(fname):
    sys.stderr.write('Warning: readnml is deprecated, use readnmlfile instead.\n')
    try:
        lines = open(fname)
    except TypeError:
        lines = iter(fname)

    d = CaselessOrderedDict()
    nml = ''
    var = ''
    for line in lines:
        line = line.strip()
        if line in ['&','/']:
            tp = 'endnml'
        elif line.startswith('&'):
            tp = 'startnml'
        elif '=' in line:
            tp = 'var'
        else:
            tp = 'cont'

        if var != '':
            if tp in ['endnml','var']:
                vals = re.split(r'\s*,\s*',val.rstrip(', '))
                lst = []
                for val in vals:
                    if '*' in val:
                        count,val = val.split('*')
                        count = int(count)
                    else:
                        count = 1
                    val = parse1(val)
                    lst.extend(count*[val])
                    d[nml][var] = np.array(lst)

                var = ''
                val = ''
            elif tp in ['cont']:
                if line[:1] == ' ':
                    line = line[1:]
                print(val + '##' + line)
                val += line
            else:
                raise ParseError()

        if tp == 'startnml':
            nml = line[1:]
            d[nml] = CaselessOrderedDict()
        elif tp == 'var':
            m = re.match(r'(\w+)(\(\s+([^\)\s]*)\s+\))?\s*=\s*(.*$)$',line)
            var,_,ind,val = m.groups()
            if ind is not None:
                ParseError('Indexing not implemented')
            if val[-1] != ',': val = val + ','
        elif tp == 'endnml':
            nml = ''
        else:
            pass

    if nml != '':
        raise ParseError()

    return d


def parse1count(s):
    if '*' in s:
        count,s = s.split('*')
        count = int(count)
    else:
        count = 1

    val = parse1(s)
    return count*[val]


def parsenml(text):
    nml = CaselessOrderedDict()

    vardefs = re.split(r'\s+(\w+)(?:\(\s*([^\)\s]*)\s*\))?\s*=\s*', text.rstrip('/& '))
    nmlname = vardefs.pop(0).lstrip('&/')
    assert len(vardefs) % 3 == 0
    names = vardefs[0::3]
    indxs = vardefs[1::3]
    valss = vardefs[2::3]

    for name,indx,vals in zip(names,indxs,valss):
        if indx is not None:
            raise ParseError('Indexing not implemented')

        vals = vals.rstrip(', ')
        if vals[:1] in ['"', "'"]:
            d = vals[0]
            assert vals[-1] == d
            vals = re.split(d + r' *, *' + d, vals[1:-1])
        else:
            vals = re.split(r' *[, ] *', vals)
        # concatenate lists into array
        vals = np.r_[ tuple( parse1count(val) for val in vals ) ]
        nml[name] = vals
        
    return nmlname,nml


def readnmlfile(fname):
    try:
        lines = open(fname)
    except TypeError:
        lines = iter(fname)

    d = CaselessOrderedDict()
    for line in lines:
        line = line.strip('\n')
        if line[:1] == ' ':
            line = line[1:]
        if line.startswith('&'):
            nml = line
            while nml[-1] not in '&/':
                line = next(lines).strip('\n')
                if line[:1] == ' ':
                    line = line[1:]
                if not line.startswith('#'):
                    if '=' in line:
                        nml += ' '
                    nml = nml + line

            name,nml = parsenml(nml)
            d[name] = nml
        elif line == '' or line.startswith('#'):
            pass
        else:
            raise ParseError('Unexpected line: ' + line)

    return d


def writenml(f, d, name):
    try:
        f.write
    except AttributeError:
        f = open(f, 'w')
        needclose = True
    else:
        needclose = False

    f.write(' &{}\n'.format(name))
    try:
        d.keys
    except AttributeError:
        d = d.__dict__
    keys = d.keys()
    if not isinstance(d, CaselessOrderedDict):
        keys = sorted(keys)
    for k in keys:
        v = d[k]
        try:
            s = ', '.join(repr(f) for f in v)
        except TypeError:
            try:
                s = repr(v.tolist())
            except AttributeError:
                s = repr(v)
        f.write(' {}= {},\n'.format(k, s))
    f.write(' /\n')

    if needclose:
        f.close()


class NmlFile(CaselessOrderedDict):
    def __init__(self, fname):
        CaselessOrderedDict.__init__(self, readnmlfile(fname))

    def merge(self):
        d = CaselessOrderedDict()
        for k,v in self.items():
            d.update(v)

        return d


class NmlFiles(NmlFile):
    def __init__(self, *args):
        if len(args) == 1 and '*' in args[0]:
            globpattern = args[0]
            args = glob.glob(globpattern)
            if len(args) == 0:
                raise IOError("No files found for {}".format(globpattern))

        super(NmlFiles, self).__init__(args[0])
        for fname in args[1:]:
            self.update(readnmlfile(fname))


# should do this:
#
#   for each namelist (joined in one line):
#
#     from the start, look for r"^[^']\s+(\w+)(?:\(...\))?\s*=\s*" or "'"
#     whichever comes first:
#
#       if "'", go past quoted string (taking '' into account)
#       if "...=", split then string and store
#
# re.match(r"([^=']*'|[^']*a=)", "abcdefa=bde'fgh'").groups()
# re.split(r"(?:('[^']*')|(a=))", "a=bc'defa=bd''e'fga=h'")



