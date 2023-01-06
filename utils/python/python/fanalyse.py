'''find all write access to variables in fortran files.
First direct assignments, then indirect writes through subroutine calls.
'''
import sys
import os
import re
import glob
import shlex
import json
import string
import cgi
from pyparsing import nestedExpr, ParseException

SRPATT = re.compile(r'^ ....  *(program|subroutine|(?:[a-zA-Z0-9_*]+ +)?function) *(\w+) *(\([^!]*)?', re.I)
CALLPATT = re.compile(r'^( ....  *call *)(\w+) *(\([^!]*)', re.I)
INCLUDEPATT = re.compile(r'^# *include +"([^"]+)"')
COMMONPATT = re.compile(r'^ ....  *common */ *(\w+) */ *(.*) *$', re.I)
INLINECOMMENTPATT = re.compile(r'\!.*$')
CONTLINEPATT = re.compile(r'^     [^ ]')
parenexpr = nestedExpr('(', ')')
angleexpr = nestedExpr('<', '>')

_defdirs = ['model/inc',
            'model/src',
            'eesupp/inc',
            'eesupp/src',
            'pkg/*']

_defexcludes = ['pkg/atm_ocn_coupler',
                'pkg/aim_compon_interf',
                'pkg/chronos',
                'pkg/cfc',
                'pkg/dic',
                'pkg/ecco',
                'pkg/autodiff',
                ]

_IGNOREHEADERS = ['mpif.h',
                  'netcdf.inc',
                  'tamc.h',
                  'tamc_keys.h',
                  'PACKAGES_CONFIG.h',
                  'AD_CONFIG.h',
                  'BUILD_INFO.h',
                  'ecco_cost.h',
                  'ecco_ad_check_lev1_dir.h',
                  'ecco_ad_check_lev2_dir.h',
                  'ecco_ad_check_lev3_dir.h',
                  'ecco_ad_check_lev4_dir.h',
                  'dic_ad_check_lev1_dir.h',
                  'dic_ad_check_lev2_dir.h',
                  'dic_ad_check_lev3_dir.h',
                  'dic_ad_check_lev4_dir.h',
                  'ebm_ad_check_lev2_dir.h',
                  'ebm_ad_check_lev4_dir.h',
                  'checkpoint_lev1_template.h',
                  'checkpoint_lev1_directives.h',
                  'checkpoint_lev2_directives.h',
                  'checkpoint_lev3_directives.h',
                  'checkpoint_lev4_directives.h',
                  'ctrparam.h',
                  'adcommon.h',
                  'f_hpm.h',
                 ]

U = True
I = False
_KNOWNINTENTS = {'flush': [I],
                 'mpi_wait': [I,U,U],
                 'mpi_comm_size': [I,U,U],
                 'mpi_comm_rank': [I,U,U],
                 'mpi_cart_rank': [I,I,U,U],
                 'mpi_cart_coords': [I,I,I,U,U],
                 'mpi_cart_create': [I,I,I,I,I,U,U],
                 'mpi_bcast': [U,I,I,I,I,U],
                 'mpi_recv': [U,I,I,I,I,I,U,U],
                 'mpi_send': [I,I,I,I,I,I,U],
                 'mpi_isend': [I,I,I,I,I,I,U,U],
                 'mpi_type_commit': [I,U],
                 'mpi_type_contiguous': [I,I,U,U],
                 'mpi_type_vector': [I,I,I,I,U,U],
                 'mpi_type_hvector': [I,I,I,I,U,U],
                 'mpi_allreduce': [I,U,I,I,I,I,U],
                }

_IGNORESUBPATTS = ['active_read_',
                   'active_write_',
                   'adactive_',
                   'autodiff_',
                   'ampi_',
                   'mpi_waitall$',
                   'mpi_barrier$',
                  ]
_IGNORESUBPATTS = [ re.compile(s) for s in _IGNORESUBPATTS ]

def dump(fname, obj):
    with open(fname, 'w') as f:
        json.dump(obj, f)

def load(fname):
    with open(fname) as f:
        obj = json.load(f)
    return obj

def pathdict(dirglobs=_defdirs, ext='.h', excludes=_defexcludes, rootdir=''):
    '''return dictionary mapping names of files with extension ext in directories
    matching dirglobs under rootdir to their paths
    '''
    paths = {}
    for dg in dirglobs:
        for dir in glob.glob(os.path.join(rootdir,dg)):
            if dir.rstrip('/') not in excludes:
                for path in glob.glob('{}/*{}'.format(dir,ext)):
                    d,name = os.path.split(path)
                    if name in paths:
                        print 'duplicate:', paths[name], path
                    paths[name] = path
    return paths

def joinacrosscpp(lines, i=0, argstr=''):
    while lines[i+1][:1] == '#' or CONTLINEPATT.match(lines[i+1]):
        i += 1
        #lline = lines[i].lower()
        lline = lines[i]
        if CONTLINEPATT.match(lline):
            arg = INLINECOMMENTPATT.sub('', lline[6:]).strip()
            argstr = argstr + arg

    return i,argstr

def stripallcpp(fname):
    lines = []
    linenum = []
    with open(fname) as fid:
        for n,line in enumerate(fid):
            line = INLINECOMMENTPATT.sub('', line.rstrip().lower())
            if line[:1] == ' ':
                lines.append(line)
                linenum.append(n+1)

    return lines, linenum

def stripcpp(fname):
    with open(fname) as fid:
        lines = fid.readlines()

    # remove comments and preprocessor directives, except includes
    i = 0
    n = 1
    linenum = [1]
#    includes = []
    while i < len(lines):
        # remove newline
        lines[i] = INLINECOMMENTPATT.sub('', lines[i].rstrip())
#        m = re.match(r'# *include +"([^"]+)"', lines[i+1])
#        if m:
#            includes.append(m.group(1))

        if lines[i][:1] == ' ':
            lines[i] = lines[i].lower()

        if lines[i][:1] != ' ' and not INCLUDEPATT.match(lines[i]):
            lines.pop(i)
        else:
            linenum[i:i+1] = [n]
            i += 1

        n += 1

    return lines, linenum  #, includes

def joincontinuation(lines):
    '''joins continuation lines in place
    '''
    # join continuation lines
    # assumes that cpp directives and comments have already been stripped
    i = 0
    while i < len(lines)-1:
        if CONTLINEPATT.match(lines[i+1]):
            line = INLINECOMMENTPATT.sub('', lines.pop(i+1)[6:].strip())
            lines[i] = lines[i] + ' ' + line
        else:
            i += 1

def parseinclude(incname, paths, commsbyinc=None, varsbyinc=None):
    if commsbyinc is None: commsbyinc = {}
    if varsbyinc is None: varsbyinc = {}

    if incname in varsbyinc:
        return commsbyinc[incname], varsbyinc[incname]

    try:
        fname = paths[incname]
    except KeyError:
        if incname not in _IGNOREHEADERS:
            print 'Missing:', incname
        return {},{}

    commons = {}
    globvars = {}
    lines,_ = stripcpp(fname)
    joincontinuation(lines)
    for line in lines:
        m = INCLUDEPATT.match(line)
        if m:
            inc = m.group(1)
            if inc in paths:
                c,g = parseinclude(m.group(1), paths, commsbyinc, varsbyinc)
                commons.update(c)
                globvars.update(g)
            else:
                if inc not in _IGNOREHEADERS:
                    print 'WARNING: {}: Missing header {}'.format(fname, inc)
            continue

        m = COMMONPATT.match(line)
        if m:
            commname,varstr = m.groups()
            varnames = [ s.strip() for s in varstr.split(',') ]
            if commname in commons:
                if commons[commname] != incname:
                    print 'WARNING: common block', commname, 'in', paths[commons[commname]], fname
            else:
                commons[commname] = incname
                for varname in varnames:
                    if varname in globvars:
                        c = globvars[varname]
                        inc = commons[c]
                        if inc == incname and c == commname:
                            print 'WARNING: {}: {} appears twice in /{}/'.format(fname, varname, commname)
                        else:
                            print 'WARNING:', fname+':', varname, 'already in', paths[inc]+':'+c
                    globvars[varname] = commname

            continue

    commsbyinc[incname] = commons
    varsbyinc[incname] = globvars

    return commons, globvars

#def parseallincludes(paths):
#    c = {}
#    g = {}
#    for incname in paths:
#        parseinclude(incname, paths, c, g)
#    return c,g

def getglobals(fname, paths, commsbyinc=None, varsbyinc=None, commons=None, globvars=None):
    if commsbyinc is None: commsbyinc = {}
    if varsbyinc is None: varsbyinc = {}
    if commons is None: commons = {}
    if globvars is None: globvars = {}

    lines,_ = stripcpp(fname)
    joincontinuation(lines)
    srname = ''
    for line in lines:
        m = SRPATT.match(line)
        if m:
            _,srname,_ = m.groups()
            if srname in commons:
                print 'WARNING: {}: S/R {} appears twice in'.format(fname, srname)
            else:
                commons[srname] = {}
                globvars[srname] = {}
            continue

        m = INCLUDEPATT.match(line)
        if m:
            inc = m.group(1)
            if srname == '':
                if not 'OPTIONS.h' in inc and not 'CONFIG.h' in inc and not inc in _IGNOREHEADERS:
                    print '{}: skipping pre-subroutine include {}'.format(fname, inc)
            elif inc in paths:
                c,g = parseinclude(m.group(1), paths, commsbyinc, varsbyinc)
                commons[srname].update(c)
                globvars[srname].update(g)
            else:
                if inc not in _IGNOREHEADERS:
                    print 'WARNING: {}: Missing header {}'.format(fname, inc)
            continue

        m = COMMONPATT.match(line)
        if m:
            commname,varstr = m.groups()
            varnames = [ s.strip() for s in varstr.split(',') ]
            globvar = {}
            for varname in varnames:
                if varname in globvars[srname]:
                    c = globvars[srname][varname]
                    inc = commons[srname][c]
                    if inc is None:
                        print 'WARNING:', fname+':', varname, 'in', c, commname
                    else:
                        print 'WARNING:', fname+':', varname, 'in', paths[inc]+':'+c, fname+':'+commname
                globvar[varname] = commname

            if commname in commons[srname]:
                inc = commons[srname][commname]
                vs = set( k for k,v in globvars[srname].items() if v == commname )
                if inc is None:
                    if vs.symmetric_difference(globvar.values()):
                        print 'WARNING: {}: /{}/ appears twice'.format(fname, commname)
                else:
                    print 'WARNING:', fname+': common block', commname, 'in', paths[inc], fname
                # will overwrite commons and update globvars

            globvars[srname].update(globvar)
            commons[srname][commname] = None

            continue

    return globvars, commons

def getallglobals(fnames, incpaths, commsbyinc=None, varsbyinc=None, commons=None, globvars=None):
    if commsbyinc is None: commsbyinc = {}
    if varsbyinc is None: varsbyinc = {}
    if commons is None: commons = {}
    if globvars is None: globvars = {}
    for fname in fnames:
        getglobals(fname, incpaths, commsbyinc, varsbyinc, commons, globvars)

    return globvars, commons

    
#def parseincludes(paths, incnames=None, commons={}, globvars={}):
#    if incnames is None:
#        incnames = paths.keys()
#
#    for incname in incnames:
#        try:
#            fname = paths[incname]
#        except KeyError:
#            if incname not in _IGNOREHEADERS:
#                print 'Missing:', incname
#            continue
#
#        lines,_ = stripcpp(fname)
#        lines = joincontinuation(lines)
#        for line in lines:
#            m = INCLUDEPATT.match(line)
#            if m:
#                parseincludes(paths, [m.group(1)], commons, globvars)
#                continue
#
#            m = COMMONPATT.match(line)
#            if m:
#                name,varstr = m.groups()
#                varnames = [ s.strip() for s in varstr.split(',') ]
#                if name in commons:
#                    if commons[name] != incname:
#                        print 'WARNING: common block', name, 'in', paths[commons[name]], paths[incname]
#                else:
#                    commons[name] = incname
#                    for varname in varnames:
#                        if varname in globvars:
#                            c = globvars[varname]
#                            print 'WARNING:', varname, 'in', paths[commons[c]]+':'+c, paths[commons[name]]+':'+name
#                        globvars[varname] = name
#
#                continue
#
#    return commons, globvars

def getdirect(fname):
    # strip comments, cpp directives
    lines,linenum = stripallcpp(fname)

    # identify subroutines
    srfiles = {}
    # subargs[srname] = [arg, ...]
    subargs = {}
    # calls[srname] = [(called,args), ...]
    calls = {}
    srname = ''
    for i in range(len(lines)):
        lline = lines[i].lower()
        m = SRPATT.match(lline)
        if m:
            tp,srname,argstr = m.groups()
            srfiles[srname] = (fname, linenum[i])
            if argstr is None:
                args = []
            else:
                while lines[i+1][:1] == '#' or CONTLINEPATT.match(lines[i+1]):
                    i += 1
                    lline = lines[i].lower()
                    if CONTLINEPATT.match(lline):
                        arg = re.sub(r'!.*$', '', lline[6:]).strip()
                        argstr = argstr + arg

                argstr = argstr.strip()
                assert argstr[0] == '('
                assert argstr[-1] == ')'
                args = [ s.strip() for s in argstr[1:-1].split(',') ]

            if srname in subargs:
                print 'WARNING: {}: subroutine encountered twice: {} {} {}'.format(fname, srname, len(subargs[srname]), len(args))

            subargs[srname] = args
            calls[srname] = []
            continue

        m = CALLPATT.match(lline)
        if m:
            called = m.group(2)
            argstr = m.group(3)
            i0 = i
            i,argstr = joinacrosscpp(lines, i, argstr)
            try:
                argl, = parenexpr.parseString(argstr).asList()
            except ParseException:
                print '{}:{}'.format(fname, linenum[i]), argstr
                raise
            args1 = ' '.join([ s for s in argl if type(s) != type([]) and s[0] not in "'" + '"' ])
            def argvars(args1):
                for s in args1.split(','):
                    m1 = re.match(r' *(\w*)', s)
                    yield m1.group(1)

            args = list(argvars(args1))
            if srname == '':
                print fname, called
            l = str(linenum[i0])
            if i != i0: l += '-{}'.format(linenum[i])
            calls[srname].append((l, called, args))
            continue

    # direct[srname][varname] = [line1, line2, ...]
    direct = {}
    srname = ''
    for i,line in enumerate(lines):
        m = SRPATT.match(line)
        if m:
            srname = m.group(2)
            direct[srname] = {}
            continue

        m = re.match(r' ....  *([a-zA-Z_][a-zA-Z0-9_]*) *(?:\([^)]*\))? *=', line)
        if m:
            name = m.group(1)
            if srname == '':
                print '{}:{}: assignment outside of subroutine: {}'.format(
                      fname,linenum[i],line)
            direct[srname].setdefault(name, []).append(str(linenum[i]))
            continue

    def joinvalues(d):
        s = set()
        for l in d.values():
            s.update(l)
        return s

#    globmod = {}
#    for sr in direct:
#        globmod[sr] = [ v for v in direct[sr] if v in globvars[sr] ]

    return subargs, direct, calls, srfiles

def getdirects(patts, incpaths):
    fnames = [ s for patt in patts for s in glob.glob(patt) ]
    srfiles = {}
    subargs = {}
    calls = {}
    direct = {}
    for fname in fnames:
        s,d,c,f = getdirect(fname)
        for sub in s:
            if sub in subargs:
                print 'WARNING: {}: S/R {} already in {} {} {}'.format(
                      fname, sub, srfiles[sub], len(subargs[sub]), len(s[sub]))
                i = 2
                while str(i)+sub in subargs: i += 1
                s[str(i)+sub] = s[sub]
                d[str(i)+sub] = d[sub]
                c[str(i)+sub] = c[sub]
                f[str(i)+sub] = f[sub]
                del s[sub]
                del d[sub]
                del c[sub]
                del f[sub]

        subargs.update(s)
        calls.update(c)
        direct.update(d)
        srfiles.update(f)

    return subargs, direct, calls, srfiles

def findintents(srname, subargs, direct, calls, cache=None, worklist=None):
    if worklist is None:
        worklist = {}
    try:
        return cache[srname]
    except:
        pass

#    print srname, worklist.keys()
    sys.stdout.flush()
    worklist[srname] = True
            
    myargs = subargs[srname]
    mydirect = direct[srname]
    out = dict((k, k in mydirect) for k in myargs)
#    out = dict.fromkeys(myargs, False)
    for l,sub,args in calls[srname]:
        if sub not in subargs and sub not in cache:
            ignore = False
            for patt in _IGNORESUBPATTS:
                if patt.match(sub):
                    ignore = True
                    break
            if not ignore and sub not in ['print_error', 'print_message']:
                print '{}: skipping {}'.format(srname, sub)
            continue

        if sub == srname:
            print '{}: skipping self'.format(sub)
        elif sub in worklist:
            print 'Avoiding recursion:', srname, sub
        else:
            subout = findintents(sub, subargs, direct, calls, cache, worklist)
            if len(subout) != len(args):
                print sub+': have', len(args), 'needs', len(subout)

            for arg,so in zip(args, subout):
                if arg in out:
                    out[arg] = out[arg] or so

    isout = [ out[a] for a in myargs ]
    if '2'+srname in subargs:
        i = 2
        while str(i)+srname in subargs:
            if str(i)+srname in worklist:
                print 'Avoiding recursion:', srname, str(i)+srname
            else:
                isout2 = findintents(str(i)+srname, subargs, direct, calls, cache, worklist)
                if len(isout2) == len(isout) and isout2 != isout:
                    print 'Mismatch', str(i)+srname, isout, isout2
            i += 1

    if cache is not None:
        cache[srname] = isout

    del worklist[srname]

    return isout

def findallintents(subargs, direct, calls):
    cache = {}
    cache.update(_KNOWNINTENTS)
    for srname in subargs:
        # will fill cache
        findintents(srname, subargs, direct, calls, cache)

    return cache

def findglobmods(srname, subargs, direct, calls, intents, globvars, cache=None, worklist=None):
    if worklist is None:
        worklist = {}

    try:
        return cache[srname]
    except:
        pass

#    print srname, worklist.keys()
    sys.stdout.flush()
    worklist[srname] = True

    if srname[0] in string.digits:
        srname0 = srname.lstrip(string.digits)
    else:
        srname0 = srname
            
    mydirect = direct[srname]
    myglobals = globvars[srname0]
#    globmods = set(mydirect).intersection(myglobals)
    globmods = dict((k,v) for k,v in mydirect.items() if k in myglobals)
    for l,sub,args in calls[srname]:
        if sub not in subargs:
            ignore = sub in _KNOWNINTENTS
            for patt in _IGNORESUBPATTS:
                if patt.match(sub):
                    ignore = True
                    break

            if not ignore and sub not in ['print_error', 'print_message']:
                print '{}: skipping {}'.format(srname, sub)
            continue

        if sub == srname:
            print '{}: skipping self'.format(sub)
        elif sub in worklist:
            print 'Avoiding recursion:', srname, sub
        else:
            submods = findglobmods(sub, subargs, direct, calls, intents, globvars, cache, worklist)
            for k,v in submods.items():
                globmods.setdefault(k, []).append(l)  # '{}:{}'.format(sub, v)
            intent = intents[sub]
            if len(intent) != len(args):
                print sub+': have', len(args), 'needs', len(intent)

            for arg,isout in zip(args, intent):
                if isout and arg in myglobals:
                    globmods.setdefault(arg, []).append(l)  #  '{}({})'.format(sub, args.index(arg)+1)

    if '2'+srname in subargs:
        i = 2
        while str(i)+srname in subargs:
            if str(i)+srname in worklist:
                print 'Avoiding recursion:', srname, str(i)+srname
            else:
                gm = findglobmods(str(i)+srname, subargs, direct, calls, intents, globvars, cache, worklist)
                globmods.update(gm)
            i += 1

    if cache is not None:
        cache[srname] = globmods

    del worklist[srname]

    return globmods

def findallglobmods(subargs, direct, calls, intents, globvars):
    cache = {}
    for srname in globvars:
        findglobmods(srname, subargs, direct, calls, intents, globvars, cache)

    return cache

globformat='\033[1m{}\033[0m'
gmodformat='\033[1;31m{}\033[0m'
argformat='\033[1;34m{}\033[0m'
argmodformat='\033[1;35m{}\033[0m'

def amptodot(s):
    return re.sub(r'^ (....)&', r' \1*', s)

def escapeangles(s):
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s

def markupfile(fname, srfiles, subargs, globvars, globmods, intents, globformat=globformat,
               gmodformat=gmodformat, argformat=argformat, argmodformat=argmodformat, 
               escape=lambda x:x, mklink=lambda x,y:x):
    #lines = map(escape, map(amptodot, open(fname).readlines()))
    lines = map(escape, open(fname).readlines())
    linenum = range(1,len(lines)+1)
    srname = ''
    markup = []
    i = 0
    while i < len(lines):
        markup += '<a name="{}"></a>'.format(i+1)
        line = lines[i]
        m = SRPATT.match(line)
        if m:
            tp,srname,argstr = m.groups()
            srname = srname.lower()
            def format(arg, intent=None, write=True, iswrite=False):
                if srname in subargs and arg.lower() in subargs[srname]:
                    if write and (iswrite or intent):
                        arg = argmodformat.format(arg)
                    else:
                        arg = argformat.format(arg)
                elif srname in globvars and arg.lower() in globvars[srname]:
                    if intent is None:
                        intent = arg.lower() in globmods[srname]
                    if write and intent:
                        arg = gmodformat.format(arg)
                    else:
                        arg = globformat.format(arg)
                return arg

            if argstr and srname in intents:
                ints = intents[srname]
                markup += '      ' + tp + ' ' + srname + '('
                line = argstr[1:]
                pre = ''
                iarg = 0
                while True:
                    markup += pre
                    myargs = [ s.strip(' )') for s in line.strip().split(',') ]
                    for j in range(len(myargs)):
                        arg = myargs[j]
                        if arg != '':
                            if iarg < len(ints):
#                                print subargs[srname][iarg], arg
                                if arg.lower() != subargs[srname][iarg]:
                                    print 'WARNING: {}: argument mismatch {}: {} {}'.format(
                                        srname, iarg+1, subargs[srname][iarg], arg)
                                markup += format(arg, iswrite=ints[iarg])
                            else:
                                markup += arg
                                print '{}: extra argument: {}'.format(srname, arg)
                            iarg += 1
                            if j < len(myargs) - 1:
                                markup += ','
                                if myargs[j+1] != '':
                                    markup += ' '

                    extra = []
                    while lines[i+1][:1] == '#':
                        i += 1
                        extra += lines[i]

                    i += 1
                    if CONTLINEPATT.match(lines[i]):
                        pre = lines[i][:6] + '        '
                        line = lines[i][6:].strip()
                        markup += '\n'
                        markup += extra
                    else:
                        markup += ')\n'
                        markup += extra
                        break 

                continue

        m = CALLPATT.match(line)
        if m:
            if srname == '':
                print fname, called

            prefix,called,argstr = m.groups()
            i1,argstr = joinacrosscpp(lines, i, argstr)
            try:
                argl, = parenexpr.parseString(argstr).asList()
            except ParseException:
                print '{}:{}'.format(fname, linenum[i]), argstr
                raise
            args = []
            sufs = []
            arg = ''
            suf = ''
            def reparen(l):
                res = []
                for s in l:
                    if type(s) == type([]):
                        res += reparen(s)
                    else:
                        res += s
                return '(' + ''.join(res) + ')'

            for s in argl:
                if type(s) == type([]):
                    suf += reparen(s)
                elif s[0] in "'"+'"':
                    suf += s
                else:
                    arg += s
                    if ',' in arg:
                        pr,sf = re.split(r',', arg, maxsplit=1)
                        arg = pr.strip()
                        args.append(arg)
                        arg = sf

                        sufs.append(suf)
                        suf = ''

                        while ',' in arg:
                            pr,sf = re.split(',', arg, maxsplit=1)
                            arg = pr.strip()
                            args.append(arg)
                            arg = sf
                            sufs.append('')

            args.append(arg)
            sufs.append(suf)

            if called.lower() in srfiles:
#                srfile,srline = srfiles[called.lower()]
                s = mklink(called, srfiles[called.lower()])
                myline = prefix + '{}('.format(s)
            else:
                myline = prefix + '{}('.format(called)
            textline = prefix + '{}('.format(called)
            def join(args,sufs,ints):
                for arg,suf,intent in zip(args,sufs,ints):
                    arg = format(arg, intent)
                    yield arg + suf

            try:
                ints = intents[called.lower()]
            except KeyError:
                print 'No intent founds for', called.lower()
                ints = len(args)*[False]
            joined = list(join(args,sufs,ints))
            pre = ''
            for j in joined:
                e = angleexpr.parseString('<'+j+'>').asList()[0]
                text = ''.join( s for s in e if type(s) != type([]) )
                if len(textline+text) > 72:
                    markup += myline
                    pre = ',\n     &        '
                    myline = ''
                    textline = ''

                myline += pre + j
                textline += pre + text
                pre = ', '
                    
            markup += myline + ')\n'

#            args1 = ' '.join([ s for s in argl if type(s) != type([]) and s[0] not in "'" + '"' ])
#            def argvars(args1):
#                for s in args1.split(','):
#                    m1 = re.match(r' *(\w*)', s)
#                    yield m1.group(1)
#            args = list(argvars(args1))

            i = i1 + 1
            continue

#        m = re.match(r'( ....  *)([a-zA-Z_][a-zA-Z0-9_]*)( *\([^)]*\))?( *=.*$)', line)
        m = re.match(r'( ....  *)([^=]*)(=.*$)', line)
        if m:
            #pre,name,args,rhs = m.groups()
            pre,lhs,rhs = m.groups()
            try:
                # check parenthesis on lhs are balanced, or '=' may be in a keyword argument
                _ = parenexpr.parseString('(' + lhs + ')').asList()[0]
            except ParseException:
                iscode = False
            else:
                if '(' in lhs:
                    lhs,args = re.split(r'\(', lhs, maxsplit=1)
                    args = '(' + args
                else:
                    args = ''
                if srname == '':
                    print '{}:{}: assignment outside of subroutine: {}'.format(
                          fname,linenum[i],line)
                else:
                    lhs = format(lhs, iswrite=True)
    #                if lhs.lower() in globvars[srname]:
    #                    if lhs.lower() in globmods[srname]:
    ##                        line = re.sub(lhs, '#{}#'.format(lhs), line, 1, re.I)
    #                        lhs = gmodformat.format(lhs)
    #                    else:
    ##                        line = re.sub(lhs, '|{}|'.format(lhs), line, 0, re.I)
    #                        lhs = globformat.format(lhs)
                markup += pre + lhs
#                if args:
#                    markup += args
    #            lhs,rhs = re.split(r'=', line, maxsplit=1)
    #            markup += lhs + '='
                line = args + rhs + '\n'
                iscode = True
        else:
            iscode = False

        if iscode or line[:1] == ' ':
            s = shlex.shlex(line)
            try:
                tokens = list(s)
            except ValueError:
                pass
            else:
                # collaps .*. operators
                j = 0
                while j < len(tokens)-2:
                    if tokens[j] == '.' and tokens[j+2] == '.':
                        if all( s in string.letters for s in tokens[j+1] ):
                            tokens[j:j+3] = [ ''.join(tokens[j:j+3]) ]
                    j += 1

                j = 0
                for tok in tokens:
                    lentok = len(tok)
                    while line[j] in s.whitespace:
                        markup += line[j]
                        j += 1
                    # skip over tok
                    if line[j:j+lentok] != tok:
                        if (line[j] == '.' and tok[0] == '.' and tok[-1] == '.' and
                            line[j+1:].lstrip()[:lentok-2] == tok[1:-1] and
                            line[j+1:].lstrip()[lentok-2:].lstrip()[0] == '.'):
                            namestart = line[j+1:].lstrip()
                            nblank = len(line[j+1:]) - len(namestart)
                            nblank += len(namestart[lentok-2:]) - len(namestart[lentok-2:].lstrip())
                            tok = line[j:j+lentok+nblank]
                            print '{}: joined token {}'.format(fname, tok)
                        else:
                            raise ValueError('{}: expecting token {} found {}'.format(fname, tok, line[j:j+lentok]))
                    assert line[j:j+len(tok)] == tok
                    j += len(tok)
                    if srname:
                        tok = format(tok, write=False)
    #                if srname in globvars and tok.lower() in globvars[srname]:
    #                    tok = globformat.format(tok)
                    markup += tok

                line = line[j:]

        markup += line

        i += 1

    return ''.join(markup)

def mkhtmllink(name, fnameline):
    fname,line = fnameline
    link = re.sub(r'^.*/', '', fname)
    link = re.sub(r'(\.F)?$', '.html', link)
    return '<a href="{}#{}"><font color="#000000">{}</font></a>'.format(link, line, name)

def markuphtml(outname, fname, srfiles, subargs, globvars, globmods, intents, link='', fmt='{}'):
    f = open(outname, 'w')
    f.write('''<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
<title>''' + fname + '''</title>
</head>
<body>
<pre>
''')
    if link:
        globformat = '<a href="{}{{0}}"><font color="#000000"><b>{{0}}</b></font></a>'.format(link)
        gmodformat = '<a href="{}{{0}}"><font color="#ff0000"><b>{{0}}</b></font></a>'.format(link)
        argformat = '<a href="{}{{0}}"><font color="#0000ff"><b>{{0}}</b></font></a>'.format(link)
        argmodformat = '<a href="{}{{0}}"><font color="#ff00ff"><b>{{0}}</b></font></a>'.format(link)
    else:
        globformat = '<b>{}</b>'.format(fmt)
        gmodformat = '<font color="#ff0000"><b>{}</b></font>'.format(fmt)
        argformat = '<font color="#0000ff"><b>{}</b></font>'.format(fmt)
        argmodformat = '<font color="#ff00ff"><b>{}</b></font>'.format(fmt)
    markup = markupfile(fname, srfiles, subargs, globvars, globmods, intents,
        globformat, gmodformat, argformat, argmodformat, escape=escapeangles,
        mklink=mkhtmllink)
    f.write(markup)
    f.write('''</pre>
</body>
</html>
''')


if __name__ == '__main__':
    if '-l' in sys.argv[1:]:
        subargs = load('subargs.json')
        direct = load('direct.json')
        calls = load('calls.json')
        srfiles = load('srfiles.json')
        intents = load('intents.json')
        commons = load('commons.json')
        globvars = load('globvars.json')
        globmods = load('globmods.json')
    else:
        incpaths = pathdict(rootdir='', ext='.h')
        forpaths = pathdict(rootdir='', ext='.F')
        globvars,commons = getallglobals(forpaths.values(), incpaths)
        subargs,direct,calls,srfiles = getdirects(forpaths.values(), incpaths)
        intents = findallintents(subargs, direct, calls)
        globmods = findallglobmods(subargs, direct, calls, intents, globvars)

        dump('subargs.json', subargs)
        dump('direct.json', direct)
        dump('calls.json', calls)
        dump('srfiles.json', srfiles)
        dump('intents.json', intents)
        dump('commons.json', commons)
        dump('globvars.json', globvars)
        dump('globmods.json', globmods)

