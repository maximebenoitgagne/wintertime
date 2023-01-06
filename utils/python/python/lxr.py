#!/usr/bin/env python
"""Usage: headerdep.py [-db db] [-p patt] [-v] [-V] headerfile ...

-db db   database [lxrdarwin2]
-p patt  filename pattern [.h$]
-v       list variables used
-V       list variables declared

headerfile must include path (MITgcm/...)
"""
import sys
import os
import re
from mysql.connector import connect,errors

tables = {
    'lxr_files': {
        'filename': 'char(255)',
        'revision': 'char(255)',
        'fileid': 'int',  # auto
        'primary': 'key',
    },
    'lxr_symbols': {
        'symname': 'char(255)',
        'symid': 'int',  # auto
        'primary': 'key',
    },
    'lxr_indexes': {
        'symid': 'int',
        'fileid': 'int',
        'line': 'int',
        'langid': 'tinyint',
        'type': 'smallint',
        'relsym': 'int',
        'indexid': 'int',  # auto
    },
    'lxr_filedescriptions': {
        'fileid': 'int',
        'description': 'char(255)',
    },
    'lxr_descriptions': {
        'description': 'char(255)',
        'indexid': 'int',
        'line': 'int',
    },
    'lxr_paramfiles': {
        'symid': 'int',
        'fileid': 'int',
        'line': 'int',
        'paramfile': 'char(255)',
    },
    'lxr_releases': {
        'fileid': 'int',
        'therelease': 'char(255)',
    },
    'lxr_useage': {
        'fileid': 'int',
        'line': 'int',
        'symid': 'int',
        'funcid': 'int',
    },
    'lxr_calls': {
        'fileid': 'int',
        'line': 'int',
        'symid': 'int',
        'callerid': 'int',
    },
    'lxr_functions': {
        'fileid': 'int',
        'line': 'int',
        'symid': 'int',
    },
    'lxr_status': {
        'fileid': 'int',
        'status': 'tinyint',
    },
    'lxr_declarations': {
        'declid': 'smallint',  # auto
        'langid': 'tinyint',
        'declaration': 'char(255)',
    },
}


class DB(object):
    def __init__(self,db='lxrdarwin2',user='lxr',root='.',followlinks=False,release='head',pre=''):
        self.conn = connect(user=user,db=db)
        self.cursor = self.conn.cursor()
        root = re.sub(r'/$','',root)
        if root == '.':
            root = os.getcwd()
        self.root = root
        self.pre = pre
        self.followlinks = followlinks
        self.release = release
        self.idcommon, = self.queryone("select declid from lxr_declarations where langid=4 and declaration='common block'")
        self.idnamelist, = self.queryone("select declid from lxr_declarations where langid=4 and declaration='namelist'")
        self.idlocal, = self.queryone("select declid from lxr_declarations where langid=4 and declaration='local variable'")

    def close(self):
        """ close mysql cursor and connector """
        self.cursor.close()
        self.conn.close()

    def __iter__(self):
        return self.cursor.__iter__()

    def execute(self,operation,params=None):
        """ executes mysql operation (no results fetched) """
        self.cursor.execute(operation,params)

    def queryone(self,operation,params=None):
        """ executes mysql operation and returns result of fetchone """
        self.cursor.execute(operation,params)
        return self.cursor.fetchone()

    def query(self,operation,params=None):
        """ executes mysql operation and returns result of fetchall """
        self.cursor.execute(operation,params)
        return self.cursor.fetchall()

    def querycol(self,command,params=None,col=0):
        """ runs mysql command and returns column col of fetchall """
        rows = self.query(command,params)
        return [ r[col] for r in rows ]

    def clear(self):
        """ clear temporary table """
        try:
            self.cursor.execute("drop table tmp.syms")
        except errors.ProgrammingError:
            pass

    def vars(self,filename):
        """ returns all symbols declared in filename,
    except common blocks and namelists """
        try:
            self.cursor.execute("drop table tmp.syms")
        except errors.ProgrammingError:
            pass
        self.cursor.execute(' '.join(["create temporary table tmp.syms",
                  "select distinct s.symname",
                  "from lxr_files f, lxr_indexes i, lxr_symbols s, lxr_releases r",
                  "where i.fileid=f.fileid",
                    "and f.fileid=r.fileid",
                    "and i.symid=s.symid",
                    "and f.filename=%s",
                    "and r.therelease=%s",
                    "and i.type!=%s",
                    "and i.type!=%s",
                 ]), (filename,self.release,self.idcommon,self.idnamelist))
        self.cursor.execute("select * from tmp.syms")
        return [r[0] for r in self.cursor.fetchall()]

    def uses(self,filename):
        self.cursor.execute(' '.join(["select distinct s.symname",
                  "from lxr_files f, lxr_useage u, lxr_symbols s, lxr_releases r",
                  "where u.symid=s.symid",
                    "and f.fileid=u.fileid",
                    "and f.fileid=r.fileid",
                    "and f.filename=%s",
                    "and r.therelease=%s",
                  ]), (filename,self.release))
        res = [r[0] for r in self.cursor.fetchall()]
        return res

    def locals(self,filename):
        res = self.querycol(' '.join(["select distinct s.symname",
                  "from lxr_files f, lxr_indexes i, lxr_symbols s, lxr_releases r",
                  "where i.symid=s.symid",
                    "and f.fileid=i.fileid",
                    "and f.fileid=r.fileid",
                    "and f.filename=%s",
                    "and r.therelease=%s",
                    "and i.type=%s",
                  ]), (filename,self.release,self.idlocal))
        return res

    def globals(self,filename):
        uses = self.uses(filename)
        locs = self.locals(filename)
        return { k for k in uses if k not in locs }

    def tmpsymsuses(self,patt='.'):
        """ find all files matching patt that use variables in tmp.syms """
        self.cursor.execute(' '.join(["select distinct f.filename",
                  "from lxr_files f, lxr_useage u, lxr_symbols s, tmp.syms ss, lxr_releases r",
                  "where s.symname=ss.symname",
                  "and u.symid=s.symid",
                  "and f.fileid=u.fileid",
                  "and f.fileid=r.fileid",
                  "and f.filename rlike %s",
                  "and r.therelease=%s",
                  ]), (patt,self.release))
        files = [r[0] for r in self.cursor.fetchall()]
        return files

    def tmpsymsusesinfile(self,filename):
        """ find all variables from tmp.syms used in filename """
        uses = {}
        self.cursor.execute(' '.join(["select distinct s.symname",
              "from lxr_files f, lxr_useage u, lxr_symbols s, lxr_releases r, tmp.syms ss",
              "where s.symname=ss.symname",
              "and u.symid=s.symid",
              "and f.fileid=u.fileid",
              "and f.fileid=r.fileid",
              "and f.filename=%s",
              "and r.therelease=%s",
              ]), (filename,self.release))
        uses = [r[0] for r in self.cursor.fetchall()]
        return uses

    def headeruses(self,filename,patt='.'):
        """ find all files matching patt that use variables declared in filename.
    returns dictionary of files and variables used in each file """
        # this sets up tmp.syms
        vars = self.vars(filename)
        files = self.tmpsymsuses(patt)
        uses = { f:self.tmpsymsusesinfile(f) for f in files if f != filename and not re.search('/verification/',f)}
        self.clear()
        return uses

    def indirectheaderuses(self,filename):
        """ recursively find all files that use variables declared in filename
    or any header that depends on filename """
        headers = self.headeruses(filename,r'\.h$')
        uses = {filename:self.headeruses(filename)}
        for h in headers:
            uses[h] = self.headeruses(h)

        return uses

    def recheaderuses(self,filename):
        """ recursively find all files that use variables declared in filename
    or any header that depends on filename """
#        headers = self.headeruses(filename,r'\.h$')
        headers = self.requiring[filename]
        uses = self.headeruses(filename)
        for h in headers:
            more = self.headeruses(h)
            for k,v in more.items():
                uses[k] = sorted(list(set(uses.get(k,[]) + [h+':'+v[0]])))

        return uses

    def headers(self,exclude=['verification',r'^\.']):
        """ returns all header files (*.h) under root """
        res = []
        for path,dirs,files in os.walk(self.root,followlinks=self.followlinks):
            relpath = re.sub(self.root,'',path)
            res.extend(os.path.join(relpath,f) for f in files if f.endswith('.h') and not os.path.islink(path+'/'+f))
            for ex in exclude:
                for d in dirs:
                    if re.search(ex,d):
                        dirs.remove(d)

        return res

    def sources(self,exclude=['verification',r'^\.']):
        """ returns all source files (*.F) under root """
        res = []
        for path,dirs,files in os.walk(self.root,followlinks=self.followlinks):
            relpath = re.sub(self.root,'',path)
            res.extend(os.path.join(relpath,f) for f in files if f.endswith('.F') and not f.startswith('.') and not os.path.islink(path+'/'+f))
            for ex in exclude:
                for d in dirs:
                    if re.search(ex,d):
                        dirs.remove(d)

        return res

    def includes(self,path):
        """ returns a list of headers included in path (relative to root) """
        res = []
        try:
            with open(self.root + path) as f:
                for line in f:
                    m = re.match(r' *# *include +"([^"]*)"', line)
                    if m:
                        res.append(m.group(1))
        except IOError as err:
            sys.stderr.write(str(err) + '\n')

        return res

    def allincludes(self):
        """ returns dictionary of all includes """
        return {f:self.includes(f) for f in self.sources()}

    def headerdeps(self):
        """ returns dictionary required:by of headers """
        return {p:self.headeruses(p,r'\.h$').keys() for p in self.headers()}

    def init(self):
        self.incs = self.allincludes()
        self.requiring = { p:self.headeruses(p,r'\.h$') for p in self.headers() }
        self.deps = {}
        for dep,d in self.requiring.items():
            for h,v in d.items():
                self.deps[h] = self.deps.get(h,{})
                self.deps[h][dep] = v

    def unused(self,path):
        """ return files that include path but do not need it
    requires earlier call to init """
        _,header = os.path.split(path)
        uses = self.recheaderuses(path)
        return [f for f,hs in self.incs.items() if header in hs and f not in uses.keys()]

    def allunused(self):
        """ returns dictionary of unused includes for all headers """
        self.init()
        return { path:self.unused(path) for path in self.headers() }

    def missing(self,path):
        """ returns missing includes and required variables for header path """
        _,header = os.path.split(path)
        res = {}
        for f,vs in self.recheaderuses(path).items():
          if not f.endswith('.h'):
            if os.path.exists(self.root + f):
                if header not in self.includes(f):
                    res[f] = vs
            else:
                res[f] = ['NOT FOUND']

        return res

    def allmissing(self):
        """ returns dictionary of all missing includes """
        return { path:self.missing(path) for path in self.headers() }

    def allmissingshort(self):
        """ returns dictionary of all missing includes, listing only first variable required """
        miss = self.allmissing()
        short = {m:{k:vv[0] for k,vv in v.items() if len(vv) and not k.endswith('.h')} for m,v in miss.items()}
        return { k:v for k,v in short.items() if v }

    def findheader(self, name):
        res = []
        for path in self.headers():
            _,nm = os.path.split(path)
            if nm == name:
                res.append(path)

        return res

if __name__ == '__main__':
    args = sys.argv[1:]
    try:
        i = args.index('-db')
        args.pop(i)
        db = args.pop(i)
    except ValueError:
        db = 'lxrdarwin2'

    try:
        i = args.index('-p')
        args.pop(i)
        patt = args.pop(i)
    except ValueError:
        patt = '.h$'

    try:
        i = args.index('-V')
        args.pop(i)
        doprevars = True
    except ValueError:
        doprevars = False

    try:
        i = args.index('-v')
        args.pop(i)
        dovars = True
    except ValueError:
        dovars = False

    if len(args) == 0:
        sys.exit(__doc__)

    db = DB(db=db)

    for header in args:
        if not header.startswith('/'):
            header = '/' + header

        if doprevars:
            vars = db.vars(header)
            print '\n'.join(vars)

        uses = db.headeruses(header,patt)
        print header+':'
        for f in uses:
            if dovars:
                print '  ' + f + '  ' + ' '.join(uses[f])
            else:
                print '\t' + f

    db.close()

