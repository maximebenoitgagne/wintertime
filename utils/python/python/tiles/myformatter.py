import re
import fnmatch
import string
fmt = string.Formatter()

def fstring(t,n,f,c):
    s = t
    if n is not None:
        s = s + '{' + n
        if c is not None:
            s = s + '!' + c
        if f != '':
            s = s + ':' + f
        s = s + '}'
    return s

def template_replace(tmpl,d):
    s = ''
    for t,n,f,c in fmt.parse(tmpl):
        if n in d:
#            s = s + t + d[n]
            s = s + fstring(t,n,f,c).format(**d)
        else:
            s = s + fstring(t,n,f,c)
    return s

def template_replace_all(tmpl,v):
    s = ''
    for t,n,f,c in fmt.parse(tmpl):
        s = s + t
        if n is not None:
            s = s + v
    return s

def template_fields(tmpl):
    return [ n for t,n,f,c in fmt.parse(tmpl) if n is not None ]

def format2re(tmpl):
    s = ''
    keys = []
    parts = []
    for t,n,f,c in fmt.parse(tmpl):
        tt = fnmatch.translate(t)
        # strip end-of-line/string sequences
        tt = re.sub(r'[\\Z\$].*$','',tt)
        s = s + tt
        if n is not None:
            s = s + '(.*)'
            keys.append(n)
        parts.append(t)
    return s+r'\Z(?ms)',parts,keys


