import re
import string

_ivals = {'y':360,
          'm':30,
          'd':1
         }

def parsetime(s):
    t = 0
    for v,u in re.findall(r'([0-9]+)([a-zA-Z])', s):
        t += int(v)*_ivals[u]
    return t

def poptimes(l):
    i = 0
    res = []
    while i<len(l):
        if l[i][0] in string.digits and l[i][-1] in string.letters:
            res.append(parsetime(l[i]))
            l.pop(i)
        else:
            i += 1

    return res

