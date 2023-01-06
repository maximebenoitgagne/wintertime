import re

def re_parts(pat):
    """Translate a shell PATTERN to a regular expression.
    
    With a group for each wildcard.
    Also returns a list of the parts separated by the wildcards.
    
    The groups returned by a match and the parts (or others) can
    be combined to the full filename using interleave.

    There is no way to quote meta-characters.
    """

    i, n = 0, len(pat)
    res = ''
    i0 = 0
    parts = []
    while i < n:
        i1 = i
        c = pat[i]
        i = i+1
        if c == '*':
            res = res + '(.*)'
            parts.append(pat[i0:i1])
            i0 = i
        elif c == '?':
            res = res + '(.)'
            parts.append(pat[i0:i1])
            i0 = i
        elif c == '[':
            j = i
            if j < n and pat[j] == '!':
                j = j+1
            if j < n and pat[j] == ']':
                j = j+1
            while j < n and pat[j] != ']':
                j = j+1
            if j >= n:
                res = res + '\\['
            else:
                stuff = pat[i:j].replace('\\','\\\\')
                i = j+1
                if stuff[0] == '!':
                    stuff = '^' + stuff[1:]
                elif stuff[0] == '^':
                    stuff = '\\' + stuff
                res = '%s([%s])' % (res, stuff)
                parts.append(pat[i0:i1])
                i0 = i
        else:
            res = res + re.escape(c)

    parts.append(pat[i0:])

    return re.compile(res+'$'), parts


def interleave(groups, parts):
    res = ''
    for g,p in zip(groups,parts):
        res = res + p + g

    return res + parts[-1]


def format2re(pat):
    """Translate a python format string with shell wildcards to a regular expression.

    Each replacement field is replaced by a group.
    
    Also returns a list of the parts separated by the fields.
    
    The groups returned by a match and the parts (or others) can
    be combined to the full filename using interleave.

    There is no way to quote meta-characters.
    """

    i, n = 0, len(pat)
    res = ''
    i0 = 0
    parts = []
    keys = []
    while i < n:
        i1 = i
        c = pat[i]
        i = i+1
        if c == '{':
            j = i
            jkey = 0
            while j < n and pat[j] != '}':
                if pat[j] == ':':
                    jkey = j

                j += 1

            if jkey == 0:
                jkey = j

            if j >= n:
                res = res + '{'
            else:
                keys.append(pat[i:jkey])
                i = j + 1
                res = res + '(.*)'
                parts.append(pat[i0:i1])
                i0 = i
        elif c == '*':
            res = res + '.*'
        elif c == '?':
            res = res + '.'
        elif c == '[':
            j = i
            if j < n and pat[j] == '!':
                j = j+1
            if j < n and pat[j] == ']':
                j = j+1
            while j < n and pat[j] != ']':
                j = j+1
            if j >= n:
                res = res + '\\['
            else:
                stuff = pat[i:j].replace('\\','\\\\')
                i = j+1
                if stuff[0] == '!':
                    stuff = '^' + stuff[1:]
                elif stuff[0] == '^':
                    stuff = '\\' + stuff
                res = '%s[%s]' % (res, stuff)
        else:
            res = res + re.escape(c)

    parts.append(pat[i0:])

    print res+'$'
    return res+'$', parts, keys


