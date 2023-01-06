import re
import numpy as np

class AttArray(np.ndarray):
    def __new__(subtype, filename, **kwargs):
        subarr = np.loadtxt(filename, **kwargs)
        subarr = subarr.view(subtype)

        with open(filename) as file:
            for line in file:
                if re.match('#[^ \t]*:', line):
                    continue
                if re.match('#',line):
                    cols = line[1:].split()
                    for i in range(len(cols)):
                        ii = None

                        # make sure we don't overwrite one of ndarray's attributes
                        if hasattr(subarr,cols[i]):
                            ii = 0

                        # and make them unique
                        if cols[i] in cols[:i] and not cols[i] == '+-':
                            ii = 1

                        if ii is not None:
                            key = cols[i] + '_%d'%ii
                            while key in cols[:i] or hasattr(subarr, key):
                                ii += 1
                                key = cols[i] + '_%d'%ii

                            cols[i] = key
                            
                    for i in range(len(cols)-1):
                        if cols[i+1] == '+-':
                            cols[i+1] = cols[i] + '_ERR'

                    subarr.columns = cols
                    for i,v in enumerate(cols):
                        setattr(subarr, v, subarr[:,i].view(np.ndarray))

                    break

        return subarr


        def __array_finalize__(self,obj):
            self.columns = getattr(obj, 'columns', {})
            for c in self.columns:
                setattr(self, c, getattr(obj, c, {}))


        def __repr__(self):
            desc="""\
 array(data=
   %(data)s,
       columns=%(tag)s)"""
            return desc % {'data': str(self), 'tag':self.columns }


            

