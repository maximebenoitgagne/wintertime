import numpy as np

class IsopycnalFactory(object):
    def __init__(self, seamask, r, keepbottom=False):
        self.sm = seamask
        self.dep = np.sum(seamask, 0, np.int32).ravel()
        self.ioce = np.flatnonzero(self.dep)
        self.nk = len(seamask)
        self.r = r
        self.keepbottom = keepbottom

    def __call__(self, sigma, sigma0):
        sigma = np.where(self.sm, sigma, np.inf).reshape(self.nk, -1)
        for k in range(1, self.nk):
            sigma[k] = np.maximum(sigma[k-1], sigma[k])

        ki = (sigma <= sigma0).sum(0)
        k0 = np.clip(ki[self.ioce] - 1, 0, self.nk - 2)
        sigmaup = sigma[k0, self.ioce]
        sigmadn = sigma[k0 + 1, self.ioce]
        w1 = np.zeros(sigmaup.shape)
        wh = np.where(abs(sigmadn) < np.inf)
        with np.errstate(invalid='raise'):
            try:
                w1[wh] = ((sigma0 - sigmaup[wh])/(sigmadn[wh] - sigmaup[wh])).clip(0., 1.)
            except:
                whh = np.where(~np.isfinite(sigmaup[wh])|~np.isfinite(sigmadn[wh]))
                for _ in wh:
                    print(np.unravel_index(_[whh], self.sm.shape[1:]))
                print(sigmaup[wh][whh])
                print(sigmadn[wh][whh])
                raise
        k = np.zeros(self.dep.size) + np.nan
        k[self.ioce] = np.minimum(k0 + w1, self.dep[self.ioce] - 1.) 
        k[sigma[0] > sigma0] = -np.inf
        if not self.keepbottom:
            k[ki == self.dep] = np.inf
        k[0 == self.dep] = np.nan
        k.shape = self.sm.shape[1:]
        return Isopycnal(k, self.r, ~self.sm[0])


class Isopycnal(object):
    '''
    interpolate 3d array to given fractional k level
    '''
    def __init__(self, kf, r, lm):
        '''
        r is only given for reference
        '''
        self.r = r
        self.kf = kf
        self.lm = lm
        self.nk = nk = len(r)

        k, w = divmod(kf.clip(0, nk - 1), 1.)
        k = k.clip(0, nk - 2).astype(int)
        w = (kf - k).clip(0., 1.)
        k[lm] = 0

        if np.any(k < 0):
            print('Isopycnal: k<0: {}'.format(np.where(k<0)))
            raise

        self.k = k
        self.w = w
        self.i = np.arange(k.size).reshape(k.shape)
        self.r = r[k]*(1-w) + r[k+1]*w
        self.land = np.isnan(kf)
        self.bott = kf == np.inf
        self.outc = kf == -np.inf

    def __call__(self, a):
        a = a.reshape(self.nk, -1)
        ak = (a[self.k, self.i]*(1 - self.w) + a[self.k + 1, self.i]*self.w)
        return ak

    def save(self, fname):
        self.kf.astype('f8').tofile(fname)

