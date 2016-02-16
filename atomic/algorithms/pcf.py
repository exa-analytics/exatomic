# -*- coding: utf-8 -*-
'''
Pair Correlation Functions
============================
'''
from exa import _np as np
from exa.frames import DataFrame
from atomic import Length


class PCF(DataFrame):
    '''
    '''
    def plot(self, count=True, **kwargs):
        '''
        Note:
            Because we set_index before plotting, the "actual" plotting function
            called is that of :class:`~exa.frames.DataFrame`.
        '''
        if 'secondary_y' not in kwargs and count:
            kwargs['secondary_y'] = self.columns[2]
        return self.set_index(self.columns[0]).plot(**kwargs)


def compute_radial_pair_correlation(universe, a, b, dr=0.05, start=None,
                                    stop=None, unit='A'):
    '''
    '''
    start = universe.two['distance'].min() - 0.1 if start is None else start
    stop = universe.two['distance'].max() - 0.1 if stop is None else stop
    distances = universe.two.ix[(universe.two['symbols'].isin([a + b, b + a])), 'distance']
    bins = np.arange(start, stop, dr)
    bins = np.append(bins, bins[-1] + dr)
    hist, bins = np.histogram(distances, bins)
    n = len(distances)
    m = len(universe)
    f = universe._frame.index[0]
    vol = universe.frame.ix[f, 'cell_volume']
    nats = (universe.atom['symbol'].value_counts() // m).astype(np.int64)
    na = nats[a]
    nb = nats[b]
    rho = 2 * n / vol
    if a == b:
        rho /= 2
    nn = max((na, nb))
    r3 = bins[1:]**3 - bins[:-1]**3
    g = hist / (4 / 3 * np.pi * r3 * rho)
    r = (bins[1:] + bins[:-1]) / 2
    r *= Length['au', unit]
    c = hist.cumsum().astype(np.int64) / n
    c *= nn
    n1 = 'Pair Correlation Function ({0}, {1})'.format(a, b)
    n2 = 'Distance ({0})'.format(unit)
    n3 = 'Pair Count ({0}, {1})'.format(a, b)
    return PCF.from_dict({n1: g, n2: r, n3: c})
