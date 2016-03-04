# -*- coding: utf-8 -*-
'''
Pair Correlation Functions
============================
'''
from exa import _np as np
from exa.frames import DataFrame
from atomic import Length
from atomic.two import dmax


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


def compute_radial_pair_correlation(universe, a, b, dr=0.1, rr=dmax, start=None, stop=None, unit='A'):
    '''
    Compute the angularly independent pair correlation function.

    This function is sometimes called the radial distribution function.

    .. math::

        g_{AB}\left(r\\right) = \\frac{V}{4\pi r^{2}\Delta r MN_{A}N_{B}}
        \sum_{m=1}^{M}\sum_{a=1}^{N_{A}}\sum_{b=1}^{N_{B}}Q_{m}
        \left(r_{a}, r_{b}; r, \Delta r\\right)

        Q_{m}\\left(r_{a}, r_{b}; r, \\Delta r\\right) = \\begin{cases} \\
            &1\\ \\ if\\ r - \\frac{\Delta r}{2} \le \left|r_{a} - r_{b}\\right|\lt r + \\frac{\Delta r}{2} \\\\
            &0\\ \\ otherwise \\\\
        \\end{cases}

    Note:
        Depending on the type of two body computation (or data) used, the volume
        may not be the cell volume; the normalization factor (the prefactor) is
        the volume sampled during computation of two body properties divided by
        the number of properties used in the histogram (the triple summation
        above, divided by the normalization for the radial distance outward).
    '''
    distances = universe.two.ix[(universe.two['symbols'].isin([a + b, b + a])), 'distance']
    dmin = distances.min()
    dmax = distances.max()
    start = dmin - 0.4 if start is None else start
    stop = dmax - 0.2 if stop is None else stop
    bins = np.arange(start, stop, dr)
    bins = np.append(bins, bins[-1] + dr)
    hist, bins = np.histogram(distances, bins)
    n = len(distances)
    m = len(universe)
    f = universe._frame.index[0]
    nats = (universe.atom['symbol'].astype('O').value_counts() // m).astype(np.int64)
    na = nats[a]
    nb = nats[b]
    rho = n / (4 / 3 * np.pi * (rr - 0.3)**3)
    nn = max((na, nb)) // 2
    nn = na * nb // 2
    if na == 1 or nb == 1:
        nn = na * nb
    elif a == b:
        nn = na * (na - 1) // 2
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
