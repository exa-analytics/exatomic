# -*- coding: utf-8 -*-
'''
Pair Correlation Functions
============================
'''
from exa import _np as np
from exa.frames import DataFrame
from atomic import Length
from atomic.two import dmax, dmin


class PCF(DataFrame):
    '''
    '''
    def plot(self, count=True, secondary_y=True, **kwargs):
        '''
        Note:
            Because we set_index before plotting, the "actual" plotting function
            called is that of :class:`~exa.frames.DataFrame`.
        '''
        if secondary_y:
            kwargs['secondary_y'] = self.columns[2]
        ax = self.set_index(self.columns[0]).plot(**kwargs)
        patches1, labels1 = ax.get_legend_handles_labels()
        patches2, labels2 = ax.right_ax.get_legend_handles_labels()
        legend = ax.legend(patches1+patches2, labels1+labels2, loc='best', frameon=True)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        return ax


def compute_radial_pair_correlation(universe, a, b, dr=0.1, rmax=dmax, rmin=dmin,
                                    start=None, stop=None, unit='A'):
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

    Args:
        universe (:class:`~atomic.Universe`): The universe (with two body data)
        a (str):

    Note:
        Depending on the type of two body computation (or data) used, the volume
        may not be the cell volume; the normalization factor (the prefactor) is
        the volume sampled during computation of two body properties divided by
        the number of properties used in the histogram (the triple summation
        above, divided by the normalization for the radial distance outward).
    '''
    distances = universe.two.ix[(universe.two['symbols'].isin([a + b, b + a])), 'distance']
    dist_min = distances.min()
    dist_max = distances.max()
    start = start if start else dist_min - 0.1
    stop = stop if stop else dist_max - 0.1
    bins = np.arange(start, stop, dr)
    bins = np.append(bins, bins[-1] + dr)
    hist, bins = np.histogram(distances, bins)
    n = len(distances)
    m = len(universe)
    nats = (universe.atom['symbol'].astype('O').value_counts() // m).astype(np.int64)
    na = nats[a]
    nb = nats[b]
    rho = n / (4 / 3 * np.pi * (rmax - rmin - 0.4)**3)
    nn = na * nb // 2
    if a == b:
        nn = na * (na - 1) // 2
    if na == 1 or nb == 1:
        nn *= 2
    r3 = bins[1:]**3 - bins[:-1]**3
    g = hist / (4 / 3 * np.pi * r3 * rho)
    r = (bins[1:] + bins[:-1]) / 2
    r *= Length['au', unit]
    c = hist.cumsum().astype(np.int64) / n
    v_cell = universe.frame['cell_volume'].values[0]
    v_two = 4 / 3 * np.pi * (rmax + rmin + 0.2)**3
    c *= v_two / v_cell * nn
    n1 = 'Pair Correlation Function ({0}, {1})'.format(a, b)
    n2 = 'Distance ({0})'.format(unit)
    n3 = 'Pair Count ({0}, {1})'.format(a, b)
    return PCF.from_dict({n1: g, n2: r, n3: c})
