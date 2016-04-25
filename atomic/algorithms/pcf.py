# -*- coding: utf-8 -*-
'''
Pair Correlation Functions
============================
'''
import numpy as np
import pandas as pd
from atomic import Length
from atomic.two import dmax, dmin


class PCF(pd.DataFrame):
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


def radial_pair_correlation(universe, a, b, dr=0.05, vr=dmax,
                                    start=None, stop=None, length_unit='A'):
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
        a (str): First atom type
        b (str): Second atom type
        dr (float): Radial step size
        vr (float): (Max) radius used during computation of two body properties
        start (float): Starting radial point
        stop (float): Stopping radial point
        length_unit (str): Output unit of length

    Note:
        Depending on the type of two body computation (or data) used, the volume
        may not be the cell volume; the normalization factor (the prefactor) is
        the volume sampled during computation of two body properties divided by
        the number of properties used in the histogram (the triple summation
        above, divided by the normalization for the radial distance outward).
    '''
    distances = universe.two.ix[(universe.two['symbols'].isin([a + b, b + a])), 'distance']    # Collect distances of interest
    start = start if start else distances.min() - 0.1
    stop = stop if stop else distances.max() - 0.1
    bins = np.arange(start, stop, dr)           # The summation over Q_m (see docstring)
    bins = np.append(bins, bins[-1] + dr)       # is simply a histogram of the distances
    hist, bins = np.histogram(distances, bins)
    n = len(distances)
    m = len(universe)
    nats = (universe.atom['symbol'].astype('O').value_counts() // m).astype(np.int64)
    na = nats[a]
    nb = nats[b]
    v_two = 4 / 3 * np.pi * vr**3    # The volume used in computation of two body
    rho = n / v_two                  # properties is part of the normalization factor
    nn = na * nb // 2                # but not part of the pair count factor
    if a == b:
        nn = na * (na - 1) // 2
    elif na == 1 or nb == 1:
        nn *= 2
    r3 = bins[1:]**3 - bins[:-1]**3        # No need for approximations for the
    g = hist / (4 / 3 * np.pi * r3 * rho)  # volume of a spherical shell
    r = (bins[1:] + bins[:-1]) / 2
    r *= Length['au', length_unit]
    c = hist.cumsum().astype(np.int64) / n
    v_cell = universe.frame['cell_volume'].values[0]
    c *= v_two / v_cell * nn
    n1 = 'Pair Correlation Function ({0}, {1})'.format(a, b)
    n2 = 'Distance ({0})'.format(length_unit)
    n3 = 'Pair Count ({0}, {1})'.format(a, b)
    return PCF.from_dict({n1: g, n2: r, n3: c})
