# -*- coding: utf-8 -*-
'''
Pair Correlation Functions
============================
'''
import numpy as np
import pandas as pd
from atomic import Length


def radial_pair_correlation(universe, a, b, dr=0.05, start=None, stop=None,
                            output_length_unit='au', window=None):
    '''
    Compute the angularly independent pair correlation function.

    This function is sometimes called the pair radial distribution function. The
    quality of the result depends strongly on the amount of two body distances
    computed (see :func:`~atomic.two.compute_two_body`) in the case of a
    periodic unvierse. Furthermore, the result can be skewed if only a single
    atom a (or b) exists in each frame. In these situations one can use the
    **window** and **dr** parameter to adjust the result accordingly. Reasonable
    values for **dr** range from 0.1 to 0.01 and reasonable values for **window**
    range from 1 to 5 (default is 1 - no smoothing).

    .. code-block:: Python

        pcf = radial_pair_correlation(universe, 'O', 'O')
        pcf.plot(secondary_y='Pair Count')

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
        start (float): Starting radial point
        stop (float): Stopping radial point
        output_length_unit (str): Output unit of length
        window (int): Smoothen data (useful when only a single a or b exist)

    Returns:
        pcf (:class:`~pandas.DataFrame`): Pair correlation distribution and count

    Note:
        Depending on the type of two body computation (or data) used, the volume
        may not be the cell volume; the normalization factor (the prefactor) is
        the volume sampled during computation of two body properties divided by
        the number of properties used in the histogram (the triple summation
        above, divided by the normalization for the radial distance outward).
    '''
    if universe.is_variable_cell:
        raise NotImplementedError('No support for variable cell PCFs')
    # Filter relevant distances from the two body table and get some data
    distances = universe.two.ix[universe.two['symbols'].isin([a+b, b+a]), 'distance']
    vc = universe.atom['symbol'].value_counts()
    vc.index = vc.index.astype(str)
    n = vc[a] / len(universe)
    m = vc[b] / len(universe)
    if a == b:
        m -= 1    # Can't be paired with self
        if window is None:
            window = 2
    if window is None:
        window = 1
    npairs = n * m // 2 # Total number of pairs per each entire frame
    # Select appropriate defaults
    start = distances.min() - 0.3
    stop = distances.max()
    bins = np.arange(start, stop, dr)
    # and compute the histogram (triple summation)
    hist, bins = np.histogram(distances, bins)
    r3 = bins[1:]**3 - bins[:-1]**3    # Exact spherical shell volume
    r = (bins[1:] + bins[:-1]) / 2
    nn = hist.sum()                    # Actual number of distances considered
    g = hist * stop**3 / (r3 * nn)
    count = hist.cumsum() * npairs / nn
    if a == b:
        count *= stop**3 / universe.frame['cell_volume'].max()
    elif n == 1 or m == 1:
        count *= 2 * stop / universe.frame['rx'].max()
    df = pd.DataFrame.from_dict({'g$_{\mathrm{' + a + b + '}}$($r$)': g, 'Pair Count': count})
    # Normalize for the fact that frames may have variable effective "stop" values
    df.iloc[:, 1] *= (df.iloc[-4:, 1]**-1).mean()
    # Smoothen if requested, necessary when n == 1 or m == 1, see note in docstring
    df.iloc[:, 1] = df.iloc[:, 1].rolling(window=window).mean()
    df.index = r
    df.dropna(inplace=True)
    if output_length_unit in Length.aliases:
        output_length_unit = Length.aliases[output_length_unit]
    df.index *= Length['au', output_length_unit]
    if output_length_unit == 'A':
        df.index.name = r'$r$ ($\mathrm{\AA}$)'
    else:
        df.index.name = r'$r$ ({})'.format(output_length_unit)
    return df
