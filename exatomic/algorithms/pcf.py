# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Pair Correlation Functions
############################
"""
import numpy as np
import pandas as pd
from exa.util.units import Length


def radial_pair_correlation(universe, a, b, dr=0.05, start=1.0, stop=13.0,
                            length="A", window=1):
    """
    Compute the angularly independent pair correlation function.

    This function is sometimes called the pair radial distribution function. The
    quality of the result depends strongly on the amount of two body distances
    computed (see :func:`~exatomic.atom_two.compute_two_body`) in the case of a
    periodic unvierse. Furthermore, the result can be skewed if only a single
    atom a (or b) exists in each frame. In these situations one can use the
    **window** and **dr** parameter to adjust the result accordingly. Reasonable
    values for **dr** range from 0.1 to 0.01 and reasonable values for **window**
    range from 1 to 5 (default is 1 - no smoothing).

    .. code-block:: Python

        pcf = radial_pair_correlation(universe, "O", "O")
        pcf.plot(secondary_y="Pair Count")

    .. math::

        g_{AB}\left(r\\right) = \\frac{V}{4\pi r^{2}\Delta r MN_{A}N_{B}}
        \sum_{m=1}^{M}\sum_{a=1}^{N_{A}}\sum_{b=1}^{N_{B}}Q_{m}
        \left(r_{a}, r_{b}; r, \Delta r\\right)

        Q_{m}\\left(r_{a}, r_{b}; r, \\Delta r\\right) = \\begin{cases} \\
            &1\\ \\ if\\ r - \\frac{\Delta r}{2} \le \left|r_{a} - r_{b}\\right|\lt r + \\frac{\Delta r}{2} \\\\
            &0\\ \\ otherwise \\\\
        \\end{cases}

    Args:
        universe (:class:`~exatomic.Universe`): The universe (with two body data)
        a (str): First atom type
        b (str): Second atom type
        dr (float): Radial step size
        start (float): Starting radial point
        stop (float): Stopping radial point
        length (str): Output unit of length
        window (int): Smoothen data (useful when only a single a or b exist, default no smoothing)

    Returns:
        pcf (:class:`~pandas.DataFrame`): Pair correlation distribution and count

    Note:
        Depending on the type of two body computation (or data) used, the volume
        may not be the cell volume; the normalization factor (the prefactor) is
        the volume sampled during computation of two body properties divided by
        the number of properties used in the histogram (the triple summation
        above, divided by the normalization for the radial distance outward).
    """
    bins = np.arange(start, stop, dr)                     # Discrete values of r for histogram
    symbol = universe.atom["symbol"].astype(str)          # To select distances, map to symbols
    symbol0 = universe.atom_two["atom0"].map(symbol)
    symbol1 = universe.atom_two["atom1"].map(symbol)
    symbols = symbol0 + symbol1
    indexes = symbols[symbols.isin([a + b, b + a])].index # Distances of interest or those that
    distances = universe.atom_two.ix[indexes, "distance"] # match symbol pairs
    hist, bins = np.histogram(distances, bins)            # Compute histogram
    nn = hist.sum()                                       # Number of observations
    bmax = bins.max()                                     # Note that bins is unchanged by np.hist..
    rx, ry, rz = universe.frame[["rx", "ry", "rz"]].mean().values
    ratio = (((bmax/rx + bmax/ry + bmax/rz)/3)**3).mean() # Variable actual vol and bin vol
    v_shell = bins[1:]**3 - bins[:-1]**3                  # Volume of each bin shell
    v_cell = universe.frame["cell_volume"].mean()         # Actual volume
    g = hist*v_cell*ratio/(v_shell*nn)                    # Compute pair correlation
    na = universe.atom[universe.atom["symbol"] == a].groupby("frame").size().mean()
    nb = universe.atom[universe.atom["symbol"] == b].groupby("frame").size().mean()
    if a == b:
        nb -= 1
    n = hist.cumsum()/nn*na*nb*4/3*np.pi*bmax**3/v_cell
    r = (bins[1:] + bins[:-1])/2*Length["au", length]
    unit = "au"
    if length in ["A", "angstrom", "ang"]:
        unit = r"\AA"
    rlabel = r"$r\ \mathrm{(" + unit + ")}$"
    glabel = r"$g_\mathrm{" + a + b + r"}(r)$"
    nlabel = r"$n_\mathrm{" + a + b + r"}(r)$"
    df = pd.DataFrame.from_dict({rlabel: r, glabel: g, nlabel: n})
    if window > 1:
        df = df.rolling(window=window).mean()
        df = df.iloc[window:]
    df.set_index(rlabel, inplace=True)
    return df
