# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Pair Correlation Functions
############################
"""
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import FloatProgress
from exa.util.units import Length
from exatomic.core.universe import Universe


def radial_pair_correlation(universe, a, b, dr=0.05, start=1.0, stop=13.0,
                            length="Angstrom", window=1):
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
        a (str, list, array): First atom type (see Note)
        b (str, list, array): Second atom type (see Note)
        dr (float): Radial step size
        start (float): Starting radial point
        stop (float): Stopping radial point
        length (str): Output unit of length
        window (int): Smoothen data (useful when only a single a or b exist, default no smoothing)

    Returns:
        pcf (:class:`~pandas.DataFrame`): Pair correlation distribution and count

    Note:
        If a, b are strings pairs are determined using atomic symbols. If integers
        or lists/tuples are passed pairs are determined by atomic labels (see
        :func:`~exatomic.core.atom.Atom.get_atom_labels`). Arrays are assumed to
        be index values directly.

    Tip:
        Depending on the type of two body computation (or data) used, the volume
        may not be the cell volume; the normalization factor (the prefactor) is
        the volume sampled during computation of two body properties divided by
        the number of properties used in the histogram (the triple summation
        above, divided by the normalization for the radial distance outward).

    Warning:
        Using a start and stop length different from 0 and simple cubic cell dimension
        will cause the y axis magnitudes to be inaccurate. This can be remedied by
        rescaling values appropriately.
    """
    bins = np.arange(start, stop, dr)                     # Discrete values of r for histogram
    if isinstance(a, str):
        a_idx = universe.atom[universe.atom['symbol'] == a].index.values
    elif isinstance(a, (int, list, tuple, np.int64, np.int32)):
        a = [a] if not isinstance(a, (list, tuple)) else a
        a_idx = universe.atom[universe.atom['label'].isin(a)].index.values
    else:
        a_idx = a
    if isinstance(b, str):
        b_idx = universe.atom[universe.atom['symbol'] == b].index.values
    elif isinstance(a, (int, list, tuple, np.int64, np.int32)):
        b = [b] if not isinstance(b, (list, tuple)) else b
        b_idx = universe.atom[universe.atom['label'].isin(b)].index.values
    else:
        b_idx = b
    if "distance" in universe.atom_two.columns:
        c = "distance"
    else:
        c = "dr"
    distances = universe.atom_two.loc[(universe.atom_two['atom0'].isin(a_idx) &
                                       universe.atom_two['atom1'].isin(b_idx)) |
                                      (universe.atom_two['atom0'].isin(b_idx) &
                                       universe.atom_two['atom1'].isin(a_idx)), c]
    hist, bins = np.histogram(distances, bins)            # Compute histogram
    nn = hist.sum()                                       # Number of observations
    bmax = bins.max()                                     # Note that bins is unchanged by np.hist..
    rx, ry, rz = universe.frame[["rx", "ry", "rz"]].mean().values
    ratio = (((bmax/rx + bmax/ry + bmax/rz)/3)**3).mean() # Variable actual vol and bin vol
    v_shell = bins[1:]**3 - bins[:-1]**3                  # Volume of each bin shell
    if 'cell_volume' in universe.frame.columns:
        v_cell = universe.frame["cell_volume"].mean()         # Actual volume
    elif 'Volume' in universe.frame.columns:
        v_cell = universe.frame["Volume"].mean()         # Actual volume
        c = 'Volume'
    elif 'volume' in universe.frame.columns:
        v_cell = universe.frame["volume"].mean()         # Actual volume
    else:
        v_cell = universe.frame["rx"].max()**3
    g = hist*v_cell*ratio/(v_shell*nn)                    # Compute pair correlation
    numa = len(a_idx)/len(universe)
    numb = len(b_idx)/len(universe)
    n = hist.cumsum()/nn*numa*numb*4/3*np.pi*bmax**3/v_cell
    r = (bins[1:] + bins[:-1])/2*Length["au", length]
    unit = "au"
    if length in ["A", "angstrom", "ang", "Angstrom"]:
        unit = r"\AA"
    rlabel = r"$r\ \mathrm{(" + unit + ")}$"
    glabel = r"$g(r)$"
    nlabel = r"$n(r)$"
    df = pd.DataFrame.from_dict({rlabel: r, glabel: g, nlabel: n})
    if window > 1:
        df = df.rolling(window=window).mean()
        df = df.iloc[window:]
    df.set_index(rlabel, inplace=True)
    return df


def radial_pcf_out_of_core(hdftwo, hdfout, u, pairs, **kwargs):
    """
    Out of core radial pair correlation calculation.

    Atomic two body data is expected to have been computed (see
    :func:`~exatomic.core.two.compute_atom_two_out_of_core`)
    An example is given below. Note the importance of the definition
    of pairs and the presence of additional arguments.

    .. code:: Python

        radial_pcf_out_of_core("in.hdf", "out.hdf", uni, {"O_H": ([0], "H")},
                               length="Angstrom", dr=0.01)

    Args:
        hdftwo (str): HDF filepath containing atomic two body data
        hdfout (str): HDF filepath to which radial PCF data will be written (see Note)
        u (:class:`~exatomic.core.universe.Universe`): Universe
        pairs (dict): Dictionary of string name keys, values of ``a``, ``b`` arguments (see Note)
        kwargs: Additional keyword arguments to be passed (see Note)

    Note:
        Results will be stored in the hdfout HDF file. Keys are of the form
        ``radial_pcf_key``. The keys of ``pairs`` are used to store the output
        while the values are used to perform the pair correlation itself.
    """
    f = u.atom['frame'].unique()
    n = len(f)
    fp = FloatProgress(description="Computing:")
    display(fp)
    fdx = f[0]
    twokey = "frame_" + str(fdx) + "/atom_two"
    atom = u.atom[u.atom['frame'] == fdx].copy()
    uu = Universe(atom=atom, frame=u.frame.loc[[fdx]],
    atom_two = pd.read_hdf(hdftwo, twokey))
    pcfs = {}
    for key, ab in pairs.items():
        pcfs[key] = radial_pair_correlation(uu, ab[0], ab[1], **kwargs).reset_index()
    fp.value = 1/n*100
    for i, fdx in enumerate(f[1:]):
        twokey = "frame_" + str(fdx) + "/atom_two"
        atom = u.atom[u.atom['frame'] == fdx].copy()
        uu = Universe(atom=atom, frame=u.frame.loc[[fdx]],
        atom_two = pd.read_hdf(hdftwo, twokey))
        for key, ab in pairs.items():
            pcfs[key] += radial_pair_correlation(uu, ab[0], ab[1], **kwargs).reset_index()
        fp.value = (i+1)/n*100
    store = pd.HDFStore(hdfout)
    for key in pairs.keys():
        pcfs[key] /= n
        store.put("radial_pcf_"+key, pcfs[key])
    store.close()
    fp.close()
