# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Delocalization
################################
Miscellaneous functions for computing the curvature in the energy
of a system as a function of the electron number. These functions
require results from 3 different quantum chemical calculations on
an (N-1), N, and (N+1) electron system.
"""
from __future__ import print_function
import os
import numpy as np
import pandas as pd
import seaborn as sns
from exa.util.mpl import _gen_figure
from exa.util.units import Energy
from exatomic import gaussian, nwchem
#from exatomic.mpl import plot_j2_surface, plot_j2_contour

sns.mpl.pyplot.rcParams.update({'text.usetex': True,
                                'font.family': 'serif',
                                'font.serif': ['Times']})
#                                'ytick.labelsize': 24,
#                                'xtick.labelsize': 24})


def plot_energy(curv, color=None, title='', figsize=(21,5),
                nylabel=3, nxlabel=5, fontsize=24):
    """
    Accepts the output of compute_curvature or combine_curvature and
    returns a figure with appropriate styling.
    """
    def _deltaE(col):
        if col.name == 'n': return col
        cat = np.linspace(col.values[0], 0, 51)
        an = np.linspace(0, col.values[-1], 51)
        return col - np.hstack([cat, an])
    figargs = {'figsize': figsize}
    fig = _gen_figure(nxplot=1, nyplot=3, nxlabel=nxlabel,
                      figargs=figargs, fontsize=fontsize)
    ax, axnone, ax1 = fig.get_axes()
    axnone.set_visible(False)
    color = sns.color_palette('cubehelix', curv.shape[1] - 1) \
            if color is None else color
    plargs = {'x': 'n', 'color': color, 'title': title, 'legend': False}
    curvy = curv.apply(_deltaE)
    curv.plot(ax=ax, **plargs)
    ax.set_ylim([curv.min().min(), curv.max().max()])
    ax.set_ylabel('$\Delta$E (eV)', fontsize=fontsize)
    ax.set_xlabel('$\Delta$N', fontsize=fontsize)
    curvy.plot(ax=ax1, **plargs)
    del curvy['n']
    ax1.set_ylim([curvy.min().min(), curvy.max().max()])
    ax1.set_ylabel('$\Delta \Delta$E (eV)', fontsize=fontsize)
    ax.set_xlabel('$\Delta$N', fontsize=fontsize)
    loc = [1.2, (9 - curv.shape[1]) / 25]
    ax.legend(*ax.get_legend_handles_labels(), loc=loc)
    return fig


def combine_curvature(curvs, order=None):
    """
    Given a list of the results of compute_curvature, return a single
    dataframe containing all of the E(N) results.
    """
    if order is not None:
        oldorder = [curv.name for curv in curvs]
        reordered = []
        for ordr in order:
            for i, old in enumerate(oldorder):
                if ordr.upper() == old.upper():
                    reordered.append(curvs[i])
    else:
        reordered = curvs
    for i, curv in enumerate(reordered):
        if not i: continue
        try: reordered[i].drop('n', axis=1, inplace=True)
        except ValueError: pass
    return pd.concat(reordered, axis=1)


def compute_curvature(*args, **kwargs):
    """
    Computes the curvature of the energy of a system as a function
    of the number of electrons in the system E(N).

    Args:
        args (:class:`exatomic.core.universe.Universe`): unis in ascending electron order
        neut (int): index of args corresponding to the zero energy system
        extras (bool): if True, attach the raw data to df before returning

    Returns:
        df (pd.DataFrame): The energy as a function of N
    """
    neut = kwargs.pop('neut', None)
    tag = kwargs.pop('tag', '')
    extras = kwargs.pop('extras', True)
    if len(args) == 1:
        raise Exception("Must have at least 2 systems "
                        "differing in electron number.")
    nargs = len(args)
    neut = nargs // 2 if neut is None else neut
    # Len (nargs) arrays
    totens, aorbs, borbs = [], [], []
    # Len (nargs - 1) arrays
    lumos, homos, js, diffs, es, cs = [], [], [], [], [], []
    for job in args:
        # Get the total energy of the system
        totens.append(job.frame['E_tot'].values[-1])
        # Get the highest occupied molecular orbital of the system
        aorbs.append(job.orbital.get_orbital())
        # Check for open shell, if not use alpha orbital data
        try: borbs.append(job.orbital.get_orbital(spin=1))
        except IndexError: borbs.append(aorbs[-1])
    # Cycle over the orbitals to find the right energies
    for job, alo, ahi, blo, bhi in zip(args, aorbs, aorbs[1:], borbs, borbs[1:]):
        # Ionization from alpha orbitals
        if alo.vector < ahi.vector:
            lumos.append(job.orbital.get_orbital(index=alo.name + 1).energy)
            homos.append(ahi.energy)
        # Ionization from beta orbitals
        elif blo.vector < bhi.vector:
            lumos.append(job.orbital.get_orbital(index=blo.name + 1).energy)
            homos.append(bhi.energy)
        # Well it has to be one or the other
        else: raise Exception("Can't find electronic relationship.")

    # Compute J^2
    for homo, elo, ehi in zip(homos, totens, totens[1:]):
        diffs.append(ehi - elo)
        js.append(homo - diffs[-1])
    j2 = sum([j ** 2 for j in js])
    # Compute E(n) and curvature coefficients
    atv = Energy['Ha', 'eV']
    n = np.linspace(0, 1, 51)
    # Energy adjustments so everything is relative to neut
    aboves = [ sum(diffs[neut:i]) for i in range(len(homos))]
    belows = [-sum(diffs[i:neut]) for i in range(len(homos))]
    adjs = belows[:neut] + aboves[neut:]
    # Compute energies and curvature
    for homo, lumo, dif, adj in zip(homos, lumos, diffs, adjs):
        es.append((dif*n + ((lumo - dif)*(1 - n) +
                  (dif - homo)*n)*n*(1 - n) + adj) * atv)
        cs.append((np.sum((lumo - dif)*(6*n - 4) +
                  (dif - homo)*(2 - 6*n)))/(len(n)*2) * atv)

    # Collect all the data into a dataframe
    colname = '{} (' + ','.join(['{:.2f}' for coef in cs]) + ')'
    args = [tag] + cs
    colname = colname.format(*args)
    start = -neut
    stop = nargs - neut - 1
    ns = [np.linspace(lo, lo + 1, 51) for lo in range(start, stop)]
    data = pd.DataFrame({'n': np.concatenate(ns),
                         colname: np.concatenate(es)})
    data.name = tag
    data.j2 = j2
    if extras:
        data.curs = cs
        data.ens = es
    return data


def _dir_to_dict(adir, tuning=False):
    if not adir.endswith(os.sep): adir += os.sep
    files = {}
    for fl in os.listdir(adir):
        if os.path.isdir(adir + fl): continue
        if fl.startswith('.'): continue
        if tuning:
            try:
                comp, gam, alp, chgext = fl.split('-')
                func = '-'.join([gam, alp])
            except ValueError:
                print('{} is ignored.'.format(fl))
                continue
        else:
            try:
                comp, func, chgext = fl.split('-')
            except ValueError:
                print('{} is ignored.'.format(fl))
                continue
        files.setdefault(func, {})
        files[func][chgext.split('.')[0]] = adir + fl
    return files


def functional_results(adir, code='gaussian', ip=False,
                       ea=False, labels=None, timer=True):
    if ip and ea: raise Exception("Can't do just ip as well as just ea.")
    codemap = {'gaussian': gaussian.Output,
                 'nwchem': nwchem.Output}
    if ip: keys = ['cat', 'neut']
    elif ea: keys = ['neut', 'an']
    else: keys = ['cat', 'neut', 'an']
    curvs = []
    files = _dir_to_dict(adir)
    comp = files[list(files.keys())[0]][keys[0]].split(os.sep)[-1].split('-')[0]
    for func in files:
        if any((key not in files[func] for key in keys)): continue
        if timer: print('|', end='')
        outs = [codemap[code](files[func][chg]) for chg in keys]
        tag = labels[func] if labels is not None else ''
        try:
            curvs.append(compute_curvature(*outs, tag=tag))
        except Exception:
            print(comp, func, 'not computed')
    return curvs


def tuning_results(adir, code='gaussian', ip=False, ea=False,
                   deep=False, debug=0):
    """
    Given a directory containing output files with systematic file names,
    return a dataframe containing summary information about the calculations.

    Args:
        adir (str): path to the directory containing outputs
        ip (bool): if true only consider (N-1, N) systems
        ea (bool): if true only consider (N, N+1) systems
        deep (bool): if true must have (N-2, N-1, N, N+1, N+2) systems
        ext (str): output file extension

    Returns:
        data (pd.DataFrame): summarized results
    """
    if ip and ea: raise Exception("Can't do just ip as well as just ea.")
    codemap = {'gaussian': gaussian.Output,
                 'nwchem': nwchem.Output}
    if ip: keys = ['cat', 'neut']
    elif ea: keys = ['neut', 'an']
    else:
        if deep: keys = ['cat2', 'cat', 'neut', 'an', 'an2']
        else: keys = ['cat', 'neut', 'an']
    files = _dir_to_dict(adir, tuning=True)
    dtype = [('gamma', 'f8'), ('alpha', 'f8'), ('j2', 'f8'),
             ('cat2cur', 'f8'), ('catcur', 'f8'), ('ancur', 'f8'),
             ('an2cur', 'f8')]
    together = [[fls[key] for key in keys] for func, fls in files.items()]
    data = np.empty((len(together),), dtype=dtype)
    for i, mix in enumerate(together):
        comp, gam, alp, _ = mix[0].split(os.sep)[-1].split('-')    # _ == ion
        tag = '-'.join([gam, alp])
        if debug: print(mix)
        outs = [codemap[code](m) for m in mix]
        try:
            curv = compute_curvature(*outs, tag=tag)
        except:
            print(comp, tag, 'not computed')
            data[i] = (gam, alp) + (np.nan,) *(len(dtype)-2)
            continue
        miss = 4 - len(curv.curs)
        pre = miss // 2 + miss % 2
        pos = miss // 2
        curs = (0,) * pre + tuple(curv.curs) + (0,) * pos
        data[i] = (gam, alp, curv.j2) + curs
    return pd.DataFrame(data)
