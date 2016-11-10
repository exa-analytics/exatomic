# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Delocalization
################################
Miscellaneous functions for computing the curvature in the energy
of a system as a function of the electron number. These functions
require results from 3 different quantum chemical calculations on
an (N-1), N, and (N+1) electron system.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from exatomic import Energy

sns.mpl.pyplot.rcParams.update({'text.usetex': True,
                                'font.family': 'serif',
                                'font.serif': ['Times'],
                                'ytick.labelsize': 24,
                                'xtick.labelsize': 24})

def _deltaE(col):
    if col.name == 'n': return col
    cat = np.linspace(col.max(), 0, 51)
    an = np.linspace(0, col.min(), 51)
    return col - np.hstack([cat, an[1:]])

def plot_en(deloc, title='', delta=None, xlabel='$\Delta$N',
            ylabel='$\Delta$E (eV)', figsize=(5,5), legpos=[1.1,-0.0]):
    fig = sns.mpl.pyplot.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    color = sns.color_palette('viridis', deloc.shape[1] - 1)
    if delta is not None:
        deloc.apply(_deltaE).plot(ax=ax, x='n', color=color, title=title)
    else:
        deloc.plot(ax=ax, x='n', color=color, title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.mpl.pyplot.legend(loc=legpos)
    return fig


def combine_deloc(delocs, order=None):
    """
    Given a list of the results of compute_deloc, return a single
    dataframe containing all of the E(N) results.
    """
    if order is not None:
        oldorder = [deloc.name for deloc in delocs]
        reordered = []
        for ordr in order:
            for i, old in enumerate(oldorder):
                if ordr.upper() == old.upper():
                    reordered.append(delocs[i])
    else:
        reordered = delocs
    for i, deloc in enumerate(reordered):
        if i > 0:
            try:
                reordered[i].drop('n', axis=1, inplace=True)
            except ValueError:
                pass
    return pd.concat(reordered, axis=1)


def compute_curvature(*args, neut=None, tag='', extras=True):
    """
    Computes the curvature of the energy of a system as a function
    of the number of electrons in the system E(N).

    Args
        args (exatomic.container.Universes): in ascending electron order
        neut (int): index of args corresponding to the zero energy system
        extras (bool): if True, attach the raw data to df before returning

    Returns
        df (pd.DataFrame): The energy as a function of N
    """
    if len(args) == 1:
        raise Exception("Must have at least 2 systems "
                        "differing in electron number.")
    nargs = len(args)
    neut = nargs // 2 if neut None else neut
    # Len (nargs) arrays
    totens, aorbs, borbs = [], [], []
    # Len (nargs - 1) arrays
    lumos, homos, js, diffs, es, cs = [], [], [], [], [], []
    for job in jobs:
        # Get the total energy of the system
        totens.append(job.frame['E_tot'].values[-1])
        # Get the highest occupied molecular orbital of the system
        aorbs.append(job.orbital.get_orbital())
        # Check for open shell, if not use alpha orbital data
        try: borbs.append(job.orbital.get_orbital(spin=1))
        except IndexError: borbs.append(aorbs[-1])
    # Cycle over the orbitals to find the right energies
    for job, alo, ahi, blo, bhi in zip(args, aorbs, aorbs[:1], borbs, borbs[:1]):
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
    atv = exatomic.Energy['Ha', 'eV']
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

#def compute_deloc(cat, neut, an, tag='', debug=False, jtype=None):
#    """
#    Computes the curvature of the energy of a system as a function
#    of the number of electrons in the system E(N).
#
#    Args
#        cat (exatomic.Universe): N-1 electron system
#        neut (exatomic.Universe): N electron system
#        an (exatomic.Universe): N+1 electron system
#        debug (bool): verbose printing
#        jtype (str): 'IP' or 'EA' if not both
#
#    Returns
#        ret (pd.DataFrame): The energy as a function of N
#    """
#    cat_en = cat.frame.ix[0].total_energy
#    neut_en = neut.frame.ix[0].total_energy
#    an_en = an.frame.ix[0].total_energy
#
#    # Get the highest occupied molecular orbitals of each system
#    alpha_cat_homo = cat.orbital.get_orbital()
#    alpha_neut_homo = neut.orbital.get_orbital()
#    alpha_an_homo = an.orbital.get_orbital()
#
#    # Check for open shell nature of any of the systems
#    try:
#        beta_cat_homo = cat.orbital.get_orbital(spin=1)
#    except IndexError:
#        beta_cat_homo = alpha_cat_homo
#    try:
#        beta_neut_homo = neut.orbital.get_orbital(spin=1)
#    except IndexError:
#        beta_neut_homo = alpha_neut_homo
#    try:
#        beta_an_homo = an.orbital.get_orbital(spin=1)
#    except IndexError:
#        beta_an_homo = alpha_an_homo
#
#    # Find the right orbital energies
#    if alpha_cat_homo.vector < alpha_neut_homo.vector:
#        lumoca = cat.orbital.get_orbital(index=alpha_cat_homo.name + 1).energy
#        homone = alpha_neut_homo.energy
#    if beta_cat_homo.vector < beta_neut_homo.vector:
#        lumoca = cat.orbital.get_orbital(index=beta_cat_homo.name + 1).energy
#        homone = beta_neut_homo.energy
#    if alpha_neut_homo.vector < alpha_an_homo.vector:
#        lumone = neut.orbital.get_orbital(index=alpha_neut_homo.name + 1).energy
#        homoan = alpha_an_homo.energy
#    if beta_neut_homo.vector < beta_an_homo.vector:
#        lumone = neut.orbital.get_orbital(index=beta_neut_homo.name + 1).energy
#        homoan = beta_an_homo.energy
#
#    #Compute J^2
#    jone = homone + (cat_en - neut_en)
#    jtwo = homoan + (neut_en - an_en)
#    jtype = None
#    j2 = jone ** 2 + jtwo ** 2
#    if jtype == 'EA':
#        j2 = jone ** 2
#    elif jtype == 'IP':
#        j2 = jtwo ** 2
#
#    #Compute E(n) and curvature coefficients
#    q = np.linspace(0, 1, 51)
#    negdE = an_en - neut_en
#    posdE = neut_en - cat_en
#    autoev = Energy['Ha', 'eV']
#    negE = (negdE*q + ((lumone - negdE)*(1 - q) + (negdE - homoan)*q)*q*(1 - q)) * autoev
#    posE = (-posdE*q + ((lumoca - posdE)*(1 - q) + (posdE - homone)*q)*q*(1 - q)) * autoev
#    ancur = (np.sum((lumone - negdE)*(6*q - 4) + (negdE - homoan)*(2 - 6*q)))/(len(q)*2) * autoev
#    catcur = (np.sum((lumoca - posdE)*(6*q - 4) + (posdE - homone)*(2 - 6*q)))/(len(q)*2) * autoev
#    colname = '{} ({:.2f},{:.2f})'.format(tag, catcur, ancur)
#    data = np.empty((len(q)*2 - 1,), dtype = [('n', 'f8'), (colname, 'f8')])
#    data['n'] = np.concatenate((np.fliplr([-q])[0][:-1], q))
#    data[colname] = np.concatenate((np.fliplr([posE])[0][:-1], negE))
#
#    #Proper object and tack on tidbits
#    ret = pd.DataFrame(data)
#    ret.ancur = ancur
#    ret.catcur = catcur
#    ret.j2 = j2
#    ret.name = tag
#    ret.colname = colname
#    if debug:
#        print('============', tag, '============')
#        print('alpha cation HOMO =', alpha_cat_homo, sep='\n')
#        print('beta cation HOMO =', beta_cat_homo, sep='\n')
#        if alpha_cat_homo.vector < alpha_neut_homo.vector:
#            print('lumoca =',
#                  cat.orbital.get_orbital(index=alpha_cat_homo.name + 1), sep='\n')
#        if beta_cat_homo.vector < beta_neut_homo.vector:
#            print('lumoca =',
#                  cat.orbital.get_orbital(index=beta_cat_homo.name + 1), sep='\n')
#        print('alpha neutral HOMO =', alpha_neut_homo, sep='\n')
#        print('beta neutral HOMO =', beta_neut_homo, sep='\n')
#        if alpha_neut_homo.vector < alpha_an_homo.vector:
#            print('lumone =',
#                  neut.orbital.get_orbital(index=alpha_an_homo.name + 1), sep='\n')
#        if beta_neut_homo.vector < beta_an_homo.vector:
#            print('lumone =',
#                  neut.orbital.get_orbital(index=beta_an_homo.name + 1), sep='\n')
#        print('alpha anion HOMO =', alpha_an_homo, sep='\n')
#        print('beta anion HOMO =', beta_an_homo, sep='\n')
#        print('lumoca energy = ', lumoca)
#        print('homone energy = ', homone)
#        print('lumone energy = ', lumone)
#        print('homoan energy = ', homoan)
#        print('cat energy = ', cat_en)
#        print('neut energy = ', neut_en)
#        print('an energy = ', an_en)
#    return ret
#
