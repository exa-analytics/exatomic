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


def compute_deloc(cat, neut, an, tag='', debug=False, jtype=None):
    """
    Computes the curvature of the energy of a system as a function
    of the number of electrons in the system E(N).

    Args
        cat (exatomic.Universe): N-1 electron system
        neut (exatomic.Universe): N electron system
        an (exatomic.Universe): N+1 electron system
        debug (bool): verbose printing
        jtype (str): 'IP' or 'EA' if not both

    Returns
        ret (pd.DataFrame): The energy as a function of N
    """
    cat_en = cat.frame.ix[0].total_energy
    neut_en = neut.frame.ix[0].total_energy
    an_en = an.frame.ix[0].total_energy

    # Get the highest occupied molecular orbitals of each system
    alpha_cat_homo = cat.orbital.get_orbital()
    alpha_neut_homo = neut.orbital.get_orbital()
    alpha_an_homo = an.orbital.get_orbital()

    # Check for open shell nature of any of the systems
    try:
        beta_cat_homo = cat.orbital.get_orbital(spin=1)
    except IndexError:
        beta_cat_homo = alpha_cat_homo
    try:
        beta_neut_homo = neut.orbital.get_orbital(spin=1)
    except IndexError:
        beta_neut_homo = alpha_neut_homo
    try:
        beta_an_homo = an.orbital.get_orbital(spin=1)
    except IndexError:
        beta_an_homo = alpha_an_homo

    # Find the right orbital energies
    if alpha_cat_homo.vector < alpha_neut_homo.vector:
        lumoca = cat.orbital.get_orbital(index=alpha_cat_homo.name + 1).energy
        homone = alpha_neut_homo.energy
    if beta_cat_homo.vector < beta_neut_homo.vector:
        lumoca = cat.orbital.get_orbital(index=beta_cat_homo.name + 1).energy
        homone = beta_neut_homo.energy
    if alpha_neut_homo.vector < alpha_an_homo.vector:
        lumone = neut.orbital.get_orbital(index=alpha_neut_homo.name + 1).energy
        homoan = alpha_an_homo.energy
    if beta_neut_homo.vector < beta_an_homo.vector:
        lumone = neut.orbital.get_orbital(index=beta_neut_homo.name + 1).energy
        homoan = beta_an_homo.energy

    #Compute J^2
    jone = homone + (cat_en - neut_en)
    jtwo = homoan + (neut_en - an_en)
    jtype = None
    j2 = jone ** 2 + jtwo ** 2
    if jtype == 'EA':
        j2 = jone ** 2
    elif jtype == 'IP':
        j2 = jtwo ** 2

    #Compute E(n) and curvature coefficients
    q = np.linspace(0, 1, 51)
    negdE = an_en - neut_en
    posdE = neut_en - cat_en
    autoev = Energy['Ha', 'eV']
    negE = (negdE*q + ((lumone - negdE)*(1 - q) + (negdE - homoan)*q)*q*(1 - q)) * autoev
    posE = (-posdE*q + ((lumoca - posdE)*(1 - q) + (posdE - homone)*q)*q*(1 - q)) * autoev
    ancur = (np.sum((lumone - negdE)*(6*q - 4) + (negdE - homoan)*(2 - 6*q)))/(len(q)*2) * autoev
    catcur = (np.sum((lumoca - posdE)*(6*q - 4) + (posdE - homone)*(2 - 6*q)))/(len(q)*2) * autoev
    colname = '{} ({:.2f},{:.2f})'.format(tag, catcur, ancur)
    data = np.empty((len(q)*2 - 1,), dtype = [('n', 'f8'), (colname, 'f8')])
    data['n'] = np.concatenate((np.fliplr([-q])[0][:-1], q))
    data[colname] = np.concatenate((np.fliplr([posE])[0][:-1], negE))

    #Proper object and tack on tidbits
    ret = pd.DataFrame(data)
    ret.ancur = ancur
    ret.catcur = catcur
    ret.j2 = j2
    ret.name = tag
    ret.colname = colname
    if debug:
        print('============', tag, '============')
        print('alpha cation HOMO =', alpha_cat_homo, sep='\n')
        print('beta cation HOMO =', beta_cat_homo, sep='\n')
        if alpha_cat_homo.vector < alpha_neut_homo.vector:
            print('lumoca =',
                  cat.orbital.get_orbital(index=alpha_cat_homo.name + 1), sep='\n')
        if beta_cat_homo.vector < beta_neut_homo.vector:
            print('lumoca =',
                  cat.orbital.get_orbital(index=beta_cat_homo.name + 1), sep='\n')
        print('alpha neutral HOMO =', alpha_neut_homo, sep='\n')
        print('beta neutral HOMO =', beta_neut_homo, sep='\n')
        if alpha_neut_homo.vector < alpha_an_homo.vector:
            print('lumone =',
                  neut.orbital.get_orbital(index=alpha_an_homo.name + 1), sep='\n')
        if beta_neut_homo.vector < beta_an_homo.vector:
            print('lumone =',
                  neut.orbital.get_orbital(index=beta_an_homo.name + 1), sep='\n')
        print('alpha anion HOMO =', alpha_an_homo, sep='\n')
        print('beta anion HOMO =', beta_an_homo, sep='\n')
        print('lumoca energy = ', lumoca)
        print('homone energy = ', homone)
        print('lumone energy = ', lumone)
        print('homoan energy = ', homoan)
        print('cat energy = ', cat_en)
        print('neut energy = ', neut_en)
        print('an energy = ', an_en)
    return ret
