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

from exatomic import Energy
#from exatomic.field import AtomicField
#from exatomic.container import Universe

#def stack_unis(unis, same_frame=False):
#    """
#    A non-general algorithm for combining ``universes'' into a single universe.
#    """
#    attrs = [key[1:] for key, value in vars(unis[0]).items() if key not in
#             ['_traits_need_update', '_widget', '_lines'] and key.startswith('_')]
#    dfs = {}
#    if same_frame:
#        for attr in attrs:
#            if attr == 'field':
#                continue
#            dfs[attr] = getattr(unis[0], attr)
#        if hasattr(unis[0], 'field'):
#            field_values = [i.field.field_values[0] for i in unis]
#            field_params = pd.concat([pd.DataFrame(unis[0].field)] * len(unis)).reset_index(drop=True)
#            field_params['frame'] = 0
#            #field_params['label'] = field_params['label'].astype(np.int64)
#            dfs['field'] = AtomicField(field_params, field_values=field_values)
#        return Universe(**dfs)
#    for attr in attrs:
#        if attr == 'field':
#            continue
#        subdfs = []
#        for i, uni in enumerate(unis):
#            smdf = getattr(uni, attr)
#            smdf['frame'] = i
#            subdfs.append(smdf)
#        adf = pd.concat(subdfs)
#        adf.reset_index(drop=True, inplace=True)
#        dfs[attr] = adf
#    if hasattr(unis[0], 'field'):
#        field_values = [i.field.field_values[0] for i in unis]
#        field_params = pd.concat([pd.DataFrame(unis[0].field)] * len(unis)).reset_index(drop=True)
#        field_params['frame'] = range(len(unis))
#        field_params['label'] = 0
#        field_params['label'] = field_params['label'].astype(np.int64)
#        dfs['field'] = AtomicField(field_params, field_values=field_values)
#    return Universe(**dfs)

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
            reordered[i].drop('n', axis=1, inplace=True)
    return pd.concat(reordered, axis=1)


def compute_deloc(cat, neut, an, tag='', debug=False, jtype=None):
    """
    Computes the curvature of the energy of a system as a function
    of the number of electrons in the system E(N).

    Args
        cat (exatomic.Universe): N-1 electron system
        neut (exatomic.Universe): N electron system
        an (exatomic.Universe): N+1 electron system

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
