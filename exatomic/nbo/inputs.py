# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
exnbo Input Generator and Parser
===================================
"""
#
#import numpy as np
#import pandas as pd
#
#from exa.relational.isotope import symbol_to_z
#symbol_to_Z = symbol_to_z()
#
#from exatomic import __version__
#from .editor import Editor
#from exatomic.orbital import DensityMatrix
#from exatomic.basis import (solid_harmonics, lorder,
#                            cart_lml_count, spher_lml_count)
#from itertools import combinations_with_replacement as cwr
#
#_exaver = 'exatomic.v' + __version__
#
#_header = """\
#$GENNBO NATOMS={nat}    NBAS={nbas}  UPPER  BODM BOHR $END
#$NBO BNDIDX NLMO AONBO=W AONLMO=W $END
#$COORD
#{exaver} -- {name} -- tr[D*S] = {check}
#{atom}
#$END
#$BASIS
# CENTER = {center}
#  LABEL = {label}
#$END
#$CONTRACT
# NSHELL = {nshell:>7}
#   NEXP = {nexpnt:>7}
#  NCOMP = {ncomps}
#  NPRIM = {nprims}
#   NPTR = {npntrs}
#    EXP = {expnts}
#{coeffs}
#$END"""
#
#_matrices = """
#$OVERLAP
#{overlap}
#$END
#$DENSITY
#{density}
#$END"""
#
#def _nbo_labels():
#    """Generate data frames of L, (ml | l, m, n), NBO label."""
#    sph = pd.DataFrame(list(solid_harmonics(6).keys()),
#                       columns=('L', 'ml'))
#    # See the NBO 6.0 manual for more details
#    # This is the basis function labeling scheme
#    # In order of increasing ml from most negative
#    # to most positive in the same order as the
#    # results from the solid_harmonics code.
#    sph['label'] = [1,   101, 102, 103, 251, 253, 255,
#                    252, 254, 357, 355, 353, 351, 352,
#                    354, 356, 459, 457, 455, 453, 451,
#                    452, 454, 456, 458, 561, 559, 557,
#                    555, 553, 551, 552, 554, 556, 558,
#                    560, 663, 661, 659, 657, 655, 653,
#                    651, 652, 654, 656, 658, 660, 662]
#    Ls, ls, ms, ns, label = [], [], [], [], []
#    # Even NBO 6.0 doesn't support cartesian basis
#    # functions with an l value greater than g functions
#    for i in range(5):
#        t = i * 100 + 1
#        label += list(range(t, t + cart_lml_count[i]))
#        car = [''.join(i) for i in list(cwr('xyz', i))]
#        Ls += [i for k in car]
#        ls += [i.count('x') for i in car]
#        ms += [i.count('y') for i in car]
#        ns += [i.count('z') for i in car]
#    car = pd.DataFrame({'L': Ls, 'l': ls, 'm': ms,
#                        'n': ns, 'label': label})
#    return sph, car
#
#spher, cart = _nbo_labels()
#
#def _get_labels(Ls, mls=None, ls=None, ms=None, ns=None):
#    """Get the NBO labels corresponding to L, (ml | l, m, n)."""
#    if mls is not None:
#        return [spher[(spher['L'] == l) &
#                      (spher['ml'] == ml)]['label'].iloc[0]
#                      for l, ml in zip(Ls, mls)]
#    if xs is not None:
#        return [cart[(cart['L'] == L) &
#                     (cart['l'] == l) &
#                     (cart['m'] == m) &
#                     (cart['n'] == n)]['label'].iloc[0]
#                     for L, l, m, n in zip(Ls, ls, ms, ns)]
#
#def _clean_coeffs(arr, width=16, decimals=6):
#    """Call _clean_to_string for each shell."""
#    # Format C(shell) for coeffs
#    ls = ['     {} = '.format('C' + l.upper()) for l in lorder]
#    # Clean to string by shell
#    dat = [''.join([l, _clean_to_string(ar, decimals=decimals), '\n'])
#           for l, ar in zip(ls, arr)]
#    # Return the whole minus the last line break
#    return ''.join(dat)[:-1]
#
#def _clean_to_string(arr, ncol=4, width=16, decimals='', just=True):
#    """Convert a numerical array into nicely formatted text block."""
#    # Justify the data arrays with the tags in the template
#    pad = ' ' * 10 if just else ''
#    # Some flexibility in how this function handles int/floats
#    dec = '.' + str(decimals) + 'E' if decimals else decimals
#    # A format string for the numbers in the array
#    fmt = ''.join(['{:>', str(width), dec, '}'])
#    # The formmatted array with tabs and new line breaks
#    dat = [''.join(['\n', pad, fmt.format(a)]) if not i % ncol and i > 0
#           else fmt.format(a) for i, a in enumerate(arr)]
#    return ''.join(dat)
#
#def _obtain_arrays(uni):
#    """Get numerical arrays of information from a universe."""
#    kwargs = {}
#    # Get number of functions by shell
#    shells = uni.basis_set.functions_by_shell()
#    # This is how many times each L value shows up
#    shlcnt = shells.index.get_level_values(0)
#    # Add subshells for each time L shows up
#    shells = shells.groupby(shlcnt).apply(lambda x: x.sum())
#    # Map it onto the atoms with each basis set
#    nshell = uni.atom['set'].map(shells).sum()
#    kwargs['nshell'] = nshell
#    # Group our basis sets, will be used later
#    bases = uni.basis_set[np.abs(uni.basis_set['d']) > 0].groupby('set')
#    # Exponents per basis set
#    expnts = bases.apply(lambda x: x.shape[0])
#    # mapped onto the atoms with each basis set
#    nexpnt = uni.atom['set'].map(expnts).sum()
#    kwargs['nexpnt'] = nexpnt
#    # Grab already correct arrays from basis_set_order
#    kwargs['center'] = uni.basis_set_order['center'].values.copy()
#    kwargs['L'] = uni.basis_set_order['L'].values
#    if uni.basis_set.spherical:
#        # Spherical basis set
#        kwargs['ml'] = uni.basis_set_order['ml'].values
#        lml_count = spher_lml_count
#    else:
#        # Cartesian basis set
#        kwargs['l'] = uni.basis_set_order['l'].values
#        kwargs['m'] = uni.basis_set_order['m'].values
#        kwargs['n'] = uni.basis_set_order['n'].values
#        lml_count = cart_lml_count
#    # For the NBO specicific arrays
#    lmax = uni.basis_set['L'].cat.as_ordered().max()
#    # ---- There are 3 that are length nshell
#    # The number of components per basis function (l degeneracy)
#    ncomps = np.empty(nshell, dtype=np.int64)
#    # The number of primitive functions per basis function
#    nprims = np.empty(nshell, dtype=np.int64)
#    # The pointers in the arrays above for each basis funciton
#    npntrs = np.empty(nshell, dtype=np.int64)
#    # ---- And 2 that are length nexpnt
#    # The total number of exponents in the basis set
#    expnts = np.empty(nexpnt, dtype=np.float64)
#    # The contraction coefficients within the basis set
#    ds = np.empty((lmax + 1, nexpnt), dtype=np.float64)
#    # The following algorithm must be generalized
#    # and simplified by either some bound methods
#    # on basis_set attributes
#    cnt, ptr, xpc = 0, 1, 0
#    for seht in uni.atom['set']:
#        b = bases.get_group(seht)
#        for sh, grp in b.groupby('shell'):
#            if len(grp) == 0: continue
#            ncomps[cnt] = lml_count[grp['L'].values[0]]
#            nprims[cnt] = grp.shape[0]
#            npntrs[cnt] = ptr
#            ptr += nprims[cnt]
#            cnt += 1
#        for l, d, exp in zip(b['L'], b['d'], b['alpha']):
#            expnts[xpc] = exp
#            for i, ang in enumerate(ds):
#                ds[i][xpc] = d if i == l else 0
#            xpc += 1
#    kwargs['nprims'] = nprims
#    kwargs['ncomps'] = ncomps
#    kwargs['npntrs'] = npntrs
#    kwargs['expnts'] = expnts
#    kwargs['coeffs'] = ds
#    return kwargs
#
#
#class Input(Editor):
#
#    @classmethod
#    def from_universe(cls, uni, occvec=None, column=None, name=''):
#        """
#        Generate an NBO input from a properly populated universe.
#        uni must have atom, basis_set, basis_set_order, overlap,
#        momatrix and occupation_vector information.
#
#        Args
#            uni (:class:`~exatomic.container.Universe`): containing the above attributes
#            occvec (np.array): occupation vector that relates momatrix to density matrix
#
#        Returns
#            editor (:class:`~exatomic.nbo.Input`)
#        """
#        # Grab all array data from new orbital code
#        kwargs = _obtain_arrays(uni)
#        # Manicure it slightly for NBO inputs
#        kwargs['exaver'] = _exaver
#        kwargs['name'] = name
#        kwargs['center'] += 1
#        kwargs['nat'] = kwargs['center'].max()
#        kwargs['nbas'] = len(kwargs['center'])
#        kwargs['check'] = ''
#        columns = ('Z', 'Z', 'x', 'y', 'z')
#        if 'Zeff' in uni.atom.columns:
#            columns = ('Z', 'Zeff', 'x', 'y', 'z')
#        kwargs['atom'] = uni.atom.to_xyz(columns=columns)
#        # Assign appropriate NBO basis function labels
#        if 'ml' in kwargs:
#            labargs = {'mls': kwargs['ml']}
#        else:
#            labards = {'ls': kwargs['l'],
#                       'ms': kwargs['m'],
#                       'ns': kwargs['n']}
#        kwargs['label'] = _get_labels(kwargs['L'], **labargs)
#        # Clean the arrays to strings for a text input file
#        kwargs['label'] = _clean_to_string(kwargs['label'], ncol=10, width=5)
#        kwargs['center'] = _clean_to_string(kwargs['center'], ncol=10, width=5)
#        kwargs['ncomps'] = _clean_to_string(kwargs['ncomps'], ncol=10, width=5)
#        kwargs['nprims'] = _clean_to_string(kwargs['nprims'], ncol=10, width=5)
#        kwargs['npntrs'] = _clean_to_string(kwargs['npntrs'], ncol=10, width=5)
#        kwargs['expnts'] = _clean_to_string(kwargs['expnts'], decimals=6)
#        kwargs['coeffs'] = _clean_coeffs(kwargs['coeffs'])
#        # Separated matrices for debugging the top half when these
#        # arrays are harder to come by. NBO has strict precision
#        # requirements so overlap/density must be very precise (12 decimals).
#        matargs = {'overlap': '', 'density': ''}
#        margs = {'decimals': 6, 'just': False}
#        if hasattr(uni, '_overlap'):
#            o = uni.overlap['coef'].values
#            matargs['overlap'] = _clean_to_string(o, **margs)
#        # Still no clean solution for an occupation vector yet
#        if hasattr(uni, '_density'):
#            d = uni.density
#        elif hasattr(uni, 'occupation_vector'):
#            d = DensityMatrix.from_momatrix(uni.momatrix, uni.occupation_vector)
#        elif occvec is not None:
#            if column is None:
#                raise Exception("Must provide column name if providing occvec.")
#            d = DensityMatrix.from_momatrix(uni.momatrix, occvec, column=column)
#        matargs['density'] = _clean_to_string(d['coef'].values, **margs)
#        # Compute tr[P*S] must be equal to number of electrons
#        if matargs['density']:
#            kwargs['check'] = np.trace(np.dot(d.square(), uni.overlap.square()))
#        return cls(_header.format(**kwargs) + _matrices.format(**matargs))
#
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
