# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Numerical Orbital Functions
#############################
Building discrete molecular orbitals (for visualization) requires a complex
set of operations that are provided by this module and wrapped into a clean API.
"""
# Established
import numpy as np
from numba import jit
from datetime import datetime

# Local
from exatomic.base import sym2z
from exatomic.core.field import AtomicField
from .orbital_util import (
    numerical_grid_from_field_params, gen_bfns, gen_gradients,
    _determine_fps, _determine_vector, _determine_bfns,
    _compute_current_density, _compute_orb_ang_mom,
    _compute_orbitals, _compute_density,
    _make_field, _evaluate_symbolic,)


#####################################################################
# Numba vectorized operations for Orbital, MOMatrix, Density tables #
# These will eventually be fully moved to matrices.py not meshgrid3d#
#####################################################################


@jit(nopython=True, nogil=True, parallel=True)
def build_pair_index(n):
    m = n**2
    x = np.empty((m, ), dtype=np.int64)
    y = x.copy()
    k = 0
    # Order matters so don't us nb.prange
    for i in range(n):
        for j in range(n):
            x[k] = i
            y[k] = j
            k += 1
    return x, y


@jit(nopython=True, nogil=True, parallel=True)
def density_from_momatrix(cmat, occvec):
    nbas = len(occvec)
    arlen = nbas * (nbas + 1) // 2
    dens = np.empty(arlen, dtype=np.float64)
    chi1 = np.empty(arlen, dtype=np.int64)
    chi2 = np.empty(arlen, dtype=np.int64)
    frame = np.empty(arlen, dtype=np.int64)
    cnt = 0
    for i in range(nbas):
        for j in range(i + 1):
            dens[cnt] = (cmat[i,:] * cmat[j,:] * occvec).sum()
            chi1[cnt] = i
            chi2[cnt] = j
            frame[cnt] = 0
            cnt += 1
    return chi1, chi2, dens, frame


@jit(nopython=True, nogil=True, parallel=True)
def density_as_square(denvec):
    nbas = int((-1 + np.sqrt(1 - 4 * -2 * len(denvec))) / 2)
    square = np.empty((nbas, nbas), dtype=np.float64)
    cnt = 0
    for i in range(nbas):
        for j in range(i + 1):
            square[i, j] = denvec[cnt]
            square[j, i] = denvec[cnt]
            cnt += 1
    return square


@jit(nopython=True, nogil=True, parallel=True)
def momatrix_as_square(movec):
    nbas = np.int64(len(movec) ** (1/2))
    square = np.empty((nbas, nbas), dtype=np.float64)
    cnt = 0
    for i in range(nbas):
        for j in range(nbas):
            square[j, i] = movec[cnt]
            cnt += 1
    return square


def add_orb_ang_mom(uni, field_params=None, rcoefs=None, icoefs=None,
                    frame=None, orbocc=None, maxes=None, inplace=True,
                    norm='Nd'):
    """Compute the orbital angular momentum and add it to a universe."""
    t0 = datetime.now()
    frame = uni.atom.nframes - 1 if frame is None else frame
    if (rcoefs not in uni.momatrix.columns) or \
       (icoefs not in uni.momatrix.columns):
        print("Either rcoefs {} or icoefs {} are " \
              "not in uni.momatrix".format(rcoefs, icoefs))
        return
    orbocc = rcoefs if orbocc is None else orbocc
    if orbocc not in uni.orbital.columns:
        print("orbocc {} is not in uni.orbital".format(orbocc))
        return
    if maxes is None:
        print("If magnetic axes are not an identity matrix, specify maxes.")
        maxes = np.eye(3)

    _determine_bfns(uni, frame, norm)
    fps = _determine_fps(uni, field_params, 4)
    x, y, z = numerical_grid_from_field_params(fps)
    occvec = uni.orbital[orbocc].values

    bfns = uni.basis_functions[frame]
    grx, gry, grz = gen_gradients(bfns)
    t1 = datetime.now()
    bvs = _evaluate_symbolic(bfns, x, y, z)
    grx = _evaluate_symbolic(grx, x, y, z)
    gry = _evaluate_symbolic(gry, x, y, z)
    grz = _evaluate_symbolic(grz, x, y, z)
    t2 = datetime.now()
    print('Timing: grid evaluation     - {:.2f}s'.format((t2-t1).total_seconds()))
    cmatr = uni.momatrix.square(column=rcoefs).values
    cmati = uni.momatrix.square(column=icoefs).values
    curx, cury, curz = _compute_current_density(bvs, grx, gry, grz, cmatr, cmati, occvec)
    t3 = datetime.now()
    print('Timing: current density 1D  - {:.2f}s'.format((t3-t2).total_seconds()))
    ang_mom = _compute_orb_ang_mom(x, y, z, curx, cury, curz, maxes)
    if not inplace: return _make_field(ang_mom, fps)
    uni.add_field(_make_field(ang_mom, fps))



def add_density(uni, field_params=None, mocoefs=None, orbocc=None,
                inplace=True, frame=None, norm='Nd'):
    """Compute a density and add it to a universe."""
    t1 = datetime.now()
    frame = uni.atom.nframes - 1 if frame is None else frame
    if mocoefs is None:
        mocoefs = 'coef'
        if orbocc is None:
            orbocc = 'occupation'
    else:
        if orbocc is None:
            orbocc = mocoefs
    if (mocoefs not in uni.momatrix.columns) or \
       (orbocc not in uni.orbital.columns):
        print('Either mocoefs {} is not in uni.momatrix or'.format(mocoefs))
        print('orbocc {} is not in uni.orbital'.format(orbocc))
        return
    _determine_bfns(uni, frame, norm)
    orbs = uni.momatrix.groupby('orbital')
    vector = np.array(range(uni.momatrix.orbital.max() + 1))
    fps = _determine_fps(uni, field_params, len(vector))

    x, y, z = numerical_grid_from_field_params(fps)
    bflds = _evaluate_symbolic(uni.basis_functions[frame], x, y, z)
    cmat = uni.momatrix.square(column=mocoefs).values
    oflds = _compute_orbitals(bflds, vector, cmat)
    # oflds = _compute_orbitals(bflds, vector, orbs, mocoefs)
    dens = _compute_density(oflds, uni.orbital[orbocc].values)
    t2 = datetime.now()
    print('Timing: compute density     - {:.2f}s'.format((t2-t1).total_seconds()))

    if not inplace: return _make_field(dens, fps.loc[0])
    uni.add_field(_make_field(dens, fps.loc[0]))



def add_molecular_orbitals(uni, field_params=None, mocoefs=None,
                           vector=None, frame=None, inplace=True,
                           replace=True, norm='Nd'):
    """
    If a universe contains enough information to generate
    molecular orbitals (basis_set, basis_set_order and momatrix),
    evaluate the molecular orbitals on a numerical grid. If vector
    is not provided, attempts to calculate orbitals by the orbital
    table, or by the sum of Z (Zeff) of the atoms in the atom table
    divided by two; roughly (HOMO-5,LUMO+7).

    Args
        uni (exatomic.container.Universe): a universe
        field_params (dict,pd.Series): dict with {'rmin', 'rmax', 'nr', ...}
        mocoefs (str): column in momatrix (default 'coef')
        vector (int, list, range, np.array): the MO vectors to evaluate
        inplace (bool): if False, return the field obj instead of modifying uni
        replace (bool): if False, do not delete any previous fields

    Warning:
       If replace is True, removes any fields previously attached to the universe
    """
    t1 = datetime.now()
    print('Warning: not extensively validated. Consider adding tests.')
    # Preliminary assignment and array dimensions
    frame = uni.atom.nframes - 1
    vector = _determine_vector(uni, vector)
    if mocoefs is None: mocoefs = 'coef'
    if mocoefs not in uni.momatrix.columns:
        print('mocoefs {} is not in uni.momatrix'.format(mocoefs))
        return
    fps = _determine_fps(uni, field_params, len(vector))

    print('Evaluating {} basis functions once.'.format(
        len(uni.basis_set_order.index)))
    _determine_bfns(uni, frame, norm)

    x, y, z = numerical_grid_from_field_params(fps)
    orbs = uni.momatrix.groupby('orbital')
    bflds = _evaluate_symbolic(uni.basis_functions[frame], x, y, z)
    print(mocoefs)
    cmat = uni.momatrix.square(column=mocoefs).values
    oflds = _compute_orbitals(bflds, vector, cmat)
    #oflds = _compute_orbitals(bflds, vector, orbs, mocoefs)
    field = _make_field(oflds, fps)

    t2 = datetime.now()
    print('Timing: compute orbitals    - {:.2f}s'.format((t2-t1).total_seconds()))

    if not inplace: return field
    if replace:
        if hasattr(uni, '_field'):
            del uni.__dict__['_field']
    uni.add_field(field)
