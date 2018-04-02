# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Numerical Orbital Functions
#############################
Building discrete molecular orbitals (for visualization) requires a complex
set of operations that are provided by this module and wrapped into a clean API.
"""
import numpy as np
#from numba import jit
from datetime import datetime
from exatomic.base import sym2z
#from exatomic.core.field import AtomicField
from .orbital_util import (
    numerical_grid_from_field_params, _determine_fps,
    _determine_vector, _compute_orb_ang_mom, _compute_current_density,
    _compute_orbitals, _compute_density, _check_column, _make_field,
    _compute_orbitals_nojit)


def add_molecular_orbitals(uni, field_params=None, mocoefs=None,
                           vector=None, frame=0, inplace=True,
                           replace=False, verbose=True):
    """A universe must contain basis_set, basis_set_order, and
    momatrix attributes to use this function.  Evaluate molecular
    orbitals on a numerical grid.  Attempts to generate reasonable
    defaults if none are provided.  If vector is not provided,
    attempts to calculate orbitals by the orbital table, or by the
    sum of Z (Zeff) of the atoms in the atom table divided by two;
    roughly (HOMO-5,LUMO+7).

    Args
        uni (:class:`~exatomic.container.Universe`): a universe
        field_params (dict): See :func:`~exatomic.algorithms.orbital_util.make_fps`
        mocoefs (str): column in uni.momatrix (default 'coef')
        vector (int, list, range, np.array): the MO vectors to evaluate
        inplace (bool): if False, return the field obj instead of modifying uni
        replace (bool): if False, do not delete any previous fields

    Warning:
       If replace is True, removes any fields previously attached to the universe
    """
    if verbose:
        print('Warning: not extensively validated.' \
              ' Consider adding tests.')
    t1 = datetime.now()
    vector = _determine_vector(uni, vector)
    bfns = uni.basis_functions
    fps = _determine_fps(uni, field_params, len(vector))
    mocoefs = _check_column(uni, 'momatrix', mocoefs)
    if verbose:
        p1 = 'Evaluating {} basis functions once.'
        print(p1.format(len(uni.basis_set_order.index)))

    x, y, z = numerical_grid_from_field_params(fps)
    #orbs = uni.momatrix.groupby('orbital')
    bvs = bfns.evaluate(x, y, z)
    cmat = uni.momatrix.square(column=mocoefs).values
    try: ovs = _compute_orbitals(len(x), bvs, vector, cmat)
    except: ovs = _compute_orbitals_nojit(len(x), bvs, vector, cmat)
    field = _make_field(ovs, fps)
    t2 = datetime.now()
    if verbose:
        p2 = 'Timing: compute orbitals - {:>8.2f}s.'
        print(p2.format((t2-t1).total_seconds()))
    if not inplace: return field
    if replace and hasattr(uni, '_field'):
        del uni.__dict__['_field']
    uni.add_field(field)


def add_density(uni, field_params=None, mocoefs=None, orbocc=None,
                inplace=True, frame=0, norm='Nd', verbose=True):
    """A universe must contain basis_set, basis_set_order, and
    momatrix attributes to use this function.  Compute a density
    with C matrix mocoefs and occupation vector orbocc.

    Args
        uni (:class:`~exatomic.container.Universe`): a universe
        field_params (dict): See :func:`~exatomic.algorithms.orbital_util.make_fps`
        mocoefs (str): column in uni.momatrix (default 'coef')
        orbocc (str): column in uni.orbital (default 'occupation')
        inplace (bool): if False, return the field obj instead of modifying uni
    """
    t1 = datetime.now()
    mocoefs = _check_column(uni, 'momatrix', mocoefs)
    orbocc = mocoefs if orbocc is None and mocoefs != 'coef' else orbocc
    orbocc = _check_column(uni, 'orbital', orbocc)
    bfns = uni.basis_functions
    orbs = uni.momatrix.groupby('orbital')
    vector = np.array(range(uni.momatrix.orbital.max() + 1))
    fps = _determine_fps(uni, field_params, len(vector))

    x, y, z = numerical_grid_from_field_params(fps)
    bvs = bfns.evaluate(x, y, z)
    cmat = uni.momatrix.square(column=mocoefs).values
    try: ovs = _compute_orbitals(len(x), bvs, vector, cmat)
    except: ovs = _compute_orbitals_nojit(len(x), bvs, vector, cmat)
    dens = _compute_density(ovs, uni.orbital[orbocc].values)
    t2 = datetime.now()
    if verbose:
        p1 = 'Timing: compute density  - {:>8.2f}s.'
        print(p1.format((t2-t1).total_seconds()))
    if not inplace: return _make_field(dens, fps.loc[0])
    uni.add_field(_make_field(dens, fps.loc[0]))


def add_orb_ang_mom(uni, field_params=None, rcoefs=None, icoefs=None,
                    frame=0, orbocc=None, maxes=None, inplace=True,
                    norm='Nd', verbose=True):
    """A universe must contain basis_set, basis_set_order, and
    momatrix attributes to use this function.  Compute the orbital
    angular momentum.  Requires C matrices from SODIZLDENS.X.X.R,I
    files from Molcas.

    Args
        uni (:class:`~exatomic.container.Universe`): a universe
        field_params (dict): See :func:`~exatomic.algorithms.orbital_util.make_fps`
        rcoefs (str): column in uni.momatrix (default 'lreal')
        icoefs (str): column in uni.momatrix (default 'limag')
        orbocc (str): column in uni.orbital (default 'lreal')
        inplace (bool): if False, return the field obj instead of modifying uni
    """
    if rcoefs is None or icoefs is None:
        raise Exception("Must specify rcoefs and icoefs")
    t0 = datetime.now()
    orbocc = rcoefs if orbocc is None else orbocc
    rcoefs = _check_column(uni, 'momatrix', rcoefs)
    icoefs = _check_column(uni, 'momatrix', icoefs)
    if maxes is None:
        if verbose:
            print("If magnetic axes are not an identity " \
                  "matrix, specify maxes.")
        maxes = np.eye(3)
    t1 = datetime.now()
    bfns = uni.basis_functions
    fps = _determine_fps(uni, field_params, 4)

    x, y, z = numerical_grid_from_field_params(fps)
    occvec = uni.orbital[orbocc].values
    bvs = bfns.evaluate(x, y, z)
    grx = bfns.evaluate_diff(x, y, z, cart='x')
    gry = bfns.evaluate_diff(x, y, z, cart='y')
    grz = bfns.evaluate_diff(x, y, z, cart='z')
    t2 = datetime.now()
    if verbose:
        p1 = 'Timing: grid evaluation  - {:>8.2f}s.'
        print(p1.format((t2-t1).total_seconds()))

    cmatr = uni.momatrix.square(column=rcoefs).values
    cmati = uni.momatrix.square(column=icoefs).values
    curx, cury, curz = _compute_current_density(
        bvs, grx, gry, grz, cmatr, cmati, occvec, verbose=verbose)
    t3 = datetime.now()
    if verbose:
        p2 = 'Timing: current density  - {:>8.2f}s.'
        print(p2.format((t3-t2).total_seconds()))
    ang_mom = _compute_orb_ang_mom(x, y, z, curx, cury, curz, maxes)
    if not inplace: return _make_field(ang_mom, fps)
    uni.add_field(_make_field(ang_mom, fps))
