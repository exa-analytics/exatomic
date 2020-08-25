# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Numerical Orbital Functions
#############################
Building discrete molecular orbitals (for visualization) requires a complex
set of operations that are provided by this module and wrapped into a clean API.
"""
import numpy as np
from numba import TypingError
from datetime import datetime
from exatomic import func_log
from exatomic.base import sym2z
from .orbital_util import (
    numerical_grid_from_field_params, _determine_fps,
    _determine_vector, _compute_orb_ang_mom, _compute_current_density,
    _compute_density, _check_column, _make_field,
    _compute_orbitals_numba, _compute_orbitals_numpy)


def _setup_orbital(uni, verbose, vector, fps, icoefs, log=None, jcoefs=None, irrep=None):
    """Boilerplate for starting the functions in this module."""
    log = log or func_log(_setup_orbital)
    t1 = datetime.now()
    nbf = len(uni.basis_functions)
    if irrep is not None:
        nbf = len(uni.basis_set_order.groupby('irrep').get_group(irrep).index)
    if verbose:
        p1 = 'Evaluating {} basis functions once.'
        log.debug(p1.format(nbf))
    vector = _determine_vector(uni, vector, irrep)
    fps = _determine_fps(uni, fps, len(vector))
    x, y, z = numerical_grid_from_field_params(fps)
    bvs = uni.basis_functions.evaluate(x, y, z, irrep=irrep, verbose=verbose)
    icoefs = _check_column(uni, 'current_momatrix', icoefs)
    icoefs = uni.current_momatrix.square(column=icoefs, irrep=irrep).values
    if jcoefs is not None:
        jcoefs = _check_column(uni, 'current_momatrix', jcoefs)
        jcoefs = uni.current_momatrix.square(column=jcoefs).values
        return t1, vector, fps, x, y, z, bvs, icoefs, jcoefs
    return t1, vector, fps, x, y, z, bvs, icoefs

def _compute_orbital(verbose, npts, bvs, vector, cmat, log=None):
    log = log or func_log(_compute_orbital)
    try: ovs = _compute_orbitals_numba(npts, bvs, vector, cmat)
    except (ValueError, IndexError, AssertionError, TypingError):
        if verbose: log.error('numba eval failed, falling back to numpy')
        ovs = _compute_orbitals_numpy(npts, bvs, vector, cmat)
    return ovs

def _teardown_orbital(uni, verbose, field, t1, inplace, name='orbitals', log=None):
    """Boilerplate for finishing the functions in this module."""
    log = log or func_log(_teardown_orbital)
    if verbose:
        t2 = datetime.now()
        p2 = 'Timing: compute {:<8} - {:>8.2f}s.'
        log.info(p2.format(name, (t2-t1).total_seconds()))
    if not inplace: return field
    uni.add_field(field)



def add_molecular_orbitals(uni, field_params=None, mocoefs=None,
                           vector=None, frame=0, inplace=True,
                           replace=False, verbose=True, irrep=None):
    """A universe must contain basis_set, [basis_set_order], and
    momatrix attributes to use this function.  Evaluate molecular
    orbitals on a numerical grid.  Attempts to generate reasonable
    defaults if none are provided.  If vector is not provided,
    attempts to calculate vector from the orbital table, or by the
    sum of Z (Zeff) of the atoms in the atom table divided by two;
    roughly (HOMO-5, LUMO+7).

    Args:
        uni (:class:`~exatomic.core.universe.Universe`): a universe
        field_params (dict): See :func:`~exatomic.algorithms.orbital_util.make_fps`
        mocoefs (str): column in uni.current_momatrix (default 'coef')
        vector (int, list, range, np.ndarray): the MO vectors to evaluate
        inplace (bool): if False, return the field obj instead of modifying uni
        replace (bool): if False, do not delete any previous fields
        irrep (int): if symmetrized, the irrep to which the orbitals belong

    Warning:
        If replace is True, removes any fields previously attached to the universe
    """
    log = func_log(add_molecular_orbitals)
    if replace and hasattr(uni, '_field'): del uni.__dict__['_field']
    t1, vector, fps, x, y, z, bvs, mocoefs = \
        _setup_orbital(uni, verbose, vector, field_params, mocoefs, irrep=irrep, log=log)
    ovs = _compute_orbital(verbose, len(x), bvs, vector, mocoefs, log=log)
    field = _make_field(ovs, fps)
    return _teardown_orbital(uni, verbose, field, t1, inplace, log=log)


def add_density(uni, field_params=None, mocoefs=None, orbocc=None,
                inplace=True, frame=0, norm='Nd', verbose=True):
    """A universe must contain basis_set, [basis_set_order], and
    momatrix attributes to use this function.  Compute a density
    with C matrix mocoefs and occupation vector orbocc.

    Args:
        uni (:class:`~exatomic.container.Universe`): a universe
        field_params (dict): See :func:`~exatomic.algorithms.orbital_util.make_fps`
        mocoefs (str): column in uni.current_momatrix (default 'coef')
        orbocc (str): column in uni.orbital (default 'occupation')
        inplace (bool): if False, return the field obj instead of modifying uni
    """
    log = func_log(add_density)
    mocol = mocoefs
    t1, vector, fps, x, y, z, bvs, mocoefs = \
        _setup_orbital(uni, verbose, None, field_params, mocoefs, log=log)
    orbocc = mocol if orbocc is None and mocol != 'coef' else orbocc
    orbocc = _check_column(uni, 'orbital', orbocc)
    vector = uni.orbital[~np.isclose(uni.orbital[orbocc], 0)].index.values
    orbocc = uni.orbital.loc[vector][orbocc].values
    ovs = _compute_orbital(verbose, len(x), bvs, vector, mocoefs, log=log)
    field = _make_field(_compute_density(ovs, orbocc), fps.loc[0])
    return _teardown_orbital(uni, verbose, field, t1, inplace, name='density', log=log)


def add_orb_ang_mom(uni, field_params=None, rcoefs=None, icoefs=None,
                    frame=0, orbocc=None, maxes=None, inplace=True,
                    norm='Nd', verbose=True):
    """A universe must contain basis_set, [basis_set_order], and
    momatrix attributes to use this function.  Compute the orbital
    angular momentum.  Requires C matrices from SODIZLDENS.X.X.R,I
    files from Molcas.

    Args:
        uni (:class:`~exatomic.container.Universe`): a universe
        field_params (dict): See :func:`~exatomic.algorithms.orbital_util.make_fps`
        rcoefs (str): column in uni.current_momatrix (default 'lreal')
        icoefs (str): column in uni.current_momatrix (default 'limag')
        maxes (np.ndarray): 3x3 array of magnetic axes (default np.eye(3))
        orbocc (str): column in uni.orbital (default 'lreal')
        inplace (bool): if False, return the field obj instead of modifying uni
    """
    log = func_log(add_orb_ang_mom)
    if rcoefs is None or icoefs is None:
        raise Exception("Must specify rcoefs and icoefs")
    rcol = rcoefs
    t1, vector, fps, x, y, z, bvs, rcoefs, icoefs = \
        _setup_orbital(uni, verbose, None, field_params, rcoefs, jcoefs=icoefs)
    orbocc = rcol if orbocc is None else orbocc
    if maxes is None:
        maxes = np.eye(3)
        if verbose:
            log.debug("If magnetic axes are not an identity matrix, specify maxes.")
    occvec = uni.orbital[orbocc].values
    grx = uni.basis_functions.evaluate_diff(x, y, z, cart='x', verbose=verbose)
    gry = uni.basis_functions.evaluate_diff(x, y, z, cart='y', verbose=verbose)
    grz = uni.basis_functions.evaluate_diff(x, y, z, cart='z', verbose=verbose)
    t2 = datetime.now()
    if verbose:
        p1 = 'Timing: grid evaluation  - {:>8.2f}s.'
        log.info(p1.format((t2-t1).total_seconds()))
    curx, cury, curz = _compute_current_density(
        bvs, grx, gry, grz, rcoefs, icoefs, occvec, verbose=verbose)
    t3 = datetime.now()
    if verbose:
        p2 = 'Timing: current density  - {:>8.2f}s.'
        log.info(p2.format((t3-t2).total_seconds()))
    field = _make_field(_compute_orb_ang_mom(
        x, y, z, curx, cury, curz, maxes), fps)
    return _teardown_orbital(uni, verbose, field, t1, inplace, name='angmom')
