# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Molecular Orbital Utilities
##############################
Molecular orbitals are constructed symbolically
then evaluated on a numerical grid.
These are their stories.
'''
import numpy as np
import pandas as pd
from numba import jit
from numexpr import evaluate, set_vml_accuracy_mode
acc = set_vml_accuracy_mode('high')
# print('Numexpr accuracy set to:', acc)

from .basis import (CartesianBasisFunction,
                    SphericalBasisFunction)
from exatomic.core.field import AtomicField


def compare_fields(*unis, rtol=5e-5, atol=1e-12, mtol=None, signed=True, verbose=True):
    """Compare field values of multiple universe.
    It is expected that fields are in the same order."""
    flds = (uni.field.field_values for uni in unis)
    kws = {'rtol': rtol, 'atol': atol}
    fracs = []
    if verbose:
        fmt = '{:<12}:{:>18}{:>18}'
    for i, fls in enumerate(zip(*flds)):
        compare = fls[0]
        if not i and verbose: print(fmt.format(len(compare), "Np.isclose(0, 1)", "Np.isclose(1, 0)"))
        percents = []
        for fl in fls[1:]:
            n = np.isclose(compare, fl, **kws).sum()
            on = np.isclose(fl, compare, **kws).sum()
            if not signed:
                m = np.isclose(compare, -fl, **kws).sum()
                om = np.isclose(-fl, compare, **kws).sum()
                n = max(n, m)
                on = max(on, om)
            percents.append((n / len(compare)) * 100)
            percents.append((on / len(compare)) * 100)
            fracs.append(n / len(compare))
            fracs.append(on / len(compare))
        form = '{:<12}:' + '{:>18.12f}' * len(percents)
        if verbose:
            print(form.format(i, *percents))
    if not verbose:
        return fracs


def gen_bfns(uni, frame=None, norm='Nd'):
    """Generate a list of symbolic basis functions
    from a universe containing basis set information."""
    frame = uni.atom.nframes - 1 if frame is None else frame
    sets = uni.basis_set[(uni.basis_set['d'] != 0) &
                         (uni.basis_set['frame'] == frame)].groupby('set')
    funcs = uni.basis_set_order[
                uni.basis_set_order['frame'] == frame].groupby('center')
    atom = uni.atom[uni.atom['frame'] == frame]
    if uni.basis_set.spherical:
        return _gen_spher_basfns(atom, funcs, sets, norm=norm)
    return _gen_cart_basfns(atom, funcs, sets, uni.basis_set.gaussian)


def _gen_cart_basfns(atom, funcs, sets, gaussian=True):
    """Return a list of cartesian basis functions."""
    basfns = []
    args = ['L', 'l', 'm', 'n']
    if not gaussian:
        raise NotImplementedError("Re-work ADF stuff.")
        args += ['r', 'prefac']
    args += ['shell']
    for i, (seht, x, y, z) in enumerate(zip(atom['set'], atom['x'],
                                            atom['y'], atom['z'])):
        bas = sets.get_group(seht).groupby('L')
        ordr = funcs.get_group(i)
        for args in zip(*(ordr[col] for col in args)):
            shell = bas.get_group(args[0]).groupby('shell').get_group(args[-1])
            basfns.append(CartesianBasisFunction(x, y, z, shell['Nd'],
                                                 shell['alpha'], args[1],
                                                 args[2], args[3]))
    return basfns


def _gen_spher_basfns(atom, funcs, sets, norm='Nd'):
    """Return a list of spherical basis functions."""
    basfns = []
    for i, (seht, x, y, z) in enumerate(zip(atom['set'], atom['x'],
                                            atom['y'], atom['z'])):
        bas = sets.get_group(seht).groupby('L')
        ordr = funcs.get_group(i)
        for L, ml, shell in zip(ordr['L'], ordr['ml'], ordr['shell']):
            shell = bas.get_group(L).groupby('shell').get_group(shell)
            basfns.append(SphericalBasisFunction(x, y, z, shell[norm],
                                                 shell['alpha'], L, ml))
    return basfns


def gen_gradients(bfns):
    """Evaluate symbolic gradients."""
    grx = [bas.gradient(cart='x') for bas in bfns]
    gry = [bas.gradient(cart='y') for bas in bfns]
    grz = [bas.gradient(cart='z') for bas in bfns]
    return grx, gry, grz


def _evaluate_symbolic(bfns, x, y, z):
    """Evaluate symbolic functions on a numerical grid."""
    flds = np.empty((len(bfns), len(x)), dtype=np.float64)
    for i, bas in enumerate(bfns):
        flds[i] = evaluate(str(bas), optimization='moderate')
    return flds


@jit(nopython=True, nogil=True, parallel=True)
def _compute_orbitals(bvs, vecs, cmat): #orbs, mocoefs):
    """Compute orbitals from numerical basis functions."""
    ovs = np.empty((len(vecs), bvs.shape[1]), dtype=np.float64)
    for i, vec in enumerate(vecs):
        ovs[i] = np.dot(cmat[:, vec], bvs)
    return ovs


@jit(nopython=True, nogil=True, parallel=True)
def _compute_density(ovs, occvec):
    """Sum orbitals multiplied by their occupations."""
    norb, npts = ovs.shape
    dens = np.empty(npts, dtype=np.float64)
    for i in range(norb):
        ovs[i] *= ovs[i]
    dens = np.dot(occvec, ovs)
    return dens


@jit(nopython=True, nogil=True, parallel=True)
def _compute_current_density(bvs, gvx, gvy, gvz, cmatr, cmati, occvec):
    """Compute the current density in each cartesian direction."""
    nbas, npts = bvs.shape
    chigradx = np.empty(npts, dtype=np.float64)
    chigrady = np.empty(npts, dtype=np.float64)
    chigradz = np.empty(npts, dtype=np.float64)
    curx = np.zeros(npts, dtype=np.float64)
    cury = np.zeros(npts, dtype=np.float64)
    curz = np.zeros(npts, dtype=np.float64)
    for mu in range(nbas):
        for nu in range(nbas):
            cval = (-0.5 * (occvec * (cmatr[mu] * cmati[nu]
                                    - cmati[mu] * cmatr[nu]))).sum()
            chigradx = bvs[mu] * gvx[nu] - gvx[mu] * bvs[nu]
            chigrady = bvs[mu] * gvy[nu] - gvy[mu] * bvs[nu]
            chigradz = bvs[mu] * gvz[nu] - gvz[mu] * bvs[nu]
            curx += cval * chigradx
            cury += cval * chigrady
            curz += cval * chigradz
    return curx, cury, curz


@jit(nopython=True, nogil=True, parallel=True)
def _compute_orb_ang_mom(rx, ry, rz, jx, jy, jz, mxs):
    """Compute the orbital angular momentum in each direction and the sum."""
    npts = rx.shape[0]
    ang_mom = np.empty((4, npts), dtype=np.float64)
    a0 = ry * jz - rz * jy
    a1 = rz * jx - rx * jz
    a2 = rx * jy - ry * jx
    ang_mom[0] = mxs[0,0] * a0 + mxs[1,0] * a1 + mxs[2,0] * a2
    ang_mom[1] = mxs[0,1] * a0 + mxs[1,1] * a1 + mxs[2,1] * a2
    ang_mom[2] = mxs[0,2] * a0 + mxs[1,2] * a1 + mxs[2,2] * a2
    ang_mom[3] = ang_mom[0] + ang_mom[1] + ang_mom[2]
    return ang_mom


# @jit(nopython=True, nogil=True, cache=True)
# def compute_mos(basvals, cmat):
#     npts, nbas = basvals.shape
#     orbs = np.zeros((npts, nbas), dtype=np.float64)
#     for p in range(nbas):
#         for mu in range(nbas):
#             orbs[:,p] += cmat[mu, p] * basvals[:,mu]
#     return orbs

# @jit(nopython=True, nogil=True, cache=True)
# def nb_compute_density(basvals, cmat, occvec):
#     npts, nbas = basvals.shape
#     dens = np.zeros(npts, dtype=np.float64)
#     orb = np.zeros(npts, dtype=np.float64)
#     for p in range(nbas):
#         for mu in range(nbas):
#             orb += cmat[mu, p] * basvals[:,mu]
#         dens += occvec[p] * orb ** 2
#         orb[:] = 0
#     return dens

# @jit(nopython=True, nogil=True, cache=True)
# def compute_dumb_density(basvals, cmat, occvec):
#     npts, nbas = basvals.shape
#     dens = np.zeros(npts, dtype=np.float64)
#     orb = np.zeros(npts, dtype=np.float64)
#     for p in range(nbas):
#         for mu in range(nbas):
#             for nu in range(nbas):
#                 dens += occvec[p] * cmat[mu, p] * cmat[nu, p] * basvals[:,mu] * basvals[:,nu]
#     return dens

def _make_field(flds, fps):
    """Return an AtomicField from field arrays and parameters."""
    try:
        nvec = flds.shape[0]
        if len(fps.index) == nvec:
            fps.reset_index(drop=True, inplace=True)
            return AtomicField(
                fps, field_values=[flds[i] for i in range(nvec)])
        return AtomicField(
            make_fps(nrfps=nvec, fps=fps),
            field_values=[flds[i] for i in range(nvec)])
    except:
        return AtomicField(
            make_fps(nrfps=1, **fps),
            field_values=[flds])


@jit(nopython=True, nogil=True, parallel=True)
def meshgrid3d(x, y, z):
    tot = len(x) * len(y) * len(z)
    xs = np.empty(tot, dtype=np.float64)
    ys = np.empty(tot, dtype=np.float64)
    zs = np.empty(tot, dtype=np.float64)
    cnt = 0
    for i in x:
        for j in y:
            for k in z:
                xs[cnt] = i
                ys[cnt] = j
                zs[cnt] = k
                cnt += 1
    return xs, ys, zs


def numerical_grid_from_field_params(fps):
    if isinstance(fps, pd.DataFrame):
        fps = fps.loc[0]
    ox, nx, dx = fps.ox, int(fps.nx), fps.dxi
    oy, ny, dy = fps.oy, int(fps.ny), fps.dyj
    oz, nz, dz = fps.oz, int(fps.nz), fps.dzk
    mx = ox + (nx - 1) * dx
    my = oy + (ny - 1) * dy
    mz = oz + (nz - 1) * dz
    x = np.linspace(ox, mx, nx)
    y = np.linspace(oy, my, ny)
    z = np.linspace(oz, mz, nz)
    return meshgrid3d(x, y, z)


def _determine_bfns(uni, frame, norm):
    """Attach symbolic basis functions if they don't exist."""
    if hasattr(uni, 'basis_functions'):
        if frame in uni.basis_functions: pass
        else: uni.basis_functions[frame] = gen_bfns(uni, frame=frame, norm=norm)
    else: uni.basis_functions = {frame: gen_bfns(uni, frame=frame, norm=norm)}


def _determine_vector(uni, vector):
    """Find some orbital indices in a universe."""
    if isinstance(vector, int): return np.array([vector])
    typs = (list, tuple, range, np.ndarray)
    if isinstance(vector, typs): return np.array(vector)
    norb = len(uni.basis_set_order.index)
    if vector is None:
        if hasattr(uni, 'orbital'):
            homo = uni.orbital.get_orbital().vector
        elif hasattr(uni.frame, 'N_e'):
            homo = uni.frame['N_e'].values[0]
        elif hasattr(uni.atom, 'Zeff'):
            homo = uni.atom['Zeff'].sum() // 2
        elif hasattr(uni.atom, 'Z'):
            homo = uni.atom['Z'].sum() // 2
        else:
            uni.atom['Z'] = uni.atom['symbol'].map(sym2z)
            homo = uni.atom['Z'].sum() // 2
        if homo < 5:
            if norb < 10: return range(norb)
            else: return np.array(range(0, homo + 5))
        else: return np.array(range(homo - 5, homo + 7))
    else: raise TypeError('Try specifying vector as a list or int')


def _determine_fps(uni, fps, nvec):
    """Find some numerical grid paramters from a universe."""
    if fps is None:
        if hasattr(uni, 'field'):
            return make_fps(nrfps=nvec, **uni.field.loc[0])
        desc = uni.atom.describe()
        kwargs = {'xmin': desc['x']['min'] - 5,
                  'xmax': desc['x']['max'] + 5,
                  'ymin': desc['y']['min'] - 5,
                  'ymax': desc['y']['max'] + 5,
                  'zmin': desc['z']['min'] - 5,
                  'zmax': desc['z']['max'] + 5,
                  'nx': 41, 'ny': 41, 'nz': 41,
                  'nrfps': nvec}
        return make_fps(**kwargs)
    return make_fps(nrfps=nvec, **fps)


def make_fps(rmin=None, rmax=None, nr=None, nrfps=1,
             xmin=None, xmax=None, nx=None, frame=0,
             ymin=None, ymax=None, ny=None, field_type=0,
             zmin=None, zmax=None, nz=None, label=0,
             ox=None, fx=None, dxi=None, dxj=None, dxk=None,
             oy=None, fy=None, dyi=None, dyj=None, dyk=None,
             oz=None, fz=None, dzi=None, dzj=None, dzk=None,
             fps=None):
    """
    Generate the necessary field parameters of a numerical grid field
    as an exatomic.field.AtomicField.

    Args
        nrfps (int): number of field parameters with same dimensions
        rmin (float): minimum value in an arbitrary cartesian direction
        rmax (float): maximum value in an arbitrary cartesian direction
        nr (int): number of grid points between rmin and rmax
        xmin (float): minimum in x direction (optional)
        xmax (float): maximum in x direction (optional)
        ymin (float): minimum in y direction (optional)
        ymax (float): maximum in y direction (optional)
        zmin (float): minimum in z direction (optional)
        zmax (float): maximum in z direction (optional)
        nx (int): steps in x direction (optional)
        ny (int): steps in y direction (optional)
        nz (int): steps in z direction (optional)
        ox (float): origin in x direction (optional)
        oy (float): origin in y direction (optional)
        oz (float): origin in z direction (optional)
        dxi (float): x-component of x-vector specifying a voxel
        dxj (float): y-component of x-vector specifying a voxel
        dxk (float): z-component of x-vector specifying a voxel
        dyi (float): x-component of y-vector specifying a voxel
        dyj (float): y-component of y-vector specifying a voxel
        dyk (float): z-component of y-vector specifying a voxel
        dzi (float): x-component of z-vector specifying a voxel
        dzj (float): y-component of z-vector specifying a voxel
        dzk (float): z-component of z-vector specifying a voxel
        label (str): an identifier passed to the widget (optional)
        field_type (str): alternative identifier (optional)

    Returns
        fps (pd.Series): field parameters
    """
    if fps is not None: return pd.concat([fps.loc[0]] * nrfps, axis=1).T
    if any((par is None for par in [rmin, rmax, nr])):
        if all((par is not None for par in (xmin, xmax, nx,
                                            ymin, ymax, ny,
                                            zmin, zmax, nz))):
            pass
        elif all((par is None for par in (ox, dxi, dxj, dxk))):
            raise Exception("Must supply at least rmin, rmax, nr or field"
                            " parameters as specified by a cube file.")
    d = {}
    allcarts = [['x', 0, xmin, xmax, nx, ox, (dxi, dxj, dxk)],
                ['y', 1, ymin, ymax, ny, oy, (dyi, dyj, dyk)],
                ['z', 2, zmin, zmax, nz, oz, (dzi, dzj, dzk)]]
    for akey, aidx, amin, amax, na, oa, da in allcarts:
        if oa is None:
            amin = rmin if amin is None else amin
            amax = rmax if amax is None else amax
            na = nr if na is None else na
        else: amin = oa
        dw = [0, 0, 0]
        if all(i is None for i in da): dw[aidx] = (amax - amin) / na
        else: dw = da
        d[akey] = [amin, na, dw]
    fp = pd.Series({
        'dxi': d['x'][2][0], 'dyj': d['y'][2][1], 'dzk': d['z'][2][2],
        'dxj': d['x'][2][1], 'dyk': d['y'][2][2], 'dzi': d['z'][2][0],
        'dxk': d['x'][2][2], 'dyi': d['y'][2][0], 'dzj': d['z'][2][1],
        'ox': d['x'][0], 'oy': d['y'][0], 'oz': d['z'][0], 'frame': frame,
        'nx': d['x'][1], 'ny': d['y'][1], 'nz': d['z'][1], 'label': label,
        'field_type': field_type
        })
    return pd.concat([fp] * nrfps, axis=1).T
