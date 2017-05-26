# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Numerical Orbital Functions
#############################
Building discrete molecular orbitals (for visualization) requires a complex
set of operations that are provided by this module and wrapped into a clean API.
"""
# Established
import sympy as sy
import numpy as np
import pandas as pd
import numexpr as ne
from datetime import datetime
from numba import jit, vectorize
from psutil import virtual_memory
from collections import OrderedDict

# Local
from exa import Series
from exatomic._config import config
from exatomic.field import AtomicField
from exa.relational.isotope import symbol_to_z
from exatomic.algorithms.basis import solid_harmonics, clean_sh

symbol_to_z = symbol_to_z()
halfmem = virtual_memory().total / 2
solhar = clean_sh(solid_harmonics(6))

#####################################################################
# Numba vectorized operations for Orbital, MOMatrix, Density tables #
# These will eventually be fully moved to matrices.py not meshgrid3d#
#####################################################################

@jit(nopython=True)
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

@jit(nopython=True)
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

@jit(nopython=True)
def momatrix_as_square(movec):
    nbas = np.int64(len(movec) ** (1/2))
    square = np.empty((nbas, nbas), dtype=np.float64)
    cnt = 0
    for i in range(nbas):
        for j in range(nbas):
            square[j, i] = movec[cnt]
            cnt += 1
    return square

@jit(nopython=True, cache=True, nogil=True)
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

################################################################
# Functions used in the generation of basis functions and MOs. #
################################################################

def make_fps(rmin=None, rmax=None, nr=None, nrfps=1,
             xmin=None, xmax=None, nx=None, frame=0,
             ymin=None, ymax=None, ny=None, field_type=0,
             zmin=None, zmax=None, nz=None, label=0,
             ox=None, dxi=None, dxj=None, dxk=None,
             oy=None, dyi=None, dyj=None, dyk=None,
             oz=None, dzi=None, dzj=None, dzk=None):
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
    if any((par is None for par in [rmin, rmax, nr])):
        if all((par is None for par in (ox, dxi, dxj, dxk))):
            raise Exception("Must supply at least rmin, rmax, nr or field"
                            " parameters as specified by a cube file.")
    d = {}
    allcarts = [['x', 0, xmin, xmax, nx, ox, (dxi, dxj, dxk)],
                ['y', 1, ymin, ymax, ny, oy, (dyi, dyj, dyk)],
                ['z', 2, zmin, zmax, nz, oz, (dzi, dzj, dzk)]]
    for akey, aix, amin, amax, na, oa, da in allcarts:
        if oa is None:
            amin = rmin if amin is None else amin
            amax = rmax if amax is None else amax
            na = nr if na is None else na
        else: amin = oa
        dw = [0, 0, 0]
        if all(i is None for i in da): dw[aix] = (amax - amin) / na
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


def _sphr_prefac(nuc, sh, L, ml):
    """
    Create strings of the pre-exponential factor of a given
    spherical basis function as a function of l, ml quantum numbers.

    Args
        L (int): angular momentum quantum numbers
        ml (int): magnetic quantum number
        nuc (dict): atomic position
        sh (dict): cleaned solid harmonics

    Returns
        prefacs (list): pre-exponential factors
    """

    return [pre.format(**nuc) for pre in sh[(L, ml)]]


def _cart_prefac(nuc, pre, L, l, m, n):
    """
    As with _sphr_prefac, create the string version of the pre-exponential
    factor in a given basis function, this time as a function of cartesian
    powers (l, m, n) instead of (l, ml) quantum numbers.

    Args
        L (int): angular momentum quantum number
        l (int): powers of x
        m (int): powers of y
        n (int): powers of z
        nuc (dict): atomic position
        pre (str): '' unless ADF

    Returns
        prefacs (list): pre-exponential factors
    """
    if not L: return [pre]
    if pre and not pre.endswith('*'): pre = '{}*'.format(pre)
    lin, nlin = '{}*', '{}**{}*'
    for cart, powr in OrderedDict([('{x}', l),
                                   ('{y}', m),
                                   ('{z}', n)]).items():
        if not powr: continue
        stargs = [cart]
        fmt = lin if powr == 1 else nlin
        if powr > 1: stargs.append(powr)
        pre += fmt.format(*stargs)
    return [pre.format(**nuc)]

def gen_basfn(prefacs, shell, rexp, precision=10):
    """
    Given a list of pre-exponential factors and a shell of
    primitive functions (slice of basis set table), return
    the string that is the basis function written out as it
    would be on paper.

    Args
        prefacs (list): string of pre-exponential factors
        shell (exatomic.basis.BasisSet): a shell of the basis set table
        rexp (str): the r-dependent exponent (including atomic position)

    Returns
        basis function (str)
    """
    bastr = '{prefac}({prims})'
    bastrs = []
    primitive = '{{Nd:.{precision}f}}*exp' \
                '(-{{alpha:.{precision}f}}*{rexp})'.format(rexp=rexp,
                                                           precision=precision)
    for prefac in prefacs:
        primitives = shell.apply(lambda x: primitive.format(**x), axis=1)
        bastrs.append(bastr.format(prefac=prefac,
                                   prims='+'.join(primitives.values)))
    return '+'.join(bastrs)


def gen_basfns(uni, frame=None):
    """
    Given an exatomic.container.Universe that contains complete momatrix
    and basis_set attributes, generates and returns the strings corresponding
    to how the basis functions would be written out with paper and pencil.
    This is mainly for debugging and testing generality to deal with different
    computational codes' basis function ordering schemes.

    Args
        uni (exatomic.container.Universe): must contain momatrix and basis_set

    Returns
        bastrs (list): list of strings of basis functions
    """
    frame = uni.atom.nframes - 1 if frame is None else frame
    # Group the dataframes appropriately
    sets = uni.basis_set.cardinal_groupby().get_group(frame).groupby('set')
    funcs = uni.basis_set_order.cardinal_groupby().get_group(frame).groupby('center')
    atom = uni.atom.cardinal_groupby().get_group(frame)
    # Set some variables based on basis set info
    larg = {'sh': None, 'pre': None}
    if uni.basis_set.spherical:
        # Basis set order columns for spherical functions
        ordrcols = ['L', 'ml', 'shell']
        prefunc = _sphr_prefac
        # To avoid bool checking for each basis function
        lkey = 'sh'
        # Get the string versions of the symbolic solid harmonics
        larg[lkey] = solhar
    else:
        # Basis set order columns for cartesian functions
        ordrcols = ['L', 'l', 'm', 'n', 'shell']
        prefunc = _cart_prefac
        # To avoid bool checking for each basis function
        lkey = 'pre'
        # Placeholder for potential custom prefactors from ADF
        larg[lkey] = ''
    # The number of arguments to pass to _prefac
    rgslice = slice(0, len(ordrcols) - 1)
    # In the case of ADF orbitals (currently only one that
    # requires 'r' and 'prefac' on a per basis function basis
    if not uni.basis_set.gaussian:
        ordrcols = ordrcols[:-1] + ['r', 'prefac'] + ordrcols[-1:]
        exkey = 'r'
    else: exkey = 'r2'
    # Iterate over atomic positions
    basfns = []
    for i, (seht, x, y, z) in enumerate(zip(atom['set'], atom['x'],
                                            atom['y'], atom['z'])):
        # Dict of strings of atomic position
        nuc = _atompos(x, y, z)
        # Regroup dataframes
        bas = sets.get_group(seht).groupby('L')
        ordr = funcs.get_group(i)
        # Iterate over atom centered basis functions
        for args in zip(*[ordr[col] for col in ordrcols]):
            # Get the shell of primitive functions
            shell = bas.get_group(args[0]).groupby('shell').get_group(args[-1])
            try:
                # Only used for ADF orbitals currently
                preexp, prefac = args[4:6]
                preexp = '' if not preexp else '({})**{}*'.format(nuc[exkey], preexp)
                prefac = '' if not prefac else '{}*'.format(prefac)
                larg['pre'] = '{}{}'.format(preexp, prefac)
            except ValueError:
                # Otherwise it's useless
                larg['pre'] = ''
            # The pre-exponential factors (besides the additional ADF ones)
            prefacs = prefunc(nuc, larg[lkey], *args[rgslice])
            # Put together the basis function
            basfns.append(gen_basfn(prefacs, shell, nuc[exkey]))
    return basfns


def _atompos(x, y, z, precision=10):
    nuc = {}
    p = '{{}}{{}}{{:.{}f}}'.format(precision).format
    for key, cart in [('x', x), ('y', y), ('z', z)]:
        if np.isclose(cart, 0):
            nuc[key] = key
        else:
            cart, op = (cart, '-') if cart > 0 else (np.abs(cart), '+')
            nuc[key] = '({})'.format(p(key, op, cart).strip('0'))
    nuc['r2'] = '({}**2+{}**2+{}**2)'.format(nuc['x'], nuc['y'], nuc['z'])
    nuc['r'] = '{}**0.5'.format(nuc['r2'])
    return nuc


def numerical_grid_from_field_params(fld_ps):
    if isinstance(fld_ps, pd.DataFrame):
        fld_ps = fld_ps.ix[fld_ps.index.values[0]]
    ox, nx, dx = fld_ps.ox, fld_ps.nx, fld_ps.dxi
    oy, ny, dy = fld_ps.oy, fld_ps.ny, fld_ps.dyj
    oz, nz, dz = fld_ps.oz, fld_ps.nz, fld_ps.dzk
    mx = ox + (nx - 1) * dx
    my = oy + (ny - 1) * dy
    mz = oz + (nz - 1) * dz
    x = np.linspace(ox, mx, nx)
    y = np.linspace(oy, my, ny)
    z = np.linspace(oz, mz, nz)
    return meshgrid3d(x, y, z)


def _determine_field_params(universe, field_params, nvec):
    if field_params is None:
        dr = 41
        rmin = min(universe.atom['x'].min(),
                   universe.atom['y'].min(),
                   universe.atom['z'].min()) - 4
        rmax = max(universe.atom['x'].max(),
                   universe.atom['y'].max(),
                   universe.atom['z'].max()) + 4
        return make_fps(rmin, rmax, dr, nrfps=nvec)
    else:
        return make_fps(nrfps=nvec, **field_params)


def _determine_vector(uni, vector):
    if isinstance(vector, int): return [vector]
    typs = (list, tuple, range, np.ndarray)
    if isinstance(vector, typs): return vector
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
            uni.atom['Z'] = uni.atom['symbol'].map(symbol_to_z)
            homo = uni.atom['Z'].sum() // 2
        if homo < 15: return range(0, homo + 15)
        else: return range(homo - 15, homo + 5)
    else: raise TypeError('Try specifying vector as a list or int')


def add_molecular_orbitals(uni, field_params=None, mocoefs=None,
                           vector=None, frame=None, inplace=True):
    """
    If a universe contains enough information to generate
    molecular orbitals (basis_set, basis_set_summary and momatrix),
    evaluate the molecular orbitals on a discretized grid. If vector
    is not provided, attempts to calculate vectors by the sum of Z/Zeff
    of the atoms present divided by two; roughly (HOMO-15,LUMO+5).

    Args
        uni (exatomic.container.Universe): a universe
        field_params (dict,pd.Series): dict with {'rmin', 'rmax', 'nr'}
        mocoefs (str): column in momatrix (default 'coef')
        vector (int, list, range, np.array): the MO vectors to evaluate
        inplace (bool): if False, return the field obj instead of modifying uni

    Warning:
       If inplace is True, removes any fields previously attached to the universe
    """
    # Preliminary assignment and array dimensions
    vector = _determine_vector(uni, vector)
    if mocoefs is None: mocoefs = 'coef'
    if mocoefs not in uni.momatrix.columns:
        print('mocoefs {} is not in uni.momatrix'.format(mocoefs))
        return
    fld_ps = _determine_field_params(uni, field_params, len(vector))
    x, y, z = numerical_grid_from_field_params(fld_ps)
    nbas = len(uni.basis_set_order.index)
    norb = uni.momatrix.orbital.max()
    nvec = len(vector)
    npts = len(x)

    print('Warning: not extensively validated. Consider adding tests.')
    # Build the strings corresponding to basis functions
    # basfns is frame dependent but takes a few seconds to generate,
    # so cache a frame's basis functions with a dict of key {frame: basfns} value
    if hasattr(uni, 'basfns'):
        if frame in uni.basfns: pass
        else: uni.basfns[frame] = gen_basfns(uni, frame=frame)
    else: uni.basfns = {frame: gen_basfns(uni, frame=frame)}
    orbs = uni.momatrix.groupby('orbital')

    t1 = datetime.now()
    # Evaluate basis functions one time and store all in a single
    # large numpy array which is much more efficient but can require
    # a lot of memory if the resolution of the field is very fine
    if (norb * npts * 8) < halfmem:
        print('Evaluating {} basis functions once.'.format(nbas))
        fields = np.empty((npts, nvec), dtype=np.float64)
        vals = np.empty((npts, nbas), dtype=np.float64)
        for i, bas in enumerate(uni.basfns[frame]): vals[:,i] = ne.evaluate(bas)
        for i, vec in enumerate(vector):
            fields[:,i] = (vals * orbs.get_group(vec)[mocoefs].values).sum(axis=1)
    # If the resolution of fields will be memory intensive, evaluate
    # each basis function on the fly per MO which saves a large
    # np.array in memory but is redundant and less efficient
    else:
        print('Evaluating {} basis functions {} times (slow).'.format(nbas, nvec))
        fields = np.zeros((npts, nvec), dtype=np.float64)
        for i, vec in enumerate(vector):
            c = orbs.get_group(vec)[mocoefs].values
            for j, bas in enumerate(uni.basfns[frame]):
                fields[:,i] += c[j] * ne.evaluate(bas)
    t2 = datetime.now()
    print('Timing: compute MOs - {:.2f}s'.format((t2-t1).total_seconds()))

    field = AtomicField(fld_ps, field_values=[fields[:,i] for i in range(nvec)])
    if not inplace: return field
    # Don't collect infinity fields if this is run a bunch of times
    if hasattr(uni, '_field'): del uni.__dict__['_field']
    uni.field = field
    uni._traits_need_update = True
## -*- coding: utf-8 -*-
## Copyright (c) 2015-2016, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#Numerical Orbital Functions
##############################
#Building discrete molecular orbitals (for visualization) requires a complex
#set of operations that are provided by this module and wrapped into a clean API.
#"""
## Established
##import sympy as sy
#import numpy as np
#import pandas as pd
#import numexpr as ne
#from datetime import datetime
#from numba import jit#, vectorize
#from psutil import virtual_memory
#from collections import OrderedDict
#
## Local
##from exa import Series
##from exatomic._config import config
#from exatomic.field import AtomicField
#from exa.relational.isotope import symbol_to_z
#from exatomic.algorithms.basis import solid_harmonics, clean_sh
#
#symbol_to_z = symbol_to_z()
#halfmem = virtual_memory().total / 2
#solhar = clean_sh(solid_harmonics(6))
#
######################################################################
## Numba vectorized operations for Orbital, MOMatrix, Density tables #
## These will eventually be fully moved to matrices.py not meshgrid3d#
######################################################################
#
#@jit(nopython=True)
#def density_from_momatrix(cmat, occvec):
#    nbas = len(occvec)
#    arlen = nbas * (nbas + 1) // 2
#    dens = np.empty(arlen, dtype=np.float64)
#    chi1 = np.empty(arlen, dtype=np.int64)
#    chi2 = np.empty(arlen, dtype=np.int64)
#    frame = np.empty(arlen, dtype=np.int64)
#    cnt = 0
#    for i in range(nbas):
#        for j in range(i + 1):
#            dens[cnt] = (cmat[i,:] * cmat[j,:] * occvec).sum()
#            chi1[cnt] = i
#            chi2[cnt] = j
#            frame[cnt] = 0
#            cnt += 1
#    return chi1, chi2, dens, frame
#
#@jit(nopython=True)
#def density_as_square(denvec):
#    nbas = int((-1 + np.sqrt(1 - 4 * -2 * len(denvec))) / 2)
#    square = np.empty((nbas, nbas), dtype=np.float64)
#    cnt = 0
#    for i in range(nbas):
#        for j in range(i + 1):
#            square[i, j] = denvec[cnt]
#            square[j, i] = denvec[cnt]
#            cnt += 1
#    return square
#
#@jit(nopython=True)
#def momatrix_as_square(movec):
#    nbas = np.int64(len(movec) ** (1/2))
#    square = np.empty((nbas, nbas), dtype=np.float64)
#    cnt = 0
#    for i in range(nbas):
#        for j in range(nbas):
#            square[j, i] = movec[cnt]
#            cnt += 1
#    return square
#
#@jit(nopython=True, cache=True, nogil=True)
#def meshgrid3d(x, y, z):
#    tot = len(x) * len(y) * len(z)
#    xs = np.empty(tot, dtype=np.float64)
#    ys = np.empty(tot, dtype=np.float64)
#    zs = np.empty(tot, dtype=np.float64)
#    cnt = 0
#    for i in x:
#        for j in y:
#            for k in z:
#                xs[cnt] = i
#                ys[cnt] = j
#                zs[cnt] = k
#                cnt += 1
#    return xs, ys, zs
#
#################################################################
## Functions used in the generation of basis functions and MOs. #
#################################################################
#
#def make_fps(rmin=None, rmax=None, nr=None, nrfps=1,
#             xmin=None, xmax=None, nx=None, frame=0,
#             ymin=None, ymax=None, ny=None, field_type=0,
#             zmin=None, zmax=None, nz=None, label=0,
#             ox=None, dxi=None, dxj=None, dxk=None,
#             oy=None, dyi=None, dyj=None, dyk=None,
#             oz=None, dzi=None, dzj=None, dzk=None):
#    """
#    Generate the necessary field parameters of a numerical grid field
#    as an exatomic.field.AtomicField.
#
#    Args
#        nrfps (int): number of field parameters with same dimensions
#        rmin (float): minimum value in an arbitrary cartesian direction
#        rmax (float): maximum value in an arbitrary cartesian direction
#        nr (int): number of grid points between rmin and rmax
#        xmin (float): minimum in x direction (optional)
#        xmax (float): maximum in x direction (optional)
#        ymin (float): minimum in y direction (optional)
#        ymax (float): maximum in y direction (optional)
#        zmin (float): minimum in z direction (optional)
#        zmax (float): maximum in z direction (optional)
#        nx (int): steps in x direction (optional)
#        ny (int): steps in y direction (optional)
#        nz (int): steps in z direction (optional)
#        ox (float): origin in x direction (optional)
#        oy (float): origin in y direction (optional)
#        oz (float): origin in z direction (optional)
#        dxi (float): x-component of x-vector specifying a voxel
#        dxj (float): y-component of x-vector specifying a voxel
#        dxk (float): z-component of x-vector specifying a voxel
#        dyi (float): x-component of y-vector specifying a voxel
#        dyj (float): y-component of y-vector specifying a voxel
#        dyk (float): z-component of y-vector specifying a voxel
#        dzi (float): x-component of z-vector specifying a voxel
#        dzj (float): y-component of z-vector specifying a voxel
#        dzk (float): z-component of z-vector specifying a voxel
#        label (str): an identifier passed to the widget (optional)
#        field_type (str): alternative identifier (optional)
#
#    Returns
#        fps (pd.Series): field parameters
#    """
#    if any((par is None for par in [rmin, rmax, nr])):
#        if all((par is None for par in (ox, dxi, dxj, dxk))):
#            raise Exception("Must supply at least rmin, rmax, nr or field"
#                            " parameters as specified by a cube file.")
#    d = {}
#    allcarts = [['x', 0, xmin, xmax, nx, ox, (dxi, dxj, dxk)],
#                ['y', 1, ymin, ymax, ny, oy, (dyi, dyj, dyk)],
#                ['z', 2, zmin, zmax, nz, oz, (dzi, dzj, dzk)]]
#    for akey, aix, amin, amax, na, oa, da in allcarts:
#        if oa is None:
#            amin = rmin if amin is None else amin
#            amax = rmax if amax is None else amax
#            na = nr if na is None else na
#        else: amin = oa
#        dw = [0, 0, 0]
#        if all(i is None for i in da): dw[aix] = (amax - amin) / na
#        else: dw = da
#        d[akey] = [amin, na, dw]
#    fp = pd.Series({
#        'dxi': d['x'][2][0], 'dyj': d['y'][2][1], 'dzk': d['z'][2][2],
#        'dxj': d['x'][2][1], 'dyk': d['y'][2][2], 'dzi': d['z'][2][0],
#        'dxk': d['x'][2][2], 'dyi': d['y'][2][0], 'dzj': d['z'][2][1],
#        'ox': d['x'][0], 'oy': d['y'][0], 'oz': d['z'][0], 'frame': frame,
#        'nx': d['x'][1], 'ny': d['y'][1], 'nz': d['z'][1], 'label': label,
#        'field_type': field_type
#        })
#    return pd.concat([fp] * nrfps, axis=1).T
#
#
#def _sphr_prefac(nuc, sh, L, ml):
#    """
#    Create strings of the pre-exponential factor of a given
#    spherical basis function as a function of l, ml quantum numbers.
#
#    Args
#        L (int): angular momentum quantum numbers
#        ml (int): magnetic quantum number
#        nuc (dict): atomic position
#        sh (dict): cleaned solid harmonics
#
#    Returns
#        prefacs (list): pre-exponential factors
#    """
#
#    return [pre.format(**nuc) for pre in sh[(L, ml)]]
#
#
#def _cart_prefac(nuc, pre, L, l, m, n):
#    """
#    As with _sphr_prefac, create the string version of the pre-exponential
#    factor in a given basis function, this time as a function of cartesian
#    powers (l, m, n) instead of (l, ml) quantum numbers.
#
#    Args
#        L (int): angular momentum quantum number
#        l (int): powers of x
#        m (int): powers of y
#        n (int): powers of z
#        nuc (dict): atomic position
#        pre (str): '' unless ADF
#
#    Returns
#        prefacs (list): pre-exponential factors
#    """
#    if not L: return [pre]
#    if pre and not pre.endswith('*'): pre = '{}*'.format(pre)
#    lin, nlin = '{}*', '{}**{}*'
#    for cart, powr in OrderedDict([('{x}', l),
#                                   ('{y}', m),
#                                   ('{z}', n)]).items():
#        if not powr: continue
#        stargs = [cart]
#        fmt = lin if powr == 1 else nlin
#        if powr > 1: stargs.append(powr)
#        pre += fmt.format(*stargs)
#    return [pre.format(**nuc)]
#
#def gen_basfn(prefacs, shell, rexp, precision=10):
#    """
#    Given a list of pre-exponential factors and a shell of
#    primitive functions (slice of basis set table), return
#    the string that is the basis function written out as it
#    would be on paper.
#
#    Args
#        prefacs (list): string of pre-exponential factors
#        shell (exatomic.basis.BasisSet): a shell of the basis set table
#        rexp (str): the r-dependent exponent (including atomic position)
#
#    Returns
#        basis function (str)
#    """
#    bastr = '{prefac}({prims})'
#    bastrs = []
#    primitive = '{{Nd:.{precision}f}}*exp' \
#                '(-{{alpha:.{precision}f}}*{rexp})'.format(rexp=rexp,
#                                                           precision=precision)
#    for prefac in prefacs:
#        primitives = shell.apply(lambda x: primitive.format(**x), axis=1)
#        bastrs.append(bastr.format(prefac=prefac,
#                                   prims='+'.join(primitives.values)))
#    return '+'.join(bastrs)
#
#
#def gen_basfns(uni, frame=None):
#    """
#    Given an exatomic.container.Universe that contains complete momatrix
#    and basis_set attributes, generates and returns the strings corresponding
#    to how the basis functions would be written out with paper and pencil.
#    This is mainly for debugging and testing generality to deal with different
#    computational codes' basis function ordering schemes.
#
#    Args
#        uni (exatomic.container.Universe): must contain momatrix and basis_set
#
#    Returns
#        bastrs (list): list of strings of basis functions
#    """
#    frame = uni.atom.nframes - 1 if frame is None else frame
#    # Group the dataframes appropriately
#    sets = uni.basis_set.cardinal_groupby().get_group(frame).groupby('set')
#    funcs = uni.basis_set_order.cardinal_groupby().get_group(frame).groupby('center')
#    atom = uni.atom.cardinal_groupby().get_group(frame)
#    # Set some variables based on basis set info
#    larg = {'sh': None, 'pre': None}
#    if uni.basis_set.spherical:
#        # Basis set order columns for spherical functions
#        ordrcols = ['L', 'ml', 'shell']
#        prefunc = _sphr_prefac
#        # To avoid bool checking for each basis function
#        lkey = 'sh'
#        # Get the string versions of the symbolic solid harmonics
#        larg[lkey] = solhar
#    else:
#        # Basis set order columns for cartesian functions
#        ordrcols = ['L', 'l', 'm', 'n', 'shell']
#        prefunc = _cart_prefac
#        # To avoid bool checking for each basis function
#        lkey = 'pre'
#        # Placeholder for potential custom prefactors from ADF
#        larg[lkey] = ''
#    # The number of arguments to pass to _prefac
#    rgslice = slice(0, len(ordrcols) - 1)
#    # In the case of ADF orbitals (currently only one that
#    # requires 'r' and 'prefac' on a per basis function basis
#    if not uni.basis_set.gaussian:
#        ordrcols = ordrcols[:-1] + ['r', 'prefac'] + ordrcols[-1:]
#        exkey = 'r'
#    else: exkey = 'r2'
#    # Iterate over atomic positions
#    basfns = []
#    for i, (seht, x, y, z) in enumerate(zip(atom['set'], atom['x'],
#                                            atom['y'], atom['z'])):
#        # Dict of strings of atomic position
#        nuc = _atompos(x, y, z)
#        # Regroup dataframes
#        bas = sets.get_group(seht).groupby('L')
#        ordr = funcs.get_group(i)
#        # Iterate over atom centered basis functions
#        for args in zip(*[ordr[col] for col in ordrcols]):
#            # Get the shell of primitive functions
#            shell = bas.get_group(args[0]).groupby('shell').get_group(args[-1])
#            try:
#                # Only used for ADF orbitals currently
#                preexp, prefac = args[4:6]
#                preexp = '' if not preexp else '({})**{}*'.format(nuc[exkey], preexp)
#                prefac = '' if not prefac else '{}*'.format(prefac)
#                larg['pre'] = '{}{}'.format(preexp, prefac)
#            except ValueError:
#                # Otherwise it's useless
#                larg['pre'] = ''
#            # The pre-exponential factors (besides the additional ADF ones)
#            prefacs = prefunc(nuc, larg[lkey], *args[rgslice])
#            # Put together the basis function
#            print(prefacs, shell, nuc[exkey])
#            basfns.append(gen_basfn(prefacs, shell, nuc[exkey]))
#    return basfns
#
#
#def _atompos(x, y, z, precision=10):
#    nuc = {}
#    p = '{{}}{{}}{{:.{}f}}'.format(precision).format
#    for key, cart in [('x', x), ('y', y), ('z', z)]:
#        if np.isclose(cart, 0):
#            nuc[key] = key
#        else:
#            cart, op = (cart, '-') if cart > 0 else (np.abs(cart), '+')
#            nuc[key] = '({})'.format(p(key, op, cart).strip('0'))
#    nuc['r2'] = '({}**2+{}**2+{}**2)'.format(nuc['x'], nuc['y'], nuc['z'])
#    nuc['r'] = '{}**0.5'.format(nuc['r2'])
#    return nuc
#
#
#def numerical_grid_from_field_params(fld_ps):
#    if isinstance(fld_ps, pd.DataFrame):
#        fld_ps = fld_ps.ix[fld_ps.index.values[0]]
#    ox, nx, dx = fld_ps.ox, fld_ps.nx, fld_ps.dxi
#    oy, ny, dy = fld_ps.oy, fld_ps.ny, fld_ps.dyj
#    oz, nz, dz = fld_ps.oz, fld_ps.nz, fld_ps.dzk
#    mx = ox + (nx - 1) * dx
#    my = oy + (ny - 1) * dy
#    mz = oz + (nz - 1) * dz
#    x = np.linspace(ox, mx, nx)
#    y = np.linspace(oy, my, ny)
#    z = np.linspace(oz, mz, nz)
#    return meshgrid3d(x, y, z)
#
#
#def _determine_field_params(universe, field_params, nvec):
#    if field_params is None:
#        dr = 41
#        rmin = min(universe.atom['x'].min(),
#                   universe.atom['y'].min(),
#                   universe.atom['z'].min()) - 4
#        rmax = max(universe.atom['x'].max(),
#                   universe.atom['y'].max(),
#                   universe.atom['z'].max()) + 4
#        return make_fps(rmin, rmax, dr, nrfps=nvec)
#    else:
#        return make_fps(nrfps=nvec, **field_params)
#
#
#def _determine_vector(uni, vector):
#    if isinstance(vector, int): return [vector]
#    typs = (list, tuple, range, np.ndarray)
#    if isinstance(vector, typs): return vector
#    if vector is None:
#        if hasattr(uni, 'orbital'):
#            homo = uni.orbital.get_orbital().vector
#        elif hasattr(uni.frame, 'N_e'):
#            homo = uni.frame['N_e'].values[0]
#        elif hasattr(uni.atom, 'Zeff'):
#            homo = uni.atom['Zeff'].sum() // 2
#        elif hasattr(uni.atom, 'Z'):
#            homo = uni.atom['Z'].sum() // 2
#        else:
#            uni.atom['Z'] = uni.atom['symbol'].map(symbol_to_z)
#            homo = uni.atom['Z'].sum() // 2
#        if homo < 15: return range(0, homo + 15)
#        else: return range(homo - 15, homo + 5)
#    else: raise TypeError('Try specifying vector as a list or int')
#
#
#def add_molecular_orbitals(uni, field_params=None, mocoefs=None,
#                           vector=None, frame=None, inplace=True):
#    """
#    If a universe contains enough information to generate
#    molecular orbitals (basis_set, basis_set_summary and momatrix),
#    evaluate the molecular orbitals on a discretized grid. If vector
#    is not provided, attempts to calculate vectors by the sum of Z/Zeff
#    of the atoms present divided by two; roughly (HOMO-15,LUMO+5).
#
#    Args
#        uni (exatomic.container.Universe): a universe
#        field_params (dict,pd.Series): dict with {'rmin', 'rmax', 'nr'}
#        mocoefs (str): column in momatrix (default 'coef')
#        vector (int, list, range, np.array): the MO vectors to evaluate
#        inplace (bool): if False, return the field obj instead of modifying uni
#
#    Warning:
#       If inplace is True, removes any fields previously attached to the universe
#    """
#    # Preliminary assignment and array dimensions
#    vector = _determine_vector(uni, vector)
#    if mocoefs is None: mocoefs = 'coef'
#    elif mocoefs not in uni.momatrix.columns:
#        print('mocoefs {} is not in uni.momatrix'.format(mocoefs))
#        return
#    fld_ps = _determine_field_params(uni, field_params, len(vector))
#    x, y, z = numerical_grid_from_field_params(fld_ps)
#    nbas = len(uni.basis_set_order.index)
#    norb = uni.momatrix.orbital.max()
#    nvec = len(vector)
#    npts = len(x)
#
#    # Build the strings corresponding to basis functions
#    print('Warning: not extensively tested. Please be careful.')
#    basfns = gen_basfns(uni, frame=frame)
#    orbs = uni.momatrix.groupby('orbital')
#
#    # Evaluate basis functions one time and store all in a single
#    # large numpy array which is much more efficient but can require
#    # a lot of memory if the resolution of the field is very fine
#    if (norb * npts * 8) < halfmem:
#        t1 = datetime.now()
#        print('Evaluating basis functions once.')
#        fields = np.empty((npts, nvec), dtype=np.float64)
#        vals = np.empty((npts, nbas), dtype=np.float64)
#        for i, bas in enumerate(basfns): vals[:,i] = ne.evaluate(bas)
#        for i, vec in enumerate(vector):
#            fields[:,i] = (vals * orbs.get_group(vec).coef.values).sum(axis=1)
#        t2 = datetime.now()
#        print('Timing: compute MOs - {:.2f}s'.format((t2-t1).total_seconds()))
#    # If the resolution of fields will be memory intensive, evaluate
#    # each basis function on the fly per MO which saves a large
#    # np.array in memory but is redundant and less efficient
#    else:
#        t1 = datetime.now()
#        print('Evaluating basis functions per MO.')
#        fields = np.zeros((npts, nvec), dtype=np.float64)
#        for i, vec in enumerate(vector):
#            c = orbs.get_group(vec).coef.values
#            for j, bas in enumerate(basfns):
#                fields[:,i] += c[j] * ne.evaluate(bas)
#        t2 = datetime.now()
#        print('Timing: compute MOs - {:.2f}s'.format((t2-t1).total_seconds()))
#
#    field = AtomicField(fld_ps, field_values=[fields[:,i] for i in range(nvec)])
#    if not inplace: return field
#    # Don't collect infinity fields if this is run a bunch of times
#    if hasattr(uni, '_field'): del uni.__dict__['_field']
#    uni.field = field
#    uni._traits_need_update = True
### -*- coding: utf-8 -*-
### Copyright (c) 2015-2016, Exa Analytics Development Team
### Distributed under the terms of the Apache License 2.0
##"""
##Numerical Orbital Functions
###############################
##Building discrete molecular orbitals (for visualization) requires a complex
##set of operations that are provided by this module and wrapped into a clean API.
##"""
### Established
###import sympy as sy
##import numpy as np
##import pandas as pd
###import numexpr as ne
##from datetime import datetime
###from numba import jit, vectorize
##from psutil import virtual_memory
##from collections import OrderedDict
##
#### For voluminating GTFs
###import pandas as pd
##from sympy import Add, Mul
##from exatomic.algorithms.basis import solid_harmonics
###from exatomic.field import AtomicField
##from exa.relational.isotope import symbol_to_z
##from exatomic.algorithms.basis import clean_sh
##
##symbol_to_z = symbol_to_z()
##halfmem = virtual_memory().total / 2
##solhar = clean_sh(solid_harmonics(6))
##
#######################################################################
### Numba vectorized operations for Orbital, MOMatrix, Density tables #
### These will eventually be fully moved to matrices.py not meshgrid3d#
#######################################################################
##
###@jit(nopython=True)
##def density_from_momatrix(cmat, occvec):
##    print("orbital.density_from_matrix")
##    nbas = len(occvec)
##    arlen = nbas * (nbas + 1) // 2
##    dens = np.empty(arlen, dtype=np.float64)
##    chi1 = np.empty(arlen, dtype=np.int64)
##    chi2 = np.empty(arlen, dtype=np.int64)
##    frame = np.empty(arlen, dtype=np.int64)
##    cnt = 0
##    for i in range(nbas):
##        for j in range(i + 1):
##            dens[cnt] = (cmat[i,:] * cmat[j,:] * occvec).sum()
##            chi1[cnt] = i
##            chi2[cnt] = j
##            frame[cnt] = 0
##            cnt += 1
##    return chi1, chi2, dens, frame
##
###@jit(nopython=True)
##def density_as_square(denvec):
##    print("orbital.density_as_square")
##    nbas = int((-1 + np.sqrt(1 - 4 * -2 * len(denvec))) / 2)
##    square = np.empty((nbas, nbas), dtype=np.float64)
##    cnt = 0
##    for i in range(nbas):
##        for j in range(i + 1):
##            square[i, j] = denvec[cnt]
##            square[j, i] = denvec[cnt]
##            cnt += 1
##    return square
##
###@jit(nopython=True)
##def momatrix_as_square(movec):
##    print("orbital.momatrix_as_square")
##    nbas = np.int64(len(movec) ** (1/2))
##    square = np.empty((nbas, nbas), dtype=np.float64)
##    cnt = 0
##    for i in range(nbas):
##        for j in range(nbas):
##            square[j, i] = movec[cnt]
##            cnt += 1
##    return square
##
###@jit(nopython=True, cache=True, nogil=True)
##def meshgrid3d(x, y, z):
##    print("orbital.meshgrid3d")
##    tot = len(x) * len(y) * len(z)
##    xs = np.empty(tot, dtype=np.float64)
##    ys = np.empty(tot, dtype=np.float64)
##    zs = np.empty(tot, dtype=np.float64)
##    cnt = 0
##    for i in x:
##        for j in y:
##            for k in z:
##                xs[cnt] = i
##                ys[cnt] = j
##                zs[cnt] = k
##                cnt += 1
##    return xs, ys, zs
##
##################################################################
### Functions used in the generation of basis functions and MOs. #
##################################################################
##
##class Nucpos(object):
##
##    def __str__(self):
##        return 'Nucpos({},{},{},{})'.format(self.x, self.y, self.z, self.pre)
##
##    def __init__(self, x, y, z, pre=None, prefac=None):
##        print("orbital.Nucpos")
##        self.x  = '(x-{:.10f})'.format(x) if not np.isclose(x, 0) else 'x'
##        self.y  = '(y-{:.10f})'.format(y) if not np.isclose(y, 0) else 'y'
##        self.z  = '(z-{:.10f})'.format(z) if not np.isclose(z, 0) else 'z'
##        self.r2 = '({}**2+{}**2+{}**2)'.format(self.x, self.y, self.z)
##        self.r = self.r2 + '**0.5'
##        self.pre = '' if (pre is None or not pre) else '({})**{}*'.format(self.r, pre)
##        self.prefac = '' if (prefac is None or not prefac) else '{}*'.format(prefac)
##
##def make_fps(rmin=None, rmax=None, nr=None, nrfps=1,
##             xmin=None, xmax=None, nx=None, frame=0,
##             ymin=None, ymax=None, ny=None, field_type=0,
##             zmin=None, zmax=None, nz=None, label=0,
##             ox=None, dxi=None, dxj=None, dxk=None,
##             oy=None, dyi=None, dyj=None, dyk=None,
##             oz=None, dzi=None, dzj=None, dzk=None):
##    """
##    Generate the necessary field parameters of a numerical grid field
##    as an exatomic.field.AtomicField.
##
##    Args
##        nrfps (int): number of field parameters with same dimensions
##        rmin (float): minimum value in an arbitrary cartesian direction
##        rmax (float): maximum value in an arbitrary cartesian direction
##        nr (int): number of grid points between rmin and rmax
##        xmin (float): minimum in x direction (optional)
##        xmax (float): maximum in x direction (optional)
##        ymin (float): minimum in y direction (optional)
##        ymax (float): maximum in y direction (optional)
##        zmin (float): minimum in z direction (optional)
##        zmax (float): maximum in z direction (optional)
##        nx (int): steps in x direction (optional)
##        ny (int): steps in y direction (optional)
##        nz (int): steps in z direction (optional)
##        ox (float): origin in x direction (optional)
##        oy (float): origin in y direction (optional)
##        oz (float): origin in z direction (optional)
##        dxi (float): x-component of x-vector specifying a voxel
##        dxj (float): y-component of x-vector specifying a voxel
##        dxk (float): z-component of x-vector specifying a voxel
##        dyi (float): x-component of y-vector specifying a voxel
##        dyj (float): y-component of y-vector specifying a voxel
##        dyk (float): z-component of y-vector specifying a voxel
##        dzi (float): x-component of z-vector specifying a voxel
##        dzj (float): y-component of z-vector specifying a voxel
##        dzk (float): z-component of z-vector specifying a voxel
##        label (str): an identifier passed to the widget (optional)
##        field_type (str): alternative identifier (optional)
##
##    Returns
##        fps (pd.Series): field parameters
##    """
##    print("orbital.make_fps")
##    if any((par is None for par in [rmin, rmax, nr])):
##        if all((par is None for par in (ox, dxi, dxj, dxk))):
##            raise Exception("Must supply at least rmin, rmax, nr or field"
##                            " parameters as specified by a cube file.")
##    d = {}
##    allcarts = [['x', 0, xmin, xmax, nx, ox, (dxi, dxj, dxk)],
##                ['y', 1, ymin, ymax, ny, oy, (dyi, dyj, dyk)],
##                ['z', 2, zmin, zmax, nz, oz, (dzi, dzj, dzk)]]
##    for akey, aix, amin, amax, na, oa, da in allcarts:
##        if oa is None:
##            amin = rmin if amin is None else amin
##            amax = rmax if amax is None else amax
##            na = nr if na is None else na
##        else: amin = oa
##        dz = [0, 0, 0]
##        if all(i is None for i in da): dz[aix] = (amax - amin) / na
##        else: dz = da
##        d[akey] = [amin, na, dz]
##    fp = pd.Series({'ox': d['x'][0],    'oy': d['y'][0],      'oz': d['z'][0],
##                    'nx': d['x'][1],    'ny': d['y'][1],      'nz': d['z'][1],
##                   'dxi': d['x'][2][0], 'dyj': d['y'][2][1], 'dzk': d['z'][2][2],
##                   'dxj': d['x'][2][1], 'dyk': d['y'][2][2], 'dzi': d['z'][2][0],
##                   'dxk': d['x'][2][2], 'dyi': d['y'][2][0], 'dzj': d['z'][2][1],
##                   'frame': frame})
##    fp = pd.concat([fp] * nrfps, axis=1).T
##    if isinstance(label, list) and len(label) == nrfps:
##        fp['label'] = label
##    else:
##        fp['label'] = [label] * nrfps
##    if isinstance(field_type, list) and len(field_type) == nrfps:
##        fp['field_type'] = field_type
##    else:
##        fp['field_type'] = [field_type] * nrfps
##    return fp
##
##
##def clean_sh(sh):
##    """
##    Takes symbolic representation of solid harmonic functions
##    and cleans them to lists of strings to be used in generating
##    string forms of basis functions.
##
##    Args
##        sh (OrderedDict): the output of exatomic.algorithms.basis.solid_harmonics
##
##    Returns
##        clean (OrderedDict): cleaned for use in generating string basis functions
##    """
##    print("orbital.clean_sh")
##    clean = OrderedDict()
##    mlp = {1: 'nuc.x', -1: 'nuc.y', 0: 'nuc.z'}
##    for L, ml in sh:
##        if not L:
##            clean[(L, ml)] = ['']
##            continue
##        if L == 1:
##            clean[(L, ml)] = ['{{{}}}*'.format(mlp[ml])]
##            continue
##        symbolic = sh[(L, ml)].expand(basic=True)
##        if type(symbolic) == Mul:
##            coef, sym = symbolic.as_coeff_Mul()
##            coef = '{:.10f}*'.format(coef)
##            sym = str(sym).replace('*', '')
##            sym = '*'.join(['{{nuc.{}}}'.format(char) for char in sym])
##            clean[(L, ml)] = [''.join([coef, sym, '*'])]
##            continue
##        elif type(symbolic) == Add:
##            clean[(L, ml)] = []
##            dat = list(symbolic.as_coefficients_dict().items())
##            for sym, coef in dat:
##                coef ='{:.10f}*'.format(coef)
##                powers = {'x': 0, 'y': 0, 'z': 0}
##                sym = str(sym).replace('*', '')
##                for i, j in zip(sym, sym[1:]):
##                    if j.isnumeric(): powers[i] = int(j)
##                    elif i.isalpha(): powers[i] = 1
##                if sym[-1].isalpha(): powers[sym[-1]] = 1
##                lin = '{{nuc.{}}}*'
##                nlin = '{{nuc.{}}}**{}*'
##                for cart, power in powers.items():
##                    if power:
##                        stargs = [cart]
##                        fmt = lin if power == 1 else nlin
##                        if power > 1: stargs.append(power)
##                        coef += fmt.format(*stargs)
##                clean[(L, ml)].append(coef)
##    return clean
##
##
##def _sphr_prefac(clean, L, ml, nuc):
##    """
##    Create strings of the pre-exponential factor of a given
##    spherical basis function as a function of l, ml quantum numbers.
##
##    Args
##        L (int): angular momentum quantum numbers
##        ml (int): magnetic quantum number
##        nuc (dict): atomic position
##        sh (dict): cleaned solid harmonics
##
##    Returns
##        prefacs (list): pre-exponential factors
##    """
##    print("orbital._sphr_prefac")
##    cln = clean[(L, ml)]
##    if any(('nuc' in pre for pre in cln)):
##        cln = [pre.format(nuc=nuc) for pre in cln]
##    return cln
##    #return [pre.format(**nuc) for pre in sh[(L, ml)]]
##
##
##def _cart_prefac(nuc, pre, L, l, m, n):
##    """
##    As with _sphr_prefac, create the string version of the pre-exponential
##    factor in a given basis function, this time as a function of cartesian
##    powers (l, m, n) instead of (l, ml) quantum numbers.
##
##    Args
##        L (int): angular momentum quantum number
##        l (int): powers of x
##        m (int): powers of y
##        n (int): powers of z
##        nuc (dict): atomic position
##        pre (str): '' unless ADF
##
##    Returns
##        prefacs (list): pre-exponential factors
##    """
##    print("orbital._cart_prefac")
##    #prefac = '{}{}'.format(nucpos.pre, nucpos.prefac)
##    #if L == 0: return [prefac]
##    if not L: return [pre]
##    if pre and not pre.endswith('*'): pre = '{}*'.format(pre)
##    lin, nlin = '{}*', '{}**{}*'
##    for cart, powr in OrderedDict([('{x}', l),
##                                   ('{y}', m),
##                                   ('{z}', n)]).items():
##        if not powr: continue
##        stargs = [cart]
##        fmt = lin if powr == 1 else nlin
##        if powr > 1: stargs.append(powr)
##        pre += fmt.format(*stargs)
##    return [pre.format(**nuc)]
##
##def gen_basfn(prefacs, shell, rexp, precision=10):
##    """
##    Given a list of pre-exponential factors and a shell of
##    primitive functions (slice of basis set table), return
##    the string that is the basis function written out as it
##    would be on paper.
##
##    Args
##        prefacs (list): string of pre-exponential factors
##        shell (exatomic.basis.BasisSet): a shell of the basis set table
##        rexp (str): the r-dependent exponent (including atomic position)
##
##    Returns
##        basis function (str)
##    """
##    print("orbital.gen_basfn")
##    bastr = '{prefac}({prims})'
##    bastrs = []
##    primitive = '{{Nd:.{precision}f}}*exp' \
##                '(-{{alpha:.{precision}f}}*{rexp})'.format(rexp=rexp,
##                                                           precision=precision)
##    for prefac in prefacs:
##        primitives = shell.apply(lambda x: primitive.format(**x), axis=1)
##        bastrs.append(bastr.format(prefac=prefac,
##                                   prims='+'.join(primitives.values)))
##    return '+'.join(bastrs)
##
##def gen_basfns(universe, frame=None):
##    """
##    Given an exatomic.container.Universe that contains complete momatrix
##    and basis_set attributes, generates and returns the strings corresponding
##    to how the basis functions would be written out with paper and pencil.
##    This is mainly for debugging and testing generality to deal with different
##    computational codes' basis function ordering schemes.
##
##    Args
##        uni (exatomic.container.Universe): must contain momatrix and basis_set
##
##    Returns
##        bastrs (list): list of strings of basis functions
##    """
##    print("orbital.gen_basfns")
##    # Get the symbolic spherical functions
##    lmax = universe.basis_set.lmax
##    sh = clean_sh(solid_harmonics(lmax))
##    #frame = uni.atom.nframes - 1 if frame is None else frame
##    # Group the dataframes appropriately
##    #sets = uni.basis_set.cardinal_groupby().get_group(frame).groupby('set')
##    #funcs = uni.basis_set_order.cardinal_groupby().get_group(frame).groupby('center')
##    #atom = uni.atom.cardinal_groupby().get_group(frame)
##    # Set some variables based on basis set info
##    #larg = {'sh': None, 'pre': None}
##    #if uni.basis_set.spherical:
##        # Basis set order columns for spherical functions
##    #    ordrcols = ['L', 'ml', 'shell']
##    #    prefunc = _sphr_prefac
##        # To avoid bool checking for each basis function
##    #    lkey = 'sh'
##        # Get the string versions of the symbolic solid harmonics
##    #    larg[lkey] = solhar
##    #else:
##        # Basis set order columns for cartesian functions
##    #    ordrcols = ['L', 'l', 'm', 'n', 'shell']
##    #    prefunc = _cart_prefac
##        # To avoid bool checking for each basis function
##    #    lkey = 'pre'
##        # Placeholder for potential custom prefactors from ADF
##    #    larg[lkey] = ''
##    # The number of arguments to pass to _prefac
##    #rgslice = slice(0, len(ordrcols) - 1)
##    # In the case of ADF orbitals (currently only one that
##    # requires 'r' and 'prefac' on a per basis function basis
##    #if not uni.basis_set.gaussian:
##    #    ordrcols = ordrcols[:-1] + ['r', 'prefac'] + ordrcols[-1:]
##    #    exkey = 'r'
##    #else: exkey = 'r2'
##    # Iterate over atomic positions
##    basfns = []
##    for i, (seht, x, y, z) in enumerate(zip(atom['set'], atom['x'],
##                                            atom['y'], atom['z'])):
##        # Dict of strings of atomic position
##        nuc = _atompos(x, y, z)
##        # Regroup dataframes
##        bas = bases.get_group(seht).groupby('L')
##        basord = centers.get_group(i)
##        if universe.basis_set.spherical:
##            # Iterate over spherical atom-centered basis functions
##            for L, ml, shfunc in zip(basord['L'], basord['ml'], basord['shell']):
##                grp = bas.get_group(L).groupby('shell').get_group(shfunc)
##                prefacs = _sphr_prefac(sh, L, ml, nucpos)
##                basfns.append(gen_basfn(prefacs, grp, nucpos.r2))
##        else:
##            # Iterate over cartesian atom-centered basis functions
##            args = ['L', 'l', 'm', 'n', 'shell']
##            #kwargs = {'pre': None, 'prefac': None}
##            if 'pre' in basord.columns:
##                args.append('pre')
##                if 'prefac' in basord.columns: args.append('prefac')
##            for row in zip(*[basord[i] for i in args]):
##                L, l, m, n, shfunc = row[:5]
##                pre, prefac = row[5:6], row[6:7]
##                nucpos = Nucpos(x, y, z, pre=pre, prefac=prefac)
##                grp = bas.get_group(L).groupby('shell').get_group(shfunc)
##                prefacs = _cart_prefac(L, l, m, n, nucpos)
##                basfns.append(gen_basfn(prefacs, grp, nucpos.r))
##    return basfns
##
##def numerical_grid_from_field_params(field_params):
##    print("orbital.numerical_grid_from_field_params")
##    mx = field_params.ox[0] + (field_params.nx[0] - 1) * field_params.dxi[0]
##    my = field_params.oy[0] + (field_params.ny[0] - 1) * field_params.dyj[0]
##    mz = field_params.oz[0] + (field_params.nz[0] - 1) * field_params.dzk[0]
##    x = np.linspace(field_params.ox[0], mx, field_params.nx[0])
##    y = np.linspace(field_params.oy[0], my, field_params.ny[0])
##    z = np.linspace(field_params.oz[0], mz, field_params.nz[0])
##    return meshgrid3d(x, y, z)
##
##
##def _determine_field_params(universe, field_params, nvec):
##    print("orbital._determine_field_params")
##    if field_params is None:
##        dr = 41
##        rmin = min(universe.atom['x'].min(),
##                   universe.atom['y'].min(),
##                   universe.atom['z'].min()) - 4
##        rmax = max(universe.atom['x'].max(),
##                   universe.atom['y'].max(),
##                   universe.atom['z'].max()) + 4
##        return make_fps(rmin, rmax, dr, nrfps=nvec)
##    else:
##        return make_fps(nrfps=nvec, **field_params)
##
##def _determine_vectors(universe, vector):
##    print("orbital._determine_vectors")
##    if isinstance(vector, int): return [vector]
##    typs = (list, tuple, range, np.ndarray)
##    if isinstance(vector, typs): return vector
##    if vector is None:
##        if hasattr(uni, 'orbital'):
##            homo = uni.orbital.get_orbital().vector
##        elif hasattr(uni.frame, 'N_e'):
##            homo = uni.frame['N_e'].values[0]
##        elif hasattr(uni.atom, 'Zeff'):
##            homo = uni.atom['Zeff'].sum() // 2
##        elif hasattr(uni.atom, 'Z'):
##            homo = uni.atom['Z'].sum() // 2
##        else:
##            return range(homo - 15, homo + 5)
##    else:
##        raise TypeError('Try specifying vector as a list or int')
##
##def _determine_mocoefs(universe, mocoefs, vector):
##    print("orbital._determine_mocoefs")
##    if mocoefs is None: return 'coef'
##    if mocoefs not in universe.momatrix.columns:
##        raise Exception('mocoefs must be a column in universe.momatrix')
##    return mocoefs
##
##def _evaluate_fields(universe, vector, mocoefs, npoints, nbas, x, y, z):
##    print("orbital._evaluate_fields")
##    bvals = np.zeros((npoints, nbas), dtype=np.float64)
##    fvals = np.zeros((npoints, len(vector)), dtype=np.float64)
##    vectors = universe.momatrix.groupby('orbital')
##    for name, bf in universe.basis_functions.items():
##        bvals[:,int(name[2:])] = bf(x, y, z)
##    for i, vno in enumerate(vector):
##        vec = vectors.get_group(vno)
##        for chi, coef in zip(vec['chi'], vec[mocoefs]):
##            if np.abs(coef) > 1e-10:
##                fvals[:, i] += coef * bvals[:, chi]
##    return fvals
##
###def add_molecular_orbitals(universe, field_params=None, mocoefs=None,
###                           vector=None, frame=None):
###            uni.atom['Z'] = uni.atom['symbol'].map(symbol_to_z)
###            homo = uni.atom['Z'].sum() // 2
###        if homo < 15: return range(0, homo + 15)
###        else: return range(homo - 15, homo + 5)
###    else: raise TypeError('Try specifying vector as a list or int')
##
##
##def add_molecular_orbitals(uni, field_params=None, mocoefs=None,
##                           vector=None, frame=None, inplace=True):
##    """
##    If a universe contains enough information to generate
##    molecular orbitals (basis_set, basis_set_summary and momatrix),
##    evaluate the molecular orbitals on a discretized grid. If vector
##    is not provided, attempts to calculate vectors by the sum of Z/Zeff
##    of the atoms present divided by two; roughly (HOMO-15,LUMO+5).
##
##    Args
##        uni (exatomic.container.Universe): a universe
##        field_params (dict,pd.Series): dict with {'rmin', 'rmax', 'nr'}
##        mocoefs (str): column in momatrix (default 'coef')
##        vector (int, list, range, np.array): the MO vectors to evaluate
##        inplace (bool): if False, return the field obj instead of modifying uni
##
##    Warning:
##       If inplace is True, removes any fields previously attached to the universe
##    """
##    print("orbital.add_molecular_orbitals")
##    if hasattr(universe, '_field'):
##        del universe.__dict__['_field']
##    # Preliminary setup for minimal input parameters
##    vector = _determine_vectors(universe, vector)
##    mocoefs = _determine_mocoefs(universe, mocoefs, vector)
##    frame = universe.atom.nframes + 1 if frame is None else frame
##    field_params = _determine_field_params(universe, field_params, len(vector))
##    # Array dimensions
##    lastatom = universe.atom.last_frame
##    nbas = len(universe.basis_set_order.index)
##    try:
##        nprim = lastatom['set'].map(universe.basis_set.primitives().to_dict()).sum()
##    except:
##        nprim = None
##
##    if not hasattr(universe, 'basis_functions'):
##        print('Warning: not extensively tested. Please be careful.')
##        print('Compiling basis functions, may take a while.')
##        t1 = datetime.now()
##        basfns = gen_basfns(universe)
##        universe.basis_functions = _compile_basfns(basfns)
##        t2 = datetime.now()
##        tt = (t2-t1).total_seconds()
##        print("Compile time : {:.2f}s\n"
##              "Size of basis: {} primitives\n"
##              "             : {} functions".format(tt, nprim, nbas))
##
##    t1 = datetime.now()
##    x, y, z = numerical_grid_from_field_params(field_params)
##    npoints = len(x)
###    # Preliminary assignment and array dimensions
###    vector = _determine_vector(uni, vector)
###    if mocoefs is None: mocoefs = 'coef'
###    elif mocoefs not in uni.momatrix.columns:
###        print('mocoefs {} is not in uni.momatrix'.format(mocoefs))
###        return
###    fld_ps = _determine_field_params(uni, field_params, len(vector))
###    x, y, z = numerical_grid_from_field_params(fld_ps)
###    nbas = len(uni.basis_set_order.index)
###    norb = uni.momatrix.orbital.max()
##    nvec = len(vector)
##    npts = len(x)
##
##    # Build the strings corresponding to basis functions
##    print('Warning: not extensively tested. Please be careful.')
##    basfns = gen_basfns(uni, frame=frame)
##    orbs = uni.momatrix.groupby('orbital')
##
##
##if config['dynamic']['numba'] == 'true':
##    from numba import jit
##    density_from_momatrix = jit(nopython=True)(density_from_momatrix)
##    density_as_square = jit(nopython=True)(density_as_square)
##    momatrix_as_square = jit(nopython=True)(momatrix_as_square)
##    meshgrid3d = jit(nopython=True, cache=True, nogil=True)(meshgrid3d)
##    #compute_mos = jit(nopython=True, cache=True)(compute_mos)
##else:
##    print('add_molecular_orbitals will not work without having numba installed')
