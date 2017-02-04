# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Numerical Orbital Functions
#############################
Building discrete molecular orbitals (for visualization) requires a complex
set of operations that are provided by this module and wrapped into a clean API.
"""
import numpy as np
from exatomic._config import config
from datetime import datetime

## For voluminating GTFs
import re
import pandas as pd
from exa import Series
import sympy as sy
from sympy import Add, Mul
from exatomic.algorithms.basis import solid_harmonics, _vec_normalize
from exatomic.field import AtomicField
from collections import OrderedDict

#####################################################################
# Numba vectorized operations for Orbital, MOMatrix, Density tables #
#####################################################################

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

def momatrix_as_square(movec):
    nbas = np.int64(len(movec) ** (1/2))
    square = np.empty((nbas, nbas), dtype=np.float64)
    cnt = 0
    for i in range(nbas):
        for j in range(nbas):
            square[j, i] = movec[cnt]
            cnt += 1
    return square

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

class Nucpos(object):

    def __str__(self):
        return 'Nucpos({:.3f},{:.3f},{:.3f})'.format(self.x, self.y, self.z)

    def __init__(self, x, y, z):
        self.x  = '(x-{:.10f})'.format(x) if not np.isclose(x, 0) else 'x'
        self.y  = '(y-{:.10f})'.format(y) if not np.isclose(y, 0) else 'y'
        self.z  = '(z-{:.10f})'.format(z) if not np.isclose(z, 0) else 'z'
        self.r2 = '({}**2+{}**2+{}**2)'.format(self.x, self.y, self.z)


def make_fps(rmin=None, rmax=None, nr=None, nrfps=1,
             xmin=None, xmax=None, nx=None, frame=0,
             ymin=None, ymax=None, ny=None, field_type=np.nan,
             zmin=None, zmax=None, nz=None, label=np.nan):
    """
    Generate the necessary field parameters of a numerical grid field
    as an exatomic.field.AtomicField.

    Args
        rmin (float): minimum value in an arbitrary cartesian direction
        rmax (float): maximum value in an arbitrary cartesian direction
        nr (int): number of grid points between rmin and rmax
        nrfps (int): number of field parameters with same dimensions
        xmin (float): minimum in x direction (optional)
        xmax (float): maximum in x direction (optional)
        nx (int): steps in x direction (optional)
        ymin (float): minimum in y direction (optional)
        ymax (float): maximum in y direction (optional)
        ny (int): steps in y direction (optional)
        zmin (float): minimum in z direction (optional)
        zmax (float): maximum in z direction (optional)
        nz (int): steps in z direction (optional)
        label (str): an identifier passed to the widget (optional)
        field_type (str): alternative identifier (optional)

    Returns
        fps (pd.Series): field parameters
    """
    if any((par is None for par in [rmin, rmax, nr])):
        raise Exception("Must supply at least rmin, rmax, nr.")
    d = {}
    allcarts = [['x', xmin, xmax, nx],
                ['y', ymin, ymax, ny],
                ['z', zmin, zmax, nz]]
    for akey, amin, amax, na in allcarts:
        amin = rmin if amin is None else amin
        amax = rmax if amax is None else amax
        na = nr if na is None else na
        d[akey] = [amin, amax, na, (amax - amin) / na]
    fp = pd.Series({'ox': d['x'][0],  'oy': d['y'][0],  'oz': d['z'][0],
                    'nx': d['x'][2],  'ny': d['y'][2],  'nz': d['z'][2],
                   'dxi': d['x'][3], 'dyj': d['y'][3], 'dzk': d['z'][3],
                   'dxj': 0, 'dyk': 0, 'dzi': 0, 'frame': frame,
                   'dxk': 0, 'dyi': 0, 'dzj': 0,})
    fp = pd.concat([fp] * nrfps, axis=1).T
    if isinstance(label, list) and len(label) == nrfps:
        fp['label'] = label
    else:
        fp['label'] = [label] * nrfps
    if isinstance(field_type, list) and len(field_type) == nrfps:
        fp['field_type'] = field_type
    else:
        fp['field_type'] = [field_type] * nrfps
    return fp


def clean_sh(sh):
    """
    Takes symbolic representation of solid harmonic functions
    and cleans them to lists of strings to be used in generating
    string forms of basis functions.

    Args
        sh (OrderedDict): the output of exatomic.algorithms.basis.solid_harmonics

    Returns
        clean (OrderedDict): cleaned for use in generating string basis functions
    """
    clean = OrderedDict()
    mlp = {1: 'nuc.x', -1: 'nuc.y', 0: 'nuc.z'}
    for L, ml in sh:
        if not L:
            clean[(L, ml)] = ['']
            continue
        if L == 1:
            clean[(L, ml)] = ['{{{}}}*'.format(mlp[ml])]
            continue
        symbolic = sh[(L, ml)].expand(basic=True)
        if type(symbolic) == Mul:
            coef, sym = symbolic.as_coeff_Mul()
            coef = '{:.10f}*'.format(coef)
            sym = str(sym).replace('*', '')
            sym = '*'.join(['{{nuc.{}}}'.format(char) for char in sym])
            clean[(L, ml)] = [''.join([coef, sym, '*'])]
            continue
        elif type(symbolic) == Add:
            clean[(L, ml)] = []
            dat = list(symbolic.as_coefficients_dict().items())
            for sym, coef in dat:
                coef ='{:.10f}*'.format(coef)
                powers = {'x': 0, 'y': 0, 'z': 0}
                sym = str(sym).replace('*', '')
                for i, j in zip(sym, sym[1:]):
                    if j.isnumeric(): powers[i] = int(j)
                    elif i.isalpha(): powers[i] = 1
                if sym[-1].isalpha(): powers[sym[-1]] = 1
                lin = '{{nuc.{}}}*'
                nlin = '{{nuc.{}}}**{}*'
                for cart, power in powers.items():
                    if power:
                        stargs = [cart]
                        fmt = lin if power == 1 else nlin
                        if power > 1: stargs.append(power)
                        coef += fmt.format(*stargs)
                clean[(L, ml)].append(coef)
    return clean


def _sphr_prefac(clean, L, ml, nuc):
    """
    Create the string of the pre-exponential factor of a given basis
    function as a function of (L, ml) quantum numbers and nuclear
    position

    Args
        clean (OrderedDict): result of clean_sh
        L (int): angular momentum quantum number
        ml (int): magnetic quantum number
        nuc (Nucpos): nuclear position

    Returns
        prefacs (list): pre-exponential factors
    """
    cln = clean[(L, ml)]
    if any(('nuc' in pre for pre in cln)):
        cln = [pre.format(nuc=nuc) for pre in cln]
    return cln

def _cart_prefac(L, l, m, n, nucpos):
    """
    As with _sphr_prefac, create the string version of the pre-exponential
    factor in a given basis function, this time as a function of cartesian
    powers (l, m, n) instead of (l, ml) quantum numbers.

    Args
        L (int): angular momentum quantum number
        l (int): powers of x
        m (int): powers of y
        n (int): powers of z
        nucpos (Nucpos): nuclear position

    Returns
        prefacs (list): pre-exponential factors
    """
    if L == 0: return ['']
    prefac, lin, nlin  = '', '{}*', '{}**{}*'
    mapper = OrderedDict([('{nuc.x}', l), ('{nuc.y}', m), ('{nuc.z}', n)])
    for atom, key in mapper.items():
        if not key: continue
        stargs = [atom]
        fmt = lin if key == 1 else nlin
        if key > 1: stargs.append(key)
        prefac += fmt.format(*stargs)
    return [prefac.format(nuc=nucpos)]

def gen_basfn(prefacs, group, r2str):
    """
    Given a list of pre-exponential factors and a group of
    primitive functions (slice of basis set table), return
    the string that is the basis function written out

    Args
        prefacs (list): string of pre-exponential factors
        group (exatomic.basis.GaussianBasisSet): a slice of the basis set table
        r2str (str): the r**2 term in the exponent (including atomic position)
    Returns
        basis function (str)
    """
    bastr = '{prefac}({prims})'
    bastrs = []
    primitive = '{{N:.8f}}*{{d:.8f}}*np.exp(-{{alpha:.8f}}*{r2str})'.format(r2str=r2str)
    for p, prefac in enumerate(prefacs):
        primitives = group.apply(lambda x: primitive.format(alpha=x.alpha, d=x.d, N=x.N), axis=1)
        bastrs.append(bastr.format(prefac=prefac, prims='+'.join(primitives.values)))
    return '+'.join(bastrs)

def _compile_basfns(basfns):
    """
    Given string representations of basis functions,
    define them as accepting points in cartesian space,
    then JIT compile and vectorize them. This compilation
    may be slow depending on the size of the basis set
    resultant functions can be evaluated quickly for
    dynamic exploration of the wave function.

    Args
        basfns (list): string representations of basis functions

    Returns
        bfns (OrderedDict): namespace where functions were compiled
    """
    if not config['dynamic']['numba'] == 'true': raise NotImplementedError()
    bfns = OrderedDict()
    header = """import numpy as np
from numba import vectorize"""
    dec = """
@vectorize(['float64(float64,float64,float64)'], nopython=True)"""
    basformat = """
{}
def {{}}(x, y, z): return {{}}""".format(dec).format
    funcs = """"""
    for i, basfn in enumerate(basfns):
        funcs = ''.join([funcs, basformat('bf{}'.format(i), basfn)])
    code = header + funcs
    code = compile(code, '<string>', 'exec')
    exec(code, bfns)
    del bfns['np']
    del bfns['vectorize']
    return bfns

def gen_basfns(universe, frame=None):
    """
    Given an exatomic.container.Universe that contains complete momatrix
    and basis_set attributes, generates and returns the strings corresponding
    to how the basis functions would be written out with paper and pencil.
    This is mainly for debugging and testing generality to deal with different
    computational codes' basis function ordering schemes.

    Args
        universe (exatomic.container.Universe): must contain momatrix and basis_set

    Returns
        bastrs (list): list of strings of basis functions
    """
    # Get the symbolic spherical functions
    lmax = universe.basis_set.lmax
    sh = clean_sh(solid_harmonics(lmax))
    # Group the dataframes appropriately
    bases = universe.basis_set.groupby('set')
    centers = universe.basis_set_order.groupby('center')
    basfns = []
    # Iterate over atomic positions
    for i, (seht, x, y, z) in enumerate(zip(universe.atom['set'], universe.atom['x'],
                                            universe.atom['y'],   universe.atom['z'])):
        # Atomic position
        nucpos = Nucpos(x, y, z)
        r2str = nucpos.r2
        # Regroup dataframes
        bas = bases.get_group(seht).groupby('L')
        basord = centers.get_group(i)
        if universe.basis_set.spherical:
            # Iterate over spherical atom-centered basis functions
            for L, ml, shfunc in zip(basord['L'], basord['ml'], basord['shell']):
                grp = bas.get_group(L).groupby('shell').get_group(shfunc)
                prefacs = _sphr_prefac(sh, L, ml, nucpos)
                basfns.append(gen_basfn(prefacs, grp, r2str))
        else:
            # Iterate over cartesian atom-centered basis functions
            for L, l, m, n, shfunc in zip(basord['L'], basord['l'], basord['m'],
                                          basord['n'], basord['shell']):
                grp = bas.get_group(L).groupby('shell').get_group(shfunc)
                prefacs = _cart_prefac(L, l, m, n, nucpos)
                basfns.append(gen_basfn(prefacs, grp, r2str))
    return basfns

def numerical_grid_from_field_params(field_params):
    mx = field_params.ox[0] + (field_params.nx[0] - 1) * field_params.dxi[0]
    my = field_params.oy[0] + (field_params.ny[0] - 1) * field_params.dyj[0]
    mz = field_params.oz[0] + (field_params.nz[0] - 1) * field_params.dzk[0]
    x = np.linspace(field_params.ox[0], mx, field_params.nx[0])
    y = np.linspace(field_params.oy[0], my, field_params.ny[0])
    z = np.linspace(field_params.oz[0], mz, field_params.nz[0])
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

def _determine_vectors(universe, vector):
    if isinstance(vector, int): return [vector]
    if isinstance(vector, (list, tuple, range)): return vector
    if vector is None:
        try:
            homo = universe.orbita.get_orbital().vector
        except:
            try:
                homo = universe.atom['Zeff'].sum() // 2
            except KeyError:
                homo = universe.atom['Z'].sum() // 2
        if homo < 15:
            return range(0, homo + 15)
        else:
            return range(homo - 15, nclosed + 5)
    else:
        raise TypeError('Try specifying vector as a list or int')

def _determine_mocoefs(universe, mocoefs, vector):
    if mocoefs is None: return 'coef'
    if mocoefs not in universe.momatrix.columns:
        raise Exception('mocoefs must be a column in universe.momatrix')
    return mocoefs

def _evaluate_basis(universe, basis_values, x, y, z):
    for name, basis_function in universe.basis_functions.items():
        basis_values[:,int(name[2:])] = basis_function(x, y, z)
    return basis_values

def _evaluate_fields(universe, basis_values, vector, field_data, mocoefs):
    vectors = universe.momatrix.groupby('orbital')
    for i, vno in enumerate(vector):
        vec = vectors.get_group(vno)
        for chi, coef in zip(vec['chi'], vec[mocoefs]):
            field_data[:, i] += coef * basis_values[:, chi]
    return field_data

def _compute_mos(basis_values, coefs, vector):
    nfield = len(vector)
    field_data = np.zeros((basis_values.shape[0], nfield), dtype=np.float64)
    for i, vec in enumerate(vector):
        for j, bs in enumerate(coefs):
            field_data[:, i] += basis_values[:, j] * coefs[:, j]
    return [field_data[:, j] for j in range(nfield)]

def add_molecular_orbitals(universe, field_params=None, mocoefs=None,
                           vector=None, frame=None):
    """
    If a universe contains enough information to generate
    molecular orbitals (basis_set, basis_set_summary and momatrix),
    evaluate the molecular orbitals on a discretized grid. If vector
    is not provided, attempts to calculate vectors by the sum of Z/Zeff
    of the atoms present divided by two; roughly (HOMO-15,LUMO+5).

    Args
        field_params (tuple): tuple of (min, max, steps)
        mocoefs (str): column in momatrix (default is 'coef')
        vector (int, list, range): the MO vectors to evaluate

    Warning:
        Removes any fields previously attached to the universe
    """
    if hasattr(universe, '_field'):
        del universe.__dict__['_field']
    # Preliminary setup for minimal input parameters
    vector = _determine_vectors(universe, vector)
    mocoefs = _determine_mocoefs(universe, mocoefs, vector)
    frame = universe.atom.maxframe if frame is None else frame
    field_params = _determine_field_params(universe, field_params, len(vector))
    # Array dimensions
    lastatom = universe.atom.last_frame
    nbas = lastatom['set'].map(universe.basis_set.functions().to_dict()).sum()
    nprim = lastatom['set'].map(universe.basis_set.primitives().to_dict()).sum()

    if not hasattr(universe, 'basis_functions'):
        print('Warning: not extensively tested. Please be careful.')
        print('Compiling basis functions, may take a while.')
        t1 = datetime.now()
        basfns = gen_basfns(universe)
        universe.basis_functions = _compile_basfns(basfns)
        t2 = datetime.now()
        tt = (t2-t1).total_seconds()
        print("Compile time : {:.2f}s\n"
              "Size of basis: {} primitives\n"
              "             : {} functions".format(tt, nprim, nbas))

    t1 = datetime.now()
    x, y, z = numerical_grid_from_field_params(field_params)
    npoints = len(x)
    nvec = len(vector)

    basis_values = np.zeros((npoints, nbas), dtype=np.float64)
    basis_values = _evaluate_basis(universe, basis_values, x, y, z)

    field_data = np.zeros((npoints, nvec), dtype=np.float64)
    field_data = _evaluate_fields(universe, basis_values, vector, field_data, mocoefs)
    universe.field = AtomicField(field_params, field_values=[field_data[:, i]
                                 for i in range(nvec)])
    t2 = datetime.now()
    tt = (t2-t1).total_seconds()

    print("Computing MOs: {:.2f}s\n"
          "             : {} MOs".format(tt, nvec))
    universe._traits_need_update = True



if config['dynamic']['numba'] == 'true':
    from numba import jit, vectorize
    density_from_momatrix = jit(nopython=True)(density_from_momatrix)
    density_as_square = jit(nopython=True)(density_as_square)
    momatrix_as_square = jit(nopython=True)(momatrix_as_square)
    meshgrid3d = jit(nopython=True, cache=True, nogil=True)(meshgrid3d)
    #compute_mos = jit(nopython=True, cache=True)(compute_mos)
else:
    print('add_molecular_orbitals will not work without having numba installed')
