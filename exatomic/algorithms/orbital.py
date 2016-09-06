# -*- coding: utf-8 -*-
'''
Numerical Orbital Functions
#########################
Building discrete molecular orbitals (for visualization) requires a complex set of operations that
are provided by this module and wrapped into a clean API.
'''
import numpy as np
from exatomic._config import config
from datetime import datetime

## For voluminating GTFs
import re
import pandas as pd
from exa import Series
import sympy as sy
from sympy import Add, Mul
from exatomic.algorithms.basis import solid_harmonics, _vec_sloppy_normalize
from exatomic.field import AtomicField
from collections import OrderedDict

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

def make_field_params(rmin, rmax, nr, frame=0, label=np.nan, field_type=np.nan):
    """
    Helper function that generates necessary field parameters of a
    numerical grid related to the exatomic.field.AtomicField table.

    Args
        rmin (float): minimum value in a single cartesian direction
        rmax (float): maximum value in a single cartesian direction
        nr (int): number of grid points between rmin and rmax
    """
    di = (rmax - rmin) / nr
    return pd.Series({ 'ox': rmin, 'oy': rmin, 'oz': rmin,
                       'nx': nr,   'ny': nr,   'nz': nr,
                      'dxi': di,  'dyj': di,  'dzk': di,
                      'dxj': 0,   'dyk': 0,   'dzi': 0,
                      'dxk': 0,   'dyi': 0,   'dzj': 0,
                      'field_type': field_type, 'label': label,
                      'frame': frame})


def _gen_prefactor(sh, l, ml, nucpos):
    """
    For a given (l, ml) combination, determine
    the pre-exponential factor in a given basis function.

    Args
        sh: result of exatomic.algorithms.basis.solid_harmonics
        l (int): angular quantum Number
        ml (int): magnetic quantum number
        nucpos (dict): atomic x, y, z positions

    Returns
        prefacs (list): the pre-exponential factors as strings
    """
    if l == 0: return ['']
    symbolic = sh[(l, ml)]
    if l == 1:
        mld = {1: 'x', -1: 'y', 0: 'z'}
        return ['(' + nucpos[mld[ml]] + ') * ']
    if l > 1:
        prefacs = []
        symbolic = sh[(l, ml)].expand(basic=True)
        if type(symbolic) == Mul:
            coef, sym = symbolic.as_coeff_Mul()
            sym = str(sym).replace('*', '')
            sympos = ['(' + nucpos[char] + ')' for char in sym]
            return [' * '.join(['{:.8f}'.format(coef)] + sympos + [''])]
        elif type(symbolic) == Add:
            dat = list(symbolic.expand(basic=True).as_coefficients_dict().items())
            coefs = [i[1] for i in dat]
            syms = [str(i[0]).replace('*', '') for i in dat]
        for coef, sym in zip(coefs, syms):
            prefac = '{:.8f} * '.format(coef)
            powers = {'x': 0, 'y': 0, 'z': 0}
            for i, char in enumerate(sym):
                if char.isnumeric():
                    powers[sym[i - 1]] += int(char) - 1
                else:
                    powers[char] += 1
            for cart in powers:
                if powers[cart]:
                    if powers[cart] == 1:
                        prefac += '(' + nucpos[cart] + ')' + ' * '
                    else:
                        prefac += '(' + nucpos[cart] + ')**' + str(powers[cart]) + ' * '
            prefacs.append(prefac)
        return prefacs

def _cartesian_prefactor(l, xs, ys, zs, nucpos):
    """
    As with _gen_prefactor, create the string version of the pre-exponential
    factor in a given basis function, this time as a function of cartesian
    powers instead of (l, ml) quantum numbers.

    Args
        l (int): angular momentum quantum number
        xs (list): powers of x
        ys (list): powers of y
        zs (list): powers of z
        nucpos (dict): atomic x,y,z positions
    """
    if l == 0: return ['']
    prefacs = []
    # Special case for l == 1 just cuts down on characters to compile
    if l == 1:
        for x, y, z in zip(xs, ys, zs):
            prefac = ''
            if x:
                prefac += '({}) * '.format(nucpos['x'])
            if y:
                prefac += '({}) * '.format(nucpos['y'])
            if z:
                prefac += '({}) * '.format(nucpos['z'])
            prefacs.append(prefac)
        return prefacs
    for x, y, z in zip(xs, ys, zs):
        prefac = ''
        if x:
            prefac += '({})**{} * '.format(nucpos['x'], x)
        if y:
            prefac += '({})**{} * '.format(nucpos['y'], y)
        if z:
            prefac += '({})**{} * '.format(nucpos['z'], z)
        prefacs.append(prefac)
    return prefacs

def _enumerate_primitives_prefacs(prefacs, group, r2str):
    """
    Given a list of pre-exponential facors and a group of
    primitive functions (slice of basis set table), return
    the string that is the basis function written out

    Args
        prefacs (list): string of pre-exponential factors
        group (exatomic.basis.GaussianBasisSet): a slice of the basis set table
        r2str (str): the r**2 term in the exponent (including atomic position)
    """
    bastr = ''
    for p, prefac in enumerate(prefacs):
        if prefac:
            bastr += prefac + '('
        for i, (alpha, d, N) in enumerate(zip(group['alpha'], group['d'], group['N'])):
            if i == len(group) - 1:
                bastr += str(N) + ' * ' + str(d) + ' * np.exp(-' + \
                         str(alpha) + ' * (' + r2str + '))'
            else:
                bastr += str(N) + ' * ' + str(d) + ' * np.exp(-' + \
                         str(alpha) + ' * (' + r2str + ')) + '
        if p == len(prefacs) - 1 and prefac:
            bastr += ')'
        else:
            if prefac:
                bastr += ') + '
    return bastr


def _add_bfns_to_universe(universe, bastrs, inplace=True):
    """
    Given string representations of basis functions,
    define them as accepting points in cartesian space,
    then JIT compile and vectorize them. Subsequently add them
    to the universe so that the numerical grid can be changed
    without recompiling the functions. This will be
    slow in the compilation but the resulting functions
    can be evaluated quickly.

    Args
        universe (exatomic.container.Universe): universe with momatrix and basis_set
        bastrs (list): string representations of the basis functions in the universe
        inplace (bool): adds functions to universe (default)

    Returns
        if inplace == False; return OrderedDict(basis_functions)


    """
    if not config['dynamic']['numba'] == 'true':
        raise NotImplementedError()
    basis_keys = []
    bfns = OrderedDict()
    header = """import numpy as np
from numba import vectorize"""
    dec = """
@vectorize(['float64(float64,float64,float64)'], nopython=True)"""
    basformat = """
{}
def {}(x, y, z): return {}""".format
    funcs = """"""
    baskeys = []
    for i, bfn in enumerate(bastrs):
        bas = 'bf' + str(i)
        baskeys.append(bas)
        funcs = ''.join([funcs, basformat(dec, bas, bfn)])
    code = header + funcs
    exec(code, bfns)
    if inplace:
        universe.basis_functions = bfns
    else:
        return bfns


def gen_string_bfns(universe, kind='spherical'):
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
    lmax = universe.basis_set['L'].cat.as_ordered().max()
    universe.basis_set['N'] = _vec_sloppy_normalize(universe.basis_set['alpha'].values,
                                                    universe.basis_set['L'].values)
    bases = universe.basis_set.groupby('set')
    if hasattr(universe, 'basis_set_order'):
        centers = universe.basis_set_order.groupby('center')
    sh = solid_harmonics(lmax)
    bfns = []
    bastrs = []
    for i, (seht, x, y, z) in enumerate(zip(universe.atom['set'], universe.atom['x'],
                                            universe.atom['y'],   universe.atom['z'])):
        spherical = universe.basis_set_summary.ix[seht].spherical
        xastr = 'x' if np.isclose(x, 0) else 'x - ' + str(x)
        yastr = 'y' if np.isclose(y, 0) else 'y - ' + str(y)
        zastr = 'z' if np.isclose(z, 0) else 'z - ' + str(z)
        nucpos = {'x': xastr, 'y': yastr, 'z': zastr}
        r2str = '(' + xastr + ')**2 + (' + yastr + ')**2 + (' + zastr + ')**2'
        if hasattr(universe, 'basis_set_order'):
            bas = bases.get_group(seht).groupby('L')
            basord = centers.get_group(i + 1)
            for L, ml, shfunc in zip(basord['L'], basord['ml'], basord['shell_function']):
                grp = bas.get_group(L).groupby('shell_function').get_group(shfunc)
                bastr = ''
                prefacs = _gen_prefactor(sh, L, ml, nucpos)
                bastrs.append(_enumerate_primitives_prefacs(prefacs, grp, r2str))
        else:
            bas = bases.get_group(seht).groupby('shell_function')
            for f, grp in bas:
                if len(grp) == 0: continue
                l = grp['L'].values[0]
                if spherical:
                    sym_keys = universe.spherical_gtf_order.symbolic_keys(l)
                    for L, ml in sym_keys:
                        bastr = ''
                        prefacs = _gen_prefactor(sh, L, ml, nucpos)
                        bastrs.append(_enumerate_primitives_prefacs(prefacs, grp, r2str))
                else:
                    subcart = universe.cartesian_gtf_order[universe.cartesian_gtf_order['l'] == l]
                    prefacs = _cartesian_prefactor(l, subcart['x'], subcart['y'], subcart['z'], nucpos)
                    for prefac in prefacs:
                        bastrs.append(_enumerate_primitives_prefacs([prefac], grp, r2str))
    return bastrs

def numerical_grid_from_field_params(field_params):
    mx = field_params.ox + (field_params.nx - 1) * field_params.dxi
    my = field_params.oy + (field_params.ny - 1) * field_params.dyj
    mz = field_params.oz + (field_params.nz - 1) * field_params.dzk
    x = np.linspace(field_params.ox, mx, field_params.nx)
    y = np.linspace(field_params.oy, my, field_params.ny)
    z = np.linspace(field_params.oz, mz, field_params.nz)
    return meshgrid3d(x, y, z)

def _determine_field_params(universe, field_params):
    if field_params is None:
        dr = 41
        rmin = min(universe.atom['x'].min(),
                   universe.atom['y'].min(),
                   universe.atom['z'].min()) - 4
        rmax = max(universe.atom['x'].max(),
                   universe.atom['y'].max(),
                   universe.atom['z'].max()) + 4
        return make_field_params(rmin, rmax, dr)
    else:
        return make_field_params(*field_params)

def _determine_vectors(universe, vector):
    if isinstance(vector, int):
        return [vector]
    elif isinstance(vector, (list, tuple, range)):
        return vector
    elif vector is None:
        try:
            nclosed = universe.atom['Zeff'].sum() // 2
        except KeyError:
            nclosed = universe.atom['Z'].sum() // 2
        if nclosed - 15 < 0:
            return range(0, nclosed + 15)
        else:
            return range(nclosed - 15, nclosed + 5)
    else:
        raise TypeError('Try specifying vector as a list or int')

def _determine_mocoefs(universe, mocoefs):
    if mocoefs is None:
        return 'coefficient'
    else:
        if mocoefs not in universe.momatrix.columns:
            raise Exception('mocoefs must be a column in universe.momatrix')
        if vector is None:
            raise Exception('Must supply vector if non-canonical MOs are used')

def _evaluate_basis(universe, basis_values, x, y, z):
    for name, basis_function in universe.basis_functions.items():
        if 'bf' in name:
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

def add_mos_to_universe(universe, field_params=None, mocoefs=None, vector=None):
    """
    If a universe contains enough information to generate
    molecular orbitals (basis_set, basis_set_summary and momatrix),
    evaluate the molecular orbitals on a discretized grid. If vector
    is not provided, attempts to calculate vectors by the sum of Z/Zeff
    of the atoms present divided by two; roughly (HOMO-15,LUMO+5).

    Args
        field_params (tuple): tuple of (min, max, steps)
        mocoefs (str): column in momatrix (default is 'coefficient')
        vector (int, list, range): the MO vectors to evaluate

    Warning:
        Removes any fields previously attached to the universe
    """
    if hasattr(universe, '_field'):
        del universe.__dict__['_field']

    field_params = _determine_field_params(universe, field_params)
    mocoefs = _determine_mocoefs(universe, mocoefs)
    vector = _determine_vectors(universe, vector)

    ### TODO :: optimizations.
    ###         1) determine approximately the HOMO based on atoms (or have it parsed into the frame table)
    ###         2) screen the MOMatrix to determine the basis functions contributing to the orbitals of interest
    ###         3) only numerically evaluate the basis functions that show up in the MOs
    ### Would look something like this? Preliminary thoughts are that most basis
    ### functions are required for a reasonable span of MOs so this may not be the way to go.
    #bases_of_int = np.unique(np.concatenate([universe.momatrix.contributions(i)['chi'].values for i in vector]))
    #basfns = [basfns[i] for i in bases_of_int]

    basfns = gen_string_bfns(universe)
    print('Warning: not extensively tested. Please be careful.')
    print('Compiling basis functions, may take a while.')
    t1 = datetime.now()

    _add_bfns_to_universe(universe, basfns)
    x, y, z = numerical_grid_from_field_params(field_params)
    nbas = universe.basis_set_summary['function_count'].sum()
    npoints = len(x)
    nvec = len(vector)

    t2 = datetime.now()
    try:
        print('Took {:.2f}s to compile basis functions ' \
              'with {} characters, {} primitives and {} contracted functions'.format(
                (t2-t1).total_seconds(), sum([len(b) for b in basfns]),
                universe.basis_set_summary['primitive_count'].sum(),
                nbas))
    except KeyError:
        print('Took {:.2f}s to compile basis functions ' \
              'with {} characters and {} contracted functions'.format(
                (t2-t1).total_seconds(), sum([len(b) for b in basfns]),
                nbas))

    basis_values = np.zeros((npoints, nbas), dtype=np.float64)
    basis_values = _evaluate_basis(universe, basis_values, x, y, z)

    field_data = np.zeros((npoints, nvec), dtype=np.float64)
    field_data = _evaluate_fields(universe, basis_values, vector, field_data, mocoefs)

    #coefs = universe.momatrix.square(column=mocoefs)
    #field_values = compute_mos(basis_values, coefs, vector)
    #print('basis values', basis_values.shape)
    #print('field values', field_values.shape)

    nfps = pd.concat([field_params] * nvec, axis=1).T
    universe.field = AtomicField(nfps, field_values=[field_data[:, i]
                                 for i in range(nvec)])
    universe._traits_need_update = True

def update_molecular_orbitals(universe, field_params=None, mocoefs=None, vector=None):
    """
    Provided the universe already contains the basis_functions attribute,
    reevaluates the MOs with the new field_params

    Args
        universe (exatomic.container.Universe): universe with basis_functions attribute
        field_params (pd.Series or rmin, rmax, nr): dimensions of new numerical grid
    """
    if hasattr(universe, '_field'):
        del universe.__dict__['_field']

    print(field_params)
    print(type(field_params))

    field_params = _determine_field_params(universe, field_params)
    mocoefs = _determine_mocoefs(universe, mocoefs)
    vector = _determine_vectors(universe, vector)
    x, y, z = numerical_grid_from_field_params(field_params)
    nbas = universe.basis_set_summary['function_count'].sum()
    npoints = len(x)
    nvec = len(vector)

    basis_values = np.zeros((npoints, nbas), dtype=np.float64)
    basis_values = _evaluate_basis(universe, basis_values, x, y, z)

    field_data = np.zeros((npoints, nvec), dtype=np.float64)
    field_data = _evaluate_fields(universe, basis_values, vector, field_data, mocoefs)

    nfps = pd.concat([field_params] * nvec, axis=1).T

    universe.field = AtomicField(nfps, field_values=[field_data[:, i]
                                 for i in range(nvec)])
    universe._traits_need_update = True


if config['dynamic']['numba'] == 'true':
    from numba import jit, vectorize
    density_from_momatrix = jit(nopython=True)(density_from_momatrix)
    density_as_square = jit(nopython=True)(density_as_square)
    momatrix_as_square = jit(nopython=True)(momatrix_as_square)
    meshgrid3d = jit(nopython=True, cache=True, nogil=True)(meshgrid3d)
    #compute_mos = jit(nopython=True, cache=True)(compute_mos)
else:
    print('add_mos_to_universe will not work without having numba installed')
