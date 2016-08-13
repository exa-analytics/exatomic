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


def _gen_prefactor(sh, l, ml, porder, corder, phase=False):
    """
    For a given (l, ml) combination, determine
    the pre-exponential factor in a given basis function.

    Args
        sh: result of exatomic.algorithms.basis.solid_harmonics
        l (int): angular quantum Number
        ml (int): magnetic quantum number
        porder (list): atomic x, y, z positions in the order of p functions
        corder (list): mapping of {'x', 'y', 'z'} in same order as porder
        phase (bool): allows for testing of Condon-Shortley phase factors

    Returns
        prefacs (list): the pre-exponential factors as strings
    """
    if l == 0: return ['']
    symbolic = sh[(l, ml)]
    if l == 1:
        return ['(' + porder[ml] + ') * ']
    if l > 1:
        prefacs = []
        symbolic = sh[(l, ml)].expand(basic=True)
        if type(symbolic) == Mul:
            coef, sym = symbolic.as_coeff_Mul()
            sym = str(sym).replace('*', '')
            sympos = ['(' + porder[corder.index(char)] + ')' for char in sym]
            if phase:
                return [' * '.join(['-' + '{:.8f}'.format(coef)] + sympos + [''])]
            else:
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
                        if len(porder[corder.index(cart)]) == 1:
                            prefac += porder[corder.index(cart)] + ' * '
                        else:
                            prefac += '(' + porder[corder.index(cart)] + ')' + ' * '
                    else:
                        prefac += '(' + porder[corder.index(cart)] + ')**' + str(powers[cart]) + ' * '
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
        xastr = 'x' if np.isclose(x, 0) else 'x - ' + str(x)
        yastr = 'y' if np.isclose(y, 0) else 'y - ' + str(y)
        zastr = 'z' if np.isclose(z, 0) else 'z - ' + str(z)
        porder = [zastr, xastr, yastr]
        corder = ['z', 'x', 'y']
        r2str = '(' + xastr + ')**2 + (' + yastr + ')**2 + (' + zastr + ')**2'
        if hasattr(universe, 'basis_set_order'):
            bas = bases.get_group(seht).groupby('L')
            basord = centers.get_group(i + 1)
            for L, ml, shfunc in zip(basord['L'], basord['ml'], basord['shell_function']):
                grp = bas.get_group(L).groupby('shell_function').get_group(shfunc)
                bastr = ''
                prefacs = _gen_prefactor(sh, L, ml, porder, corder)
                bastrs.append(_enumerate_primitives_prefacs(prefacs, grp, r2str))
        else:
            bas = bases.get_group(seht).groupby('shell_function')
            for f, grp in bas:
                if len(grp) == 0: continue
                l = grp['L'].values[0]
                if kind == 'spherical':
                    sym_keys = universe.spherical_gtf_order.symbolic_keys(l)
                elif kind == 'cartesian':
                    sym_keys = universe.spherical_gtf_order.symbolic_keys(l)
                for L, ml in sym_keys:
                    bastr = ''
                    prefacs = _gen_prefactor(sh, L, ml, porder, corder)
                    bastrs.append(_enumerate_primitives_prefacs(prefacs, grp, r2str))
    return bastrs

def numerical_grid_from_field_params(field_params):
    mx = field_params.ox + (field_params.nx - 1) * field_params.dxi
    my = field_params.oy + (field_params.ny - 1) * field_params.dyj
    mz = field_params.oz + (field_params.nz - 1) * field_params.dzk
    x = np.linspace(field_params.ox, mx, field_params.nx)
    y = np.linspace(field_params.oy, my, field_params.ny)
    z = np.linspace(field_params.oz, mz, field_params.nz)
    y, x, z = np.meshgrid(x, y, z)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    return x, y, z

def _determine_vectors(universe, vector):
    if isinstance(vector, int):
        vector = [vector]
    elif vector is None:
        try:
            nclosed = universe.atom['Zeff'].sum() // 2
            if nclosed -15 < 0:
                vector = range(0, nclosed + 10)
            else:
                vector = range(nclosed - 15, nclosed + 5)
        except KeyError:
            nclosed = universe.atom['Z'].sum() // 2
            if nclosed - 15 < 0:
                vector = range(0, nclosed + 10)
            else:
                vector = range(nclosed - 15, nclosed + 5)
    elif not isinstance(vector, (list, tuple, range)):
        raise TypeError()
    return vector


def add_mos_to_universe(universe, *field_params, mocoefs=None, vector=None):
    """
    Provided a universe contains enough information to regenerate
    molecular orbitals (complete basis set specification and C matrix),
    this function evaluates the molecular orbitals on a discretized grid
    determined by field_params. field_params is either a pandas.Series
    containing ['ox', 'oy', 'oz', 'nx', 'ny', 'nz', 'dxi', 'dyj', 'dzk',
    'dxj', 'dyk', 'dzi', 'dxk', 'dyi', 'dzj', 'field_type', 'label', 'frame']
    attributes (see make_field_params) or a tuple of (rmin, rmax, nr), the
    bounding box and number of points along a single cartesian direction.

    If no argument vector is passed, naively compute the number of closed
    shell orbitals by summing 'Z/Zeff' in the atom table, divide by 2, and
    produce HOMO-15 to LUMO+5.

    Warning:
        Removes any fields previously attached to the universe
    """
    if hasattr(universe, '_field'):
        del universe.__dict__['_field']
    field_params = field_params[0] if type(field_params[0]) == pd.Series else make_field_params(*field_params)
    vectors = universe.momatrix.groupby('orbital')
    if mocoefs is not None:
        if vector is None:
            raise Exception('Must supply vector if non-canonical MOs are used')
    vector = _determine_vectors(universe, vector)

    ### TODO :: optimizations.
    ###         1) determine approximately the HOMO based on atoms (or have it parsed into the frame table)
    ###         2) screen the MOMatrix to determine the basis functions contributing to the orbitals of interest
    ###         3) only numerically evaluate the basis functions that show up in the MOs
    ### Would look something like this? Preliminary thoughts are that most basis
    ### functions are required for a reasonable span of MOs so this may not be the way to go.
    #bases_of_int = np.unique(np.concatenate([universe.momatrix.contributions(i)['chi'].values for i in vector]))
    #basfns = [basfns[i] for i in bases_of_int]

    print('Warning: not extensively tested. Please be careful.')
    basfns = gen_string_bfns(universe)
    print('Compiling basis functions, may take a while.')
    t1 = datetime.now()
    _add_bfns_to_universe(universe, basfns)
    t2 = datetime.now()
    print('It took {:.8f}s to compile the basis "module" with {} characters'.format(
            (t2-t1).total_seconds(), sum([len(b) for b in basfns])))

    x, y, z = numerical_grid_from_field_params(field_params)
    nbas = universe.basis_set_summary['function_count'].sum()
    npoints = len(x)
    nvec = len(vector)

    basis_values = np.zeros((npoints, nbas), dtype=np.float64)
    for name, basis_function in universe.basis_functions.items():
        if 'bf' in name:
            basis_values[:,int(name[2:])] = basis_function(x, y, z)

    field_data = np.zeros((npoints, nvec), dtype=np.float64)

    if mocoefs is None:
        if mocoefs not in universe.momatrix.columns:
            mocoefs = 'coefficient'

    cnt = 0
    for vno in vector:
        vec = vectors.get_group(vno)
        for chi, coef in zip(vec['chi'], vec[mocoefs]):
            field_data[:, cnt] += coef * basis_values[:, chi]
        cnt += 1

    nfps = pd.concat([field_params] * nvec, axis=1).T

    universe.field = AtomicField(nfps, field_values=[field_data[:, i]
                                 for i in range(nvec)])
    universe._traits_need_update = True

def update_molecular_orbitals(universe, *field_params, mocoefs=None, vector=None):
    """
    Provided the universe already contains the basis_functions attribute,
    reevaluates the MOs with the new field_params

    Args
        universe (exatomic.container.Universe): universe with basis_functions attribute
        field_params (pd.Series or rmin, rmax, nr): dimensions of new numerical grid
    """
    del universe.__dict__['_field']
    if isinstance(field_params[0], pd.Series):
        field_params = field_params[0]
    else:
        field_params = make_field_params(*field_params)
    vectors = universe.momatrix.groupby('orbital')
    if mocoefs is not None:
        if vector is None:
            raise Exception('Must supply vector if non-canonical MOs are used')
    vector = _determine_vectors(universe, vector)

    x, y, z = numerical_grid_from_field_params(field_params)
    nbas = universe.basis_set_summary['function_count'].sum()
    npoints = len(x)
    nvec = len(vector)

    basis_values = np.zeros((npoints, nbas), dtype=np.float64)

    for name, basis_function in universe.basis_functions.items():
        if 'bf' in name:
            basis_values[:,int(name[2:])] = basis_function(x, y, z)

    field_data = np.zeros((npoints, nvec), dtype=np.float64)

    print(mocoefs)
    print(universe.momatrix.columns)

    if mocoefs is None:
        print('mocoefs is None')
        if mocoefs not in universe.momatrix.columns:
            mocoefs = 'coefficient'

    print(mocoefs)

    cnt = 0
    for vno in vector:
        vec = vectors.get_group(vno)
        for chi, coef in zip(vec['chi'], vec[mocoefs]):
            field_data[:, cnt] += coef * basis_values[:, chi]
        cnt += 1

    nfps = pd.concat([field_params] * nvec, axis=1).T

    universe.field = AtomicField(nfps, field_values=[field_data[:, i]
                                 for i in range(nvec)])
    universe._traits_need_update = True


if config['dynamic']['numba'] == 'true':
    from numba import jit, vectorize
    density_from_momatrix = jit(nopython=True)(density_from_momatrix)
    density_as_square = jit(nopython=True)(density_as_square)
    momatrix_as_square = jit(nopython=True)(momatrix_as_square)
else:
    print('add_mos_to_universe will not work without having numba installed')
