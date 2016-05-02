# -*- coding: utf-8 -*-
'''
Basis Functions
=============================
Representations of commonly used basis functions
'''
import numpy as np
import sympy as sy
from exa import _conf
from exa.analytical import Symbolic
from exa.numerical import DataFrame


ex, ey, ez = sy.symbols('x, y, z', imaginary=False)
lmap = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4,
        'h': 5, 'i': 6, 'j': 7, 'k': 8, 'l': 9,
        'm': 10, 'px': 1, 'py': 1, 'pz': 1}


class SymbolicBasis(Symbolic, DataFrame):
    '''
    '''
    pass


class Basis(DataFrame):
    '''
    '''
    _columns = ['alpha', 'c', 'function', 'shell', 'symbol']
    _indices = ['basis']
    _categories = {'symbol': str, 'shell': str, 'name': str}


def cartesian_gaussian_ijk(l):
    '''
    Generate coefficients for cartesian Gaussian type functions.

    Note:
        These coefficients result in linearly dependent Gaussian
        type functions.

    Warning:
        This function returns column-major (Fortran) order!
    '''
    m = l + 1
    n = (m + 1) * m // 2
    values = np.empty((n, 3), dtype=np.int64)
    h = 0
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if i + j + k == l:
                    values[h] = [k, j, i]
                    h += 1
    return values


def symbolic_gtfs(atom, basis):
    '''
    Generate symbolic Gaussian basis functions.

    Args:
        atom (:class:`~atomic.atom.Atom`): Atom dataframe
        basis (:class:`~atomic.basis.Basis`): Basis dataframe
    '''
    functions = []
    bases = basis.groupby('symbol')
    for symbol, x, y, z in zip(atom['symbol'], atom['x'], atom['y'], atom['z']):
        bas = bases.get_group(symbol).groupby('function')
        rx = ex - x
        ry = ey - y
        rz = ez - z
        r2 = rx**2 + ry**2 + rz**2
        for f, grp in bas:
            l = lmap[grp['shell'].values[0]]
            for i, j, k in cartesian_gaussian_ijk(l):
                function = 0
                for alpha, c in zip(grp['alpha'], grp['c']):
                    function += c * rx**int(i) * ry**int(j) * rz**int(k) * sy.exp(-alpha * r2)
                functions.append(function)
    return functions


if _conf['pkg_numba']:
    from numba import jit
    cartesian_gaussian_ijk = jit(nopython=True, cache=True)(cartesian_gaussian_ijk)
