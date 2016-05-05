# -*- coding: utf-8 -*-
'''
Basis Functions
=============================
Representations of commonly used basis functions
'''
import numpy as np
import sympy as sy
from collections import OrderedDict
from exa import _conf
from exa.analytical import Symbolic
from exa.numerical import DataFrame


ex, ey, ez = sy.symbols('x, y, z', imaginary=False)
lmap = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6, 'k': 7, 'l': 8,
        'm': 9, 'px': 1, 'py': 1, 'pz': 1}
ml_count = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9, 'h': 11, 'i': 13, 'k': 15,
            'l': 17, 'm': 19}


class SymbolicBasis(Symbolic, DataFrame):
    '''
    '''
    pass


class Basis(DataFrame):
    '''
    '''
    _default_shell_order = cartesian_gaussian_ijk
    _columns = ['alpha', 'c', 'function', 'shell', 'symbol']
    _indices = ['basis']
    _categories = {'symbol': str, 'shell': str, 'name': str}

    def __init__(self, *args, shell_order, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(shell_order, int):
            self.shell_order = cartesian_gaussian_ijk(shell_order)
        elif isinstance(shell_order, list):
            self.shell_order = shell_order
        else:
            raise TypeError()


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
                    values[h] = [i, j, k]
                    h += 1
    return values


def _symbolic_cartesian_gtfs(universe):
    '''
    Generate symbolic Gaussian basis functions.

    Args:
        universe (:class:`~atomic.unvierse.Universe`): Universe with basis dataframe

    Returns:
        basis_funcs (list): List of basis functions in atom, function order

    Warning:
        The functions returned are not normalized.
    '''
    functions = []
    bases = universe.basis.groupby('symbol')
    for symbol, x, y, z in zip(universe.atom['symbol'], universe.atom['x'], universe.atom['y'], universe.atom['z']):
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
