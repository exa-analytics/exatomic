# -*- coding: utf-8 -*-
'''
Orbital DataFrame
=============================
Orbital information such as centers and energies.

+-------------------+----------+-------------------------------------------+
| Column            | Type     | Description                               |
+===================+==========+===========================================+
| frame             | category | non-unique integer (req.)                 |
+-------------------+----------+-------------------------------------------+
| orbital           | int      | vector of MO coefficient matrix           |
+-------------------+----------+-------------------------------------------+
| label             | int      | label of orbital                          |
+-------------------+----------+-------------------------------------------+
| occupation        | float    | population of orbital                     |
+-------------------+----------+-------------------------------------------+
| energy            | float    | eigenvalue of orbital eigenvector         |
+-------------------+----------+-------------------------------------------+
| symmetry          | str      | symmetry designation (if applicable)      |
+-------------------+----------+-------------------------------------------+
| x                 | float    | orbital center in x                       |
+-------------------+----------+-------------------------------------------+
| y                 | float    | orbital center in y                       |
+-------------------+----------+-------------------------------------------+
| z                 | float    | orbital center in z                       |
+-------------------+----------+-------------------------------------------+
'''
import re
import numpy as np
import pandas as pd
from numba import vectorize, float64
from exa import DataFrame, Series
from exa.algorithms import meshgrid3d
from exatomic._config import config
from exatomic.field import AtomicField


class Orbital(DataFrame):
    '''
    Note:
        Spin zero means alpha spin or unknown and spin one means beta spin.
    '''
    _columns = ['frame', 'energy', 'x', 'y', 'z', 'occupation', 'spin', 'vector']
    _indices = ['orbital']
    _groupbys = ['frame']
    _categories = {'frame': np.int64, 'spin': np.int64}


class MOMatrix(DataFrame):
    '''
    For an atomic nucleus centered at $rx, ry, rz$, a primitive
    Gaussin function takes the form:

    .. math::

        x_{0} = x - rx \\
        y_{0} = y - ry \\
        z_{0} = z - rz \\
        r^{2} = x_{0}^{2} + y_{0}^{2} + z_{0}^{2}
        f(x_{0}, y_{0}, z_{0}; \\alpha, i, j, k) = Nx_{0}^{i}y_{0}^{j}z_{0}^{k}e^{-\\alpha r^{2}}
    '''
    _columns = ['coefficient', 'basis_function', 'orbital']
    _indices = ['momatrix']
    _groupbys = ['orbital', 'basis_function']
    _categories = {'orbital': np.int64, 'basis_function': np.int64, 'spin': np.int64}

    def square(self):
       return self.pivot('basis_function', 'orbital', 'coefficient')

class DensityMatrix(DataFrame):
    '''
    The density matrix in a contracted basis set. As it is
    square symmetric, only n_basis_functions * (n_basis_functions + 1) / 2
    rows are stored.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | chi1              | int      | first basis function                      |
    +-------------------+----------+-------------------------------------------+
    | chi2              | int      | second basis function                     |
    +-------------------+----------+-------------------------------------------+
    | coefficient       | float    | overlap matrix element                    |
    +-------------------+----------+-------------------------------------------+
    '''
    _columns = ['chi1', 'chi2', 'coefficient']
    _indices = ['index']

    def square(self):
        nbas = np.floor(np.sqrt(self.shape[0] * 2))
        return self.pivot('chi1', 'chi2', 'coefficient').fillna(value=0) + \
               self.pivot('chi2', 'chi1', 'coefficient').fillna(value=0) - np.eye(nbas)

    @classmethod
    def from_momatrix(cls, momatrix, nocc):
        square = momatrix.square()
        dens = np.empty(square.shape, dtype=np.float_)
        for mu in square.columns:
            for nu in square.index:
                dens[mu, nu] = np.sum(square.iloc[mu].values[:nocc] *
                                      square.iloc[nu].values[:nocc])
        retlen = square.shape[0] * (square.shape[0] + 1) // 2
        ret = np.empty((retlen,), dtype=[('chi1', 'i8'),
                                         ('chi2', 'i8'),
                                         ('coefficient', 'f8')])
        cnt = 0
        for mu in square.columns:
            for nu in range(mu + 1):
                ret[cnt] = (mu, nu, dens[mu, nu])
                cnt += 1
        return cls(ret)


def _voluminate_cartesian_gtfs(universe, xx, yy, zz):
    '''
    Generate symbolic ordered spherical (in cartesian coordinates)
    Gaussian type function basis for the given universe.

    Args:
        universe: Atomic universe
        xx (array): Discrete values of x
        yy (array): Discrete values of y
        zz (array): Discrete values of z

    Note:
        This function exists here because the resultant functions are orbital
        approximations.
    '''
    ex, ey, ez = sy.symbols('x y z', imaginary=False)
    ordered_spherical_gtf_basis = []
    bases = universe.basis.groupby('symbol')
    lmax = universe.basis['shell'].map(lmap).max()
    sh = solid_harmonics(lmax, True, True)
    for symbol, x, y, z in zip(universe.atom['symbol'], universe.atom['x'],
                               universe.atom['y'], universe.atom['z']):
        bas = bases.get_group(symbol).groupby('function')
        rx = ex - x
        ry = ey - y
        rz = ez - z
        r2 = rx**2 + ry**2 + rz**2
        for f, grp in bas:
            l = lmap[grp['shell'].values[0]]
            shell_functions = {}
            for i, j, k in _cartesian_ordering_function(l):
                function = 0
                for alpha, c in zip(grp['alpha'], grp['c']):
                    function += c * rx**int(i) * ry**int(j) * rz**int(k) * sy.exp(-alpha * r2)
                function = sy.lambdify(('x', 'y', 'z'), function, 'numpy')
                shell_functions['x' * i + 'y' * j + 'z' * k] = function
            # now reduce cart shell functions to spherical funcs
            # Reduce the linearly dependent cartesian basis
            # to a linearly independent spherical basis.
            if l == 0:
                ordered_spherical_gtf_basis.append(shell_functions[''])
            elif l == 1:
                for lv, ml in universe.spherical_gtf_order.symbolic_keys(l):
                    key = str(sh[(lv, ml)])
                    ordered_spherical_gtf_basis.append(shell_functions[key])
            else:
                for lv, ml in universe.spherical_gtf_order.symbolic_keys(l):
                    ordered_spherical_gtf_basis.append(sh[(lv, ml)].subs(shell_functions))
    return ordered_spherical_gtf_basis


def add_cubic_field_from_mo(universe, rmin, rmax, nr, vector=None):
    '''
    Create a cubic field from a given vector (molecular orbital).

    Args:
        universe (:class:`~exatomic.universe.Universe`): Atomic universe
        rmin (float): Starting point for field dimensions
        rmax (float): Ending point for field dimensions
        nr (float): Discretization of the field dimensions
        vector: None, list, or int corresponding to vector index to generate (None will generate all fields)

    Returns:
        fields (list): List of cubic fields corresponding to vectors
    '''
    vectors = universe.momatrix.groupby('orbital')
    if isinstance(vector, int):
        vector = [vector]
    elif vector is None:
        vector = [key for key in vectors.groups.keys()]
    elif not isinstance(vector, list):
        raise TypeError()
    x = np.linspace(rmin, rmax, nr)
    y = np.linspace(rmin, rmax, nr)
    z = np.linspace(rmin, rmax, nr)
    dxi = x[1] - x[0]
    dyj = y[1] - y[0]
    dzk = z[1] - z[0]
    dv = dxi * dyj * dzk
    x, y, z = meshgrid3d(x, y, z)
    basis_funcs = _spherical_gtfs(universe)
    basis_funcs = [sy.lambdify(('x', 'y', 'z'), func, 'numpy') for func in basis_funcs]
    nb = vectorize([float64(float64, float64, float64)], nopython=True)
    basis_funcs = [nb(func) for func in basis_funcs]
    nn = len(basis_funcs)
    n = len(vector)
    # At this point, basis_funcs contains non-normalized ufunc.
    # Now discretize and normalize the basis function values.
    bf_values = np.empty((nn, nr**3), dtype=np.float64)
    for i in range(nn):
        v = basis_funcs[i](x, y, z)
        v /= np.sqrt((v**2 * dv).sum())
        bf_values[i, :] = v
    # Finally, add basis function values to form vectors
    # (normalized molecular orbitals).
    values = np.empty((n, nr**3), dtype=np.float64)
    dxi = [dxi] * n
    dyj = [dyj] * n
    dzk = [dzk] * n
    dxj = [0.0] * n
    dxk = [0.0] * n
    dyi = [0.0] * n
    dyk = [0.0] * n
    dzi = [0.0] * n
    dzj = [0.0] * n
    nx = [nr] * n
    ny = [nr] * n
    nz = [nr] * n
    ox = [rmin] * n
    oy = [rmin] * n
    oz = [rmin] * n
    frame = np.empty((n, ), dtype=np.int64)
    label = np.empty((n, ), dtype=np.int64)
    i = 0
    for vno, vec in vectors:
        if vno in vector:
            frame[i] = universe.orbital.ix[vno, 'frame']
            label[i] = vno
            v = 0
            for c, f in zip(vec['coefficient'], vec['basis_function']):
                v += c * bf_values[f]
            v /= np.sqrt((v**2 * dv).sum())
            values[i, :] = v
            i += 1
    data = pd.DataFrame.from_dict({'dxi': dxi, 'dxj': dxj, 'dxk': dxk, 'dyi': dyi,
                                   'dyj': dyj, 'dyk': dyk, 'dzi': dzi, 'dzj': dzj,
                                   'dzk': dzk, 'nx': nx, 'ny': ny, 'nz': nz, 'label': label,
                                   'ox': ox, 'oy': oy, 'oz': oz, 'frame': frame})
    values = [Series(v) for v in values.tolist()]
    return AtomicField(values, data)
