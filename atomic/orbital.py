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
import sympy as sy
from sympy.parsing.sympy_parser import parse_expr
from sympy.physics.secondquant import KroneckerDelta as kr
from exa import DataFrame, _conf, Series
from exa.algorithms import meshgrid3d
from atomic.field import AtomicField
from atomic.basis import _cartesian_ordering_function, lmap


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
    _groupbys = ['orbital']
    _categories = {'orbital': np.int64, 'basis_function': np.int64, 'spin': np.int64}

    def as_matrix(self, spin=0):
        '''
        Generate a sparse matrix of molecular orbital coefficients.

        To fill nan values:

        .. code-block:: Python

            C = mo_matrix.as_matrix()
            C.fillna(0, inplace=True)
        '''
        df = self
        if 'spin' in self:
            df = df[df['spin'] == spin]
        return df.pivot('vector', 'basis_function', 'coefficient').to_sparse().values


def solid_harmonics(l, return_all=False, standard_symbols=True):
    '''
    Generate a set of spherical solid harmonic functions for a given angular
    momentum.

        >>> solid_harmonics(0)
        {(0, 0): 1}
        >>> solid_harmonics(1, True)
        {(0, 0): 1, (1, -1): y, (1, 0): z, (1, 1): x}

    These are the real solutions to the :math:`L^{2}` and :math:`L_{z}`
    operator's eignevalue problems (the complex solutions are called spherical
    harmonics).

    Args:
        l (int): Orbital angular moment
        return_all (bool): If true, return all computed solid harmonics
        standard_symbols (bool): Convert to standard symbol notation (e.g. x*y => xy)

    Returns:
        functions (dict): Dictionary of (l, ml) keys and symbolic function values
    '''
    x, y, z = sy.symbols('x y z', imaginary=False)
    r2 = x**2 + y**2 + z**2
    desired_l = l
    s = {(0,0): 1}
    for l in range(1, desired_l + 1):
        lminus1 = l - 1 if l >= 1 else 0
        negl = -lminus1 if lminus1 != 0 else 0
        # top
        s[(l, l)] = sy.sqrt(2**kr(lminus1, 0) * (2 * lminus1 + 1) / (2 * lminus1 + 2)) * \
                    (x * s[(lminus1, lminus1)] - (1 - kr(lminus1, 0)) * y * s[(lminus1, negl)])
        # bottom
        s[(l, negl - 1)] = sy.sqrt(2**kr(lminus1, 0) * (2 * lminus1 + 1) / (2 * lminus1 + 2)) * \
                           (y * s[(lminus1, lminus1)] + (1 - kr(lminus1, 0)) * x * s[(lminus1, negl)])
        for m in range(-l, l + 1)[1:-1]:
            lminus2 = lminus1 - 1 if lminus1 - 1 >= 0 else 0
            s_lminus2_m = 0
            if (lminus2, m) in s:
                s_lminus2_m = s[(lminus2, m)]
            s[(l, m)] = ((2 * lminus1 + 1) * z * s[(lminus1, m)] - sy.sqrt((lminus1 + m) * (lminus1 - m)) * \
                         r2 * s_lminus2_m) / sy.sqrt((lminus1 + m + 1) * (lminus1 - m + 1))
    # If true, transform the symbolic notation of things like x*y (which represents dxy)
    # to simply xy (which is also a symbol and therefore possible to manipulate
    # with .subs({})).
    if standard_symbols:
        for i in range(2, desired_l + 1):
            match0 = r'\*'.join(['([xyz])'] * i)
            replace0 = r''.join(['\\' + str(j) for j in range(1, i + 1)])
            match1 = r'([xyz])\*\*(\d+)'
            for k, v in s.items():
                if k[0] == i:
                    expr = re.sub(match0, replace0, str(v.expand()))
                    for arg, count in re.findall(r'([xyz])\*\*(\d+)', expr):
                        count = int(count)
                        f = r''.join([arg[0], ''.join([r'\*'] * count), str(count)])
                        r = r''.join([arg[0]] * count)
                        expr = re.sub(f, r, expr)
                    s[k] = parse_expr(expr)
    if return_all:
        return s
    return {key: value for key, value in s.items() if key[0] == desired_l}


def _spherical_gtfs(universe):
    '''
    Generate symbolic ordered spherical (in cartesian coordinates)
    Gaussian type function basis for the given universe.

    Note:
        This function exists here because the resultant functions are orbital
        approximations.
    '''
    ex, ey, ez = sy.symbols('x y z', imaginary=False)
    ordered_spherical_gtf_basis = []
    bases = universe.basis.groupby('symbol')
    lmax = universe.basis['shell'].map(lmap).max()
    sh = solid_harmonics(lmax, True)
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
                shell_functions['x' * i + 'y' * j + 'z' * k] = function
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
        universe (:class:`~atomic.universe.Universe`): Atomic universe
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
    if _conf['pkg_numba']:
        from numba import vectorize, float64
        nb = vectorize([float64(float64, float64, float64)], nopython=True)
        basis_funcs = [nb(func) for func in basis_funcs]
    else:
        basis_funcs = [np.vectorize(func) for func in basis_funcs]
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
