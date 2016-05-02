# -*- coding: utf-8 -*-
'''
Orbital DataFrame
=============================
Orbital information such as centers and energies.

+-------------------+----------+-------------------------------------------+
| Column            | Type     | Description                               |
+===================+==========+===========================================+
| frame             | int      | associated frame index                    |
+-------------------+----------+-------------------------------------------+
| energy            | float    | orbital energy                            |
+-------------------+----------+-------------------------------------------+
| x                 | float    | orbital center in x                       |
+-------------------+----------+-------------------------------------------+
| y                 | float    | orbital center in y                       |
+-------------------+----------+-------------------------------------------+
| z                 | float    | orbital center in z                       |
+-------------------+----------+-------------------------------------------+
'''
import numpy as np
import pandas as pd
import sympy as sy
from exa import DataFrame


class Orbital(DataFrame):
    '''
    '''
    _columns = ['frame', 'energy', 'x', 'y', 'z', 'occupation']
    _indices = ['vector']
    _groupbys = ['frame']


class MolecularOrbital(Orbital):
    '''
    '''
    _indices = ['mo']


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
    _columns = ['coefficient', 'basis_function', 'vector']
    _indices = ['index']
    _categories = {'vector': np.int64, 'basis_function': np.int64}

    def as_matrix(self):
        '''
        Generate a sparse matrix of molecular orbital coefficients.

        To fill nan values:

        .. code-block:: Python

            C = mo_matrix.as_matrix()
            C.fillna(0, inplace=True)
        '''
        c = self.pivot('vector', 'basis_function', 'coefficient').to_sparse()
        c.index.names = [None]
        c.columns.name = None
        return c


def compute_molecular_orbitals(momatrix, basis_functions):
    '''
    Args:
        momatrix (:class:`~atomic.orbital.MOMatrix`): Molecular orbital matrix
        basis_functions (list): List of symbolic functions
    '''
    x, y, z = sy.symbols('x, y, z', imaginary=False)
    orbitals = []
    for i, orbital in momatrix.groupby('vector'):
        function = 0
        for c, f in zip(orbital['coefficient'], orbital['basis_function']):
            function += c * basis_functions[f]
        #integral = sy.integrate(function**2, (x, -sy.oo, sy.oo), (y, -sy.oo, sy.oo), (z, -sy.oo, sy.oo))
        #function /= sy.sqrt(integral)
        orbitals.append(function)
    return orbitals
