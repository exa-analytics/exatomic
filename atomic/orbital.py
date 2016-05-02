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
from exa import DataFrame


class Orbital(DataFrame):
    '''
    '''
    _columns = ['frame', 'energy', 'x', 'y', 'z', 'occupation']
    _indices = ['orbital']
    _groupbys = ['frame']


class MolecularOrbital(Orbital):
    '''
    '''
    _indices = ['mo']


class OrbitalCoefficient(DataFrame):
    '''
    Store information about primitive Gaussian functions.

    For an atomic nucleus centered at $rx, ry, rz$, a primitive
    Gaussin function takes the form:

    .. math::

        x_{0} = x - rx \\
        y_{0} = y - ry \\
        z_{0} = z - rz \\
        r^{2} = x_{0}^{2} + y_{0}^{2} + z_{0}^{2}
        f(x_{0}, y_{0}, z_{0}; \\alpha, i, j, k) = Nx_{0}^{i}y_{0}^{j}z_{0}^{k}e^{-\\alpha r^{2}}
    '''
    _columns = ['alpha', 'coefficient', 'basis_function',
                'shell', 'symbol']
    _indices = ['primitive']
    _categories = {'symbol': str, 'basis_function': np.int64}
