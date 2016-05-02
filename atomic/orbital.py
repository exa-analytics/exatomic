# -*- coding: utf-8 -*-
'''
Orbital DataFrame
==========================
A dataframe containing orbital information from a quantum chemical calculation.

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

See Also:
    :class:`~atomic.universe.Universe`
'''
import numpy as np
import pandas as pd
from exa import DataFrame

class Orbital(DataFrame):
    '''
    Eigenvector of Hilbert space, in the general sense.
    '''
    _indices = ['vector']
    _groupbys = ['frame']
    _categories = {'symmetry': object}
