# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Matrices
#############
For handling matrices of common dimensionality in QM calcs.
"""

import numpy as np
import pandas as pd
from exa import DataFrame
from numba import jit


@jit(nopython=True, cache=True)
def _symmetric_from_square(square):
    ndim = len(square)
    newdim = ndim * (ndim + 1) // 2
    idxs = np.empty((newdim, 3), dtype=np.int64)
    arr = np.empty(newdim, dtype=np.float64)
    cnt = 0
    for i in range(ndim):
        for j in range(i + 1):
            idxs[cnt,:] = (i, j, 0)
            arr[cnt] = square[i, j]
            cnt += 1
    return idxs, arr

@jit(nopython=True, cache=True)
def _symmetric_to_square(ix0, ix1, values):
    ndim = int((-1 + np.sqrt(1 + 8 * len(values))) / 2)
    square = np.empty((ndim, ndim), dtype=np.float64)
    for i, j, val in zip(ix0, ix1, values):
        square[i, j] = val
        square[j, i] = val
    return square

@jit(nopython=True, cache=True)
def _square_from_square(square):
    ndim = len(square)
    newdim = ndim ** 2
    idxs = np.empty((newdim, 3), dtype=np.int64)
    arr = np.empty(newdim, dtype=np.float64)
    cnt = 0
    for i in range(ndim):
        for j in range(ndim):
            idxs[cnt,:] = (i, j, 0)
            arr[cnt] = square[i, j]
            cnt += 1
    return square

@jit(nopython=True, cache=True)
def _square_to_square(ix0, ix1, values):
    ndim = np.int64(len(values) ** 0.5)
    square = np.empty((ndim, ndim), dtype=np.float64)
    for i, j, val in zip(ix0, ix1, values):
        square[i, j] = val
    return square

#
#   Basically the only thing that is exatomic specific currently
#   is that _symmetric_from_square, _square_from_square return
#   indices including 'frame' and the from_square methods also
#   include 'frame' in the generation of the DataFrames.
#

class _Matrix(DataFrame):
    """
    Base class for square and symmetric matrices stored in
    a DataFrame format with matrix indices as columns.
    """
    _columns = ['idx0', 'idx1']
    _index = 'index'
    _categories = {'frame': np.int64}

    #@property
    #def _constructor(self):
    #    return _Matrix

    @property
    def indices(self):
        return self._columns[:2]

    @property
    def defaultcolumn(self):
        if not hasattr(self, '_data'): return
        return self.columns_to_order[0]

    @property
    def columns_to_order(self):
        if not hasattr(self, '_data'): return
        notcols = set(self.indices).union(self._categories.keys())
        return list(set(self.columns).difference(notcols))

class _Symmetric(_Matrix):
    """Base class for symmetric matrices."""
    #@property
    #def _constructor(self):
    #    return _Symmetric

    def square(self, column=None):
        """Return a square DataFrame of the matrix."""
        column = self.defaultcolumn if column is None else column
        idx0, idx1 = self.indices
        ret = pd.DataFrame(_symmetric_to_square(self[idx0].values,
                                                self[idx1].values,
                                                self[column.values]))
        ret.index.name = idx0
        ret.columns.name = idx1
        return ret

    @classmethod
    def from_square(cls, square, column=None):
        """Create a symmetric matrix DataFrame from a square array."""
        column = 'coef' if column is None else column
        idx0, idx1 = cls().indices
        if isinstance(square, pd.DataFrame):
            square = square.values
        idxs, arr = _symmetric_from_square(square)
        return cls(pd.DataFrame.from_dict({idx0: idxs[:,0],
                                           idx1: idxs[:,1],
                                        'frame': idxs[:,2],
                                         column: arr}))


class _Square(_Matrix):
    """Base class for square matrices."""
    #@property
    #def _constructor(self):
    #    return _Square

    def square(self, column=None):
        """Return a square DataFrame of the square matrix."""
        column = self.defaultcolumn if column is None else column
        idx0, idx1 = self.indices
        ret = pd.DataFrame(_square_to_square(self[idx0].values,
                                             self[idx1].values,
                                             self[column].values))
        ret.index.name = idx0
        ret.columns.name = idx1
        return ret

    @classmethod
    def from_square(cls, square, column=None):
        """Create a square matrix DataFrame from a square array."""
        column = 'coef' if column is None else column
        idx0, idx1 = cls().indices
        if isinstance(square, pd.DataFrame):
            square = square.values
        idxs, arr = _square_from_square(square)
        return cls(pd.DataFrame.from_dict({idx0: idxs[:,0],
                                           idx1: idxs[:,1],
                                        'frame': idxs[:,2],
                                         column: arr}))

#
# This is mainly exatomic specific but I suppose the merge
# could be useful in the general case.
#

class Triangle(_Symmetric):
    """
    Triangular matrices having dimensions N * (N + 1) // 2
    sharing natural indices all belong in the same table.
    As applied to exatomic that includes but is not limited to:

        - one-electron integrals (overlap, kinetic, nuclear attraction energies,
                                  multipole integrals, etc.)

    Note:
        When parsing matrices like these, it may be more efficient
        to have indices chi0 and chi1 not in the same order as they
        may be for other matrices of the same dimensions. This can
        be dangerous if one then adds a column expecting different
        index ordering. Therefore, it is recommended to only add
        columns to this table by using the merge bound method and
        instances of this class.
    """
    _columns = ['chi0', 'chi1']

    #@property
    #def _constructor(self):
    #    return Triangle

    def merge(self, other, column=None):
        """
        Correctly adds a column, ensuring that if index ordering
        is not the same between two instances, that it is remedied.
        """
        if not isinstance(other, Triangle):
            other = Triangle(other)
        idx0, idx1 = self.indices
        ocol = other.defaultcolumn if column is None else column
        if ocol in self.columns:
            raise Exception("column {} is already in self.".format(ocol))
        # The easy case when both sets of indices match
        if np.allclose(self[idx0], other[idx0]) \
        and np.allclose(self[idx1], self[idx1]):
            self[ocol] = other[ocol]
            return
        # The sort of easy case when self's indices are already
        # in the same order that comes out of the numba func
        osquare = other.square(column=ocol)
        oidxs, oarr = _symmetric_from_square(osquare.values)
        if np.allclose(self[idx0], oidxs[:,0]) \
        and np.allclose(self[idx1], oidxs[:,1]):
            self[ocol] = oarr
            return
        # The case when self is not ordered correctly
        # relative to the ordering of the numba func or other
        for i, col in enumerate(self.columns_to_order):
            square = self.square(column=col)
            idxs, arr = _symmetric_from_square(square.values)
            if not i:
                self[idx0] = idxs[:,0]
                self[idx1] = idxs[:,1]
                self[ocol] = oarr
            self[col] = arr
