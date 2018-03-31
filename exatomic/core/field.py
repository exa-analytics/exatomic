# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Atomic Field
#####################
Essentially a :class:`~exa.numerical.Field3D` used for storing cube file data
(see :mod:`~exatomic.filetypes.cube`). Cube files values are written in a
csv-like structure with the outer loop going over the x dimension, the middle
loop going over the y dimension, and the inner loop going over the z dimension.
"""
import numpy as np
import pandas as pd
from exa import DataFrame, Field, Series

class AField(DataFrame):
    _columns = ['nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'dxi', 'dxj', 'dxk',
                'dyi', 'dyj', 'dyk', 'dzi', 'dzj', 'dzk', 'frame']
    @property
    def _constructor(self):
        return AField

    def copy(self, *args, **kwargs):
        """Make a copy of this object."""
        cls = self.__class__
        df = pd.DataFrame(self).copy(*args, **kwargs)
        data = self.data.copy()
        return cls(data, data=data)

    def memory_usage(self):
        """Get the total memory usage."""
        data = super(Field, self).memory_usage()
        data['data'] = self.data.memory_usage()
        return data

    def __init__(self, *args, **kwargs):
        data = kwargs.pop("data", np.array([]))
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be an ndarray")
        if isinstance(args[0], pd.Series): args = (args[0].to_frame().T,)
        super(AField, self).__init__(*args, **kwargs)
        self.data = data




class AtomicField(Field):
    """
    Class for storing exatomic cube data (scalar field of 3D space). Note that
    this class follows the pattern established by the `cube file format`_.

    Note:
        Supports any shape "cube".

    .. _cube file format: http://paulbourke.net/dataformats/cube/
    """
    _cardinal = ('frame', np.int64)
    _categories = {'label': str, 'field_type': str}
    _columns = ['nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'dxi', 'dxj', 'dxk',
                'dyi', 'dyj', 'dyk', 'dzi', 'dzj', 'dzk', 'frame']

    @property
    def _constructor(self):
        return AtomicField

    @property
    def nfields(self):
        return len(self.field_values)

    def compute_dv(self):
        """
        Compute the volume element for each field.

        Volume of a parallelpiped whose dimensions are :math:`\mathbf{a}`,
        :math:`\mathbf{b}`, :math:`\mathbf{c}` is given by:

        .. math::

            v = \\left|\\mathbf{a}\\cdot\\left(\\mathbf{b}\\times\\mathbf{c}\\right)\\right|
        """
        def _dv(row):
            """
            Helper function that performs the operation above.
            """
            a = row[['dxi', 'dxj', 'dxk']].values.astype(np.float64)
            b = row[['dyi', 'dyj', 'dyk']].values.astype(np.float64)
            c = row[['dzi', 'dzj', 'dzk']].values.astype(np.float64)
            return np.dot(a, np.cross(b, c))
        self['dv'] = self.apply(_dv, axis=1)

    def integrate(self):
        """
        Check that field values are normalized.

        Computes the integral of the field values. For normalized fields (e.g
        orbitals), the result should be 1.

        .. math::

            \\int\\left|\\phi_{i}\\right|^{2}dV \equiv 1
        """
        if 'dv' not in self:
            self.compute_dv()
        self['sums'] = [np.sum(fv**2) for fv in self.field_values]
        norm = self['dv'] * self['sums']
        del self['sums']
        return norm

    def rotate(self, a, b, angle):
        """
        Unitary transformation of the discrete field.

        .. code-block:: Python

            newfield = myfield.rotate(0, 1, np.pi / 2)

        Args:
            a (int): Index of first field
            b (int): Index of second field
            angle (float or list of floats): angle(s) of rotation

        Return:
            rotated (:class:`~exatomic.field.AtomicField`): positive then negative linear combinations
        """
        field_params = self.ix[[a]]
        f0 = self.field_values[a]
        f1 = self.field_values[b]
        posvals, negvals = [], []
        try:
            angle = float(angle)
            angle = [angle]
        except TypeError:
            pass
        for ang in angle:
            t1 = np.cos(ang) * f0
            t2 = np.sin(ang) * f1
            posvals.append(Series(t1 + t2))
            negvals.append(Series(t1 - t2))
        num = len(posvals) + len(negvals)
        field_params = pd.concat([field_params] * num)
        field_params.reset_index(drop=True, inplace=True)
        return AtomicField(field_params, field_values=posvals + negvals)
