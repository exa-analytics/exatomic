# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
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
from exa import Field, Series


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
    def nfields(self):
        return len(self.field_values)

    def compute_final(self):
        if not all((col in self.columns for col in ['fx', 'fy', 'fz'])):
            for d, l in [('x', 'i'), ('y', 'j'), ('z', 'k')]:
                self['f'+d] = self['o'+d] + (self['n'+d] - 1) * self['d'+d+l]

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

            newfield = uni.field.rotate(0, 1, np.pi / 2)  # returns an :class:`~exatomic.core.field.AtomicField`
            uni.add_field(newfield)                       # appends new fields to :class:`~exatomic.core.universe.Universe`

        Args:
            a (int): Index of first field
            b (int): Index of second field
            angle (float or list of floats): angle(s) of rotation (in radians)

        Return:
            rotated (:class:`~exatomic.field.AtomicField`): positive then negative linear combinations
        """
        field_params = self.loc[[a]]
        f0 = self.field_values[a]
        f1 = self.field_values[b]
        posvals, negvals = [], []
        try:
            angle = [float(angle)]
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

    def __init__(self, *args, **kwargs):
        if hasattr(args[0], 'field_values') and len(args[0].field_values):
            if 'field_values' in kwargs:
                raise Exception('field_values should not exist as an attached '
                                'attribute and a kwarg at the same time.')
            kwargs['field_values'] = args[0].field_values
        super(AtomicField, self).__init__(*args, **kwargs)
        self['nx'] = self['nx'].astype(np.int64)
        self['ny'] = self['ny'].astype(np.int64)
        self['nz'] = self['nz'].astype(np.int64)
        self.compute_final()
