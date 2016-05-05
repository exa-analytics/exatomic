# -*- coding: utf-8 -*-
'''
Field
============
'''
import numpy as np
from exa.numerical import Field


class AtomicField(Field):
    '''
    Class for storing atomic cube data (scalar field of 3D space). Note that
    this class follows the pattern established by the `cube file format`_.

    Note:
        Supports any shape "cube".

    .. _cube file format: http://paulbourke.net/dataformats/cube/
    '''
    _precision = 6
    _groupbys = ['frame']
    _categories = {'frame': np.int64, 'label': str, 'field_type': str}
    _traits = ['nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'dxi', 'dxj', 'dxk',
               'dyi', 'dyj', 'dyk', 'dzi', 'dzj', 'dzk']
    _columns = ['nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'dxi', 'dxj', 'dxk',
                'dyi', 'dyj', 'dyk', 'dzi', 'dzj', 'dzk', 'frame', 'label']

    def _compute_dv(self, index):
        '''
        Warning
            Only implemented for cubic cubes currently.
        '''
        data = self.ix[[index]]
        u = (data.dxi, data.dxj, data.dxk)
        v = (data.dyi, data.dyj, data.dyk)
        w = (data.dzi, data.dzj, data.dzk)
        return np.dot(u, np.cross(v, w))

    def _integrate(self, index):
        #data = self.ix[[index]]
        dv = self._compute_dv(index)
        return (self.field_values[index] * dv).sum()

    def rotate(self, first, second, angle):
        '''
        Unitary transformation of the discrete field.

        .. code-block:: Python

            myfield.rotate(0, 1, np.pi / 2)

        Args:
            first (int): Index of first field
            second (int): Index of second field
        '''
        # First check that the field have the same dimensions
        data0 = self.ix[[first]]
        data1 = self.ix[[second]]
        #raise NotImplementedError()
        f0 = self.field_values[first]
        f1 = self.field_values[second]
        #dv = data['dv']   # See compute_dv above
        values = np.cos(angle) * f0 + np.sin(angle) * f1
        return self.__class__(values, data0)
