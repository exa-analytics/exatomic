# -*- coding: utf-8 -*-
'''
Field
============
'''
import numpy as np
from exa import Field3D


class UField3D(Field3D):
    '''
    Class for storing atomic cube data (scalar field of 3D space). Note that
    this class follows the pattern established by the `cube file format`_.

    .. _cube file format: http://paulbourke.net/dataformats/cube/
    '''
    _precision = 6
    _groupbys = ['frame']
    _categories = {'frame': np.int64, 'label': str, 'field_type': str}
    _traits = ['nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'dxi', 'dxj', 'dxk',
               'dyi', 'dyj', 'dyk', 'dzi', 'dzj', 'dzk']
    _columns = ['nx', 'ny', 'nz', 'ox', 'oy', 'oz', 'dxi', 'dxj', 'dxk',
                'dyi', 'dyj', 'dyk', 'dzi', 'dzj', 'dzk', 'frame', 'label', 'field_type']
