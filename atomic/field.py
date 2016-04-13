# -*- coding: utf-8 -*-
'''
Field
============
'''
import numpy as np
from exa import Field3D


class UField3D(Field3D):
    '''
    Class for storing atomic cube data (scalar field of 3D space).
    '''
    _precision = 6
    _columns = Field3D._columns + ['frame', 'label']
    _groupbys = ['frame']
    _categories = {'frame': np.int64, 'label': str}
