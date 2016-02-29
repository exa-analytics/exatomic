# -*- coding: utf-8 -*-
'''
Field DataFrame
======================
A dataframe storing scalar field data.
'''
from exa.frames import DataFrame

class FieldMeta(DataFrame):
    '''
    Values of a field in a single column.
    '''
    __pk__ = ['field']
    __fk__ = ['frame']
    __traits__ = [
        'ox', 'oy', 'oz',
        'nx', 'dxi', 'dxj', 'dxk',
        'ny', 'dyi', 'dyj', 'dyk',
        'nz', 'dzi', 'dzj', 'dzk',
        'label', 'frame',
    ]
    __groupby__ = 'frame'
