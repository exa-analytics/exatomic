# -*- coding: utf-8 -*-
'''
Atom DataFrame
==========================
'''
from exa import DataFrame


class Atom(DataFrame):
    '''
    Required indexes: frame, atom

    Required columns: symbol, x, y, z
    '''
    __dimensions__ = ['frame', 'atom']
    __columns__ = ['symbol', 'x', 'y', 'z']
