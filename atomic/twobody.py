# -*- coding: utf-8 -*-
'''
Two Body Properties DataFrame
===============================
Two body properties are interatomic distances.
'''
from exa import DataFrame


class TwoBody(DataFrame):
    '''
    '''
    __dimensions__ = ['frame', 'index']
    __columns__ = ['atom1', 'atom2', 'symbols', 'distance']
