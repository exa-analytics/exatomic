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
    __indices__ = ['frame', 'index']
    __columns__ = ['atom1', 'atom2', 'symbols', 'distance']


class TwoBodyPBC(DataFrame):
    '''
    Two body properties corresponding to the periodic atoms dataframe.

    See Also:
        :class:`~atomic.atom.AtomPBC`
    '''
    __indices__ = ['frame', 'index']
    __columns__ = ['superatom1', 'superatom2', 'symbols', 'distance']
