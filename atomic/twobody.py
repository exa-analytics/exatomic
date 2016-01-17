# -*- coding: utf-8 -*-
'''
Two Body Properties DataFrame
===============================
Two body properties are interatomic distances.
'''
from exa import DataFrame


bond_extra = 0.45
dmin = 0.3
dmax = 12.3


class TwoBody(DataFrame):
    '''
    '''
    __indices__ = ['frame', 'index']
    __columns__ = ['atom1', 'atom2', 'symbols', 'distance']


class SuperTwoBody(DataFrame):
    '''
    Two body properties corresponding to the super cell atoms dataframe.

    See Also:
        :class:`~atomic.atom.SuperAtom`
    '''
    __indices__ = ['frame', 'index']
    __columns__ = ['super_atom1', 'super_atom2', 'symbols', 'distance']
