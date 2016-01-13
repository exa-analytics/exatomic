# -*- coding: utf-8 -*-
'''
Atom DataFrame
==========================
'''
from exa import DataFrame


class Atom(DataFrame):
    '''
    Required indexes:
        frame, atom

    Required columns:
        symbol, x, y, z

    Optional columns:
        lots
    '''
    __dimensions__ = ['frame', 'atom']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
