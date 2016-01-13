# -*- coding: utf-8 -*-
'''
One Body DataFrame
==========================
Description of the one body dataframe (dataframe containing the nuclear
types and coordinates) along with some utilities related to this dataframe.
'''
from exa import DataFrame


class One(DataFrame):
    '''
    Required indexes:
        frame, atom

    Optional indexes:
        molecule

    Required columns:
        symbol, x, y, z

    Optional columns:
        lots
    '''
    pass
