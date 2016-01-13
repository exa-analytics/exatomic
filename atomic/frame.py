# -*- coding: utf-8 -*-
'''
Atom DataFrame
==========================
'''
from exa import DataFrame


class Frame(DataFrame):
    '''
    Required indexes: frame
    Required columns: atom_count
    '''
    __dimensions__ = ['frame']
    __columns__ = ['atom_count']


def minimal_frame_from_atoms(atoms):
    '''
    '''
    df = atoms.groupby(level='frame')['symbol'].count().to_frame()
    df.columns = ['atom_count']
    return df
