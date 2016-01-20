# -*- coding: utf-8 -*-
'''
Tools
====================
Require internal (atomic) imports.
'''
from exa.errors import MissingColumns
from exa.relational.base import create_all


def initialize_database():
    '''
    '''
    create_all()


def check(universe):
    '''
    '''
    rfc = ['rx', 'ry', 'rz', 'ox', 'oy', 'oz']    # Required columns in the Frame table for periodic calcs
    if 'periodic' in universe.frames.columns:
        if any(universe.frames['periodic'] == True):
            missing = set(rfc).difference(universe.frames.columns)
            if missing:
                raise MissingColumns(missing, universe.frames.__class__.__name__)
            return True
    return False
