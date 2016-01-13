# -*- coding: utf-8 -*-
'''
Tools
====================
Require internal (atomic) imports.
'''
from exa.relational.base import create_all


def initialize_database():
    '''
    '''
    create_all()
