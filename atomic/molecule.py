# -*- coding: utf-8 -*-
'''
Molecule Information DataFrame
===============================
Molecules are collections of bonded atoms.
'''
from exa import DataFrame


class Molecule(DataFrame):
    '''
    '''
    __indices__ = ['frame', 'molecule']
    __columns__ = ['formula', 'mass', 'cx', 'cy', 'cz']


class PeriodicMolecule(DataFrame):
    '''
    '''
    __indices__ = ['frame', 'moleculepbc']
    __columns__ = ['cx', 'cy', 'cz', 'molecule']
