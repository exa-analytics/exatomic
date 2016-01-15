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
    __dimensions__ = ['frame', 'molecule']
    __columns__ = ['formula', 'mass', 'cx', 'cy', 'cz']


class SuperMolecule(DataFrame):
    '''
    '''
    __dimensions__ = ['frame', 'supermolecule']
    __columns__ = ['cx', 'cy', 'cz', 'molecule']
