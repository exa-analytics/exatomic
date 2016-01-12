# -*- coding: utf-8 -*-
'''
Universe
====================
The atomic container object.
'''
from exa.relational.container import Container, pkid, mapper_args
#from atomic.one import One
#from atomic.two import Two
#from atomic.molecule import Molecule


class Universe(Container):
    '''
    An :class:`~atomic.universe.Universe` represents a collection of time
    dependent frames themselves containing atoms and molecules. A frame can
    be thought of as a snapshot in time. Each snaphot in time has information
    about atomic positions, energies, bond distances, etc.
    '''
    pkid = pkid
    __mapper_args__ = mapper_args

    def __init__(self, one=None, two=None, molecule=None, **kwargs):
        '''
        '''
        super().__init__(**kwargs)
        self.one = one
        self.two = two
        self.molecule = molecule
