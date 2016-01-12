# -*- coding: utf-8 -*-
'''
Universe
====================
The atomic container object.
'''
from exa import Container
from exa.relational.base import Column, Integer, ForeignKey
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
    pkid = Column(Integer, ForeignKey('container.pkid'), primary_key=True)
    frame_count = Column(Integer)
    __mapper_args__ = {'polymorphic_identity': 'universe'}

    def __init__(self, one=None, two=None, molecule=None, **kwargs):
        '''
        '''
        super().__init__(**kwargs)
