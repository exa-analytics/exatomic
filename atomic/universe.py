# -*- coding: utf-8 -*-
'''
Universe
====================
The atomic container object.
'''
from exa import Container
from exa.relational.base import Column, Integer, ForeignKey
from atomic.atom import Atom
#from atomic.two import Two
#from atomic.molecule import Molecule


class Universe(Container):
    '''
    A collection of atoms.

    An :class:`~atomic.universe.Universe` represents a collection of time
    dependent frames themselves containing atoms and molecules. A frame can
    be thought of as a snapshot in time, though the frame axis is not required
    to be time. Each frame has information about atomic positions, energies,
    bond distances, energies, etc. The following table outlines the structures
    provided by this container (specifics can be found on the relevant dataframe
    pages). The only required dataframe is the :class:`~atomic.atom.Atom` dataframe.

    +------------------------------------+-------------+------------------+
    | Attribute (DataFrame)              | Dimensions  | Required Columns |
    +====================================+=============+==================+
    | atoms (:class:`~atomic.atom.Atom`) | frame, atom | symbol, x, y, z  |
    +------------------------------------+-------------+------------------+


    .. Tip:: The only required :class:`~exa.dataframe.DataFrame` is :class:`~atomic.atom.Atom`.
    '''
    cid = Column(Integer, ForeignKey('container.pkid'), primary_key=True)
    frame_count = Column(Integer)
    __mapper_args__ = {'polymorphic_identity': 'universe'}

    def __init__(self, atoms=None, two=None, molecule=None, **kwargs):
        '''
        '''
        super().__init__(**kwargs)
        self.atoms = Atom(atoms)
#        self.two = Two(two)
        #self.molecule = Molecule(molecule)
