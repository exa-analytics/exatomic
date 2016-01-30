# -*- coding: utf-8 -*-
'''
Universe
====================
The atomic container object.
'''
from traitlets import Unicode, List
from sqlalchemy import Column, Integer, ForeignKey
from exa import Container
from exa.config import Config
from atomic.frame import Frame, minimal_frame
from atomic.atom import Atom, UnitAtom, ProjectedAtom, get_unit_atom, get_projected_atom
from atomic.two import Two, ProjectedTwo, AtomTwo, ProjectedAtomTwo, get_two_body


class Universe(Container):
    '''
    A collection of atoms, molecules, electronic data, and other relevant
    information from an atomistic simulation.
    '''
    # Relational information
    cid = Column(Integer, ForeignKey('container.pkid'), primary_key=True)
    frame_count = Column(Integer)
    __mapper_args__ = {'polymorphic_identity': 'universe'}
    __dfclasses__ = {'_frame': Frame, '_atom': Atom, '_unit_atom': UnitAtom}

    # DOMWidget settings
    _view_module = Unicode('nbextensions/exa/atomic/universe').tag(sync=True)
    _view_name = Unicode('UniverseView').tag(sync=True)
    _framelist = List().tag(sync=True)
    _atom_type = Unicode('points').tag(sync=True)

    def is_variable_cell(self):
        '''
        Variable cell universe?
        '''
        return self.frame.is_variable_cell()

    def is_periodic(self):
        return self.frame.is_periodic()

    def compute_minimal_frame(self, inplace=False):
        '''
        '''
        return minimal_frame(self.atom, inplace)

    def compute_cell_magnitudes(self, inplace=False):
        '''
        '''
        return self._frame.get_unit_cell_magnitudes(inplace)

    def compute_unit_atom(self, inplace=False):
        '''
        '''
        return get_unit_atom(self, inplace)

    def compute_projected_atom(self, inplace=False):
        '''
        '''
        return get_projected_atom(self, inplace)

    def compute_two_body(self, inplace=False, **kwargs):
        '''
        '''
        return get_two_body(self, inplace=inplace, **kwargs)

    # DataFrame properties
    @property
    def frame(self):
        if len(self._frame) == 0 and len(self._atom) > 0:
            self.compute_minimal_frame(inplace=True)
            self._framelist = self._frame.index.tolist()
        return self._frame

    @property
    def atom(self):
        return self._atom

    @property
    def unit_atom(self):
        '''
        Primitive atom positions.
        '''
        if len(self._unit_atom) == 0:
            self.compute_unit_atom(inplace=True)
        atom = self.atom.copy()
        atom.update(self._unit_atom)
        return Atom(atom)

    @property
    def projected_atom(self):
        '''
        Projected unit atom coordinates generating a 3x3x3 super cell.
        '''
        if len(self._prjd_atom) == 0:
            self.compute_projected_atom(inplace=True)
        return self._prjd_atom

    @property
    def two(self):
        '''
        '''
        if len(self._two) == 0 and len(self._prjd_two) == 0:
            self.compute_two_body(inplace=True)
        if len(self._two) == 0:
            return self._prjd_two
        else:
            return self._two

    def __len__(self):
        return len(self._framelist)

    def __init__(self, frame=None, atom=None, unit_atom=None, prjd_atom=None,
                 two=None, prjd_two=None, atomtwo=None, prjd_atomtwo=None,
                 **kwargs):
        '''
        The universe container represents all of the atoms, bonds, molecules,
        orbital/densities, etc. present within an atomistic simulations.
        '''
        super().__init__(**kwargs)
        self._frame = Frame(frame)
        self._atom = Atom(atom)
        self._unit_atom = UnitAtom(unit_atom)
        self._prjd_atom = ProjectedAtom(prjd_atom)
        self._two = Two(two)
        self._prjd_two = ProjectedTwo(prjd_two)
        self._atomtwo = AtomTwo(atomtwo)
        self._prjd_atomtwo = ProjectedAtomTwo(prjd_atomtwo)
        if Config.ipynb:
            self._update_all_traits()
        self._framelist = []
        if len(self._frame) > 0:
            self._framelist = [int(f) for f in self._frame.index.values]


def concat(universes):
    '''
    Concatenate a collection of universes.
    '''
    raise NotImplementedError()
