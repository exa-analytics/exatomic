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
from atomic.atom import Atom, UnitAtom, ProjectedAtom
from atomic.atom import _compute_projected_non_var_cell
from atomic.atom import get_unit_atom as _get_unit_atom


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

    def compute_cell_magnitudes(self, inplace=True):
        '''
        See Also:
            :func:`~atomic.atom.Atom.get_unit_cell_magnitudes`
        '''
        return self._frame.get_unit_cell_magnitudes(inplace)

    # DataFrame properties
    @property
    def frame(self):
        if len(self._frame) == 0 and len(self._atom) > 0:
            self._frame = minimal_frame(self.atom)
        if self._framelist == []:
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
            self._unit_atom = _get_unit_atom(self)
        atom = self.atom.copy()
        atom.update(self._unit_atom)
        return atom

    @property
    def prjd_atom(self):
        '''
        '''
        if len(self._prjd_atom) == 0:
            self._prjd_atom = _compute_projected_non_var_cell(self)
        return self._prjd_atom

    def __len__(self):
        return len(self._framelist)

    def __init__(self, frame=None, atom=None, unit_atom=None, prjd_atom=None,
                 two=None, **kwargs):
        '''
        The universe container represents all of the atoms, bonds, molecules,
        orbital/densities, etc. present within an atomistic simulations.
        '''
        super().__init__(**kwargs)
        self._frame = Frame(frame)
        self._atom = Atom(atom)
        self._unit_atom = UnitAtom(unit_atom)
        self._prjd_atom = ProjectedAtom(prjd_atom)
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
