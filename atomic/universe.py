# -*- coding: utf-8 -*-
'''
Universe (Container)
======================
Like all secondary (meaning dependent on `exa`_) packages, atomic has a default,
data aware container: the :class:`~atomic.container.Universe`.

Conceptually, a universe is a collection of independent (or related) frames
of a given system. For example, a universe may contain only a
single frame with the geometry of the molecule (system) of interest, a set of
snapshot geometries obtained during the course of a geometry optimization (one
per frame), the same molecule's optimized geometry with each frame containing a
different level of theory, a properties calculation test set of small molecules
with one molecule per frame, a molecular dynamics simulation with each frame
corresponding to a snaphot in time, etc., without restrictions.

.. _exa: http://exa-analytics.github.io/website
'''
import pandas as pd
import numpy as np
from sqlalchemy import Column, Integer, ForeignKey
from exa import Container, _conf, Field3D
from atomic.widget import UniverseWidget
from atomic.frame import minimal_frame, Frame
from atomic.atom import Atom, ProjectedAtom, UnitAtom
from atomic.two import Two, PeriodicTwo
from atomic.field import UniverseField3D
from atomic.atom import compute_unit_atom as _cua
from atomic.atom import compute_projected_atom as _cpa
from atomic.two import max_frames_periodic as mfp
from atomic.two import max_atoms_per_frame_periodic as mapfp
from atomic.two import max_frames as mf
from atomic.two import max_atoms_per_frame as mapf
from atomic.two import compute_two_body as _ctb


class Universe(Container):
    '''
    A container for working with computational chemistry data.

    For a conceptual description of the universe, see the module's docstring.
    '''
    unid = Column(Integer, ForeignKey('container.pkid'), primary_key=True)
    frame_count = Column(Integer)
    _widget_class = UniverseWidget
    __mapper_args__ = {'polymorphic_identity': 'universe'}
    # The arguments here should match those of init (for dataframes)
    _df_types = {'frame': Frame, 'atom': Atom, 'projected_atom': ProjectedAtom,
                 'periodic_two': PeriodicTwo, 'field': UniverseField3D,
                 'unit_atom': UnitAtom}

    @property
    def is_periodic(self):
        return self.frame.is_periodic

    @property
    def is_variable_cell(self):
        return self.frame.is_variable_cell

    @property
    def frame(self):
        if not self._is('_frame'):
            if self._is('_atom'):
                self._frame = minimal_frame(self.atom)
        return self._frame

    @property
    def atom(self):
        return self._atom

    @property
    def unit_atom(self):
        '''
        Updated atom table using only in-unit-cell positions.

        Note:
            This function returns a standard :class:`~pandas.DataFrame`
        '''
        if not self._is('_unit_atom'):
            self.compute_unit_atom()
        atom = self.atom.copy()
        atom.update(self._unit_atom)
        return atom

    @property
    def projected_atom(self):
        '''
        Projected (unit) atom positions into a 3x3x3 supercell.
        '''
        if self._projected_atom is None:
            self.compute_projected_atom()
        return self._projected_atom

    @property
    def two(self):
        if not self._is('_two') and not self._is('_periodic_two'):
            self.compute_two_body()
        elif self._is('_periodic_two'):
            return self._periodic_two
        return self._two

    @property
    def field(self):
        return self._field

    @property
    def field_values(self):
        '''
        Retrieve values of a specific field.

        Args:
            field (int): Field index (corresponding to the fields dataframe)

        Returns:
            data: Series or dataframe object containing field values
        '''
        return self._field.field_values

    def compute_unit_atom(self):
        '''
        Compute the sparse unit atom dataframe.
        '''
        self._unit_atom = _cua(self)

    def compute_projected_atom(self):
        '''
        Compute the projected supercell from the unit atom coordinates.
        '''
        self._projected_atom = _cpa(self)

    def _custom_container_traits(self):
        '''
        Create custom traits using multiple (related) dataframes.
        '''
        traits = {}
        if self._is('_two'):
            traits = self.two._get_bond_traits(self.atom['label'])
        return traits

    def compute_two_body(self, *args, truncate_projected=True, in_mem=False, **kwargs):
        '''
        Compute two body properties for the current universe.

        For arguments see :func:`~atomic.two.get_two_body`. Note that this
        operation (like all compute) operations are performed in place.

        Args:
            truncate_projected (bool): Applicable to periodic universes - decreases the size of the projected atom table
            in_mem (bool): If false, will follow defaults, if true, will force in memory algorithms
        '''
        if self.is_periodic:
            self._periodic_two = _ctb(self, *args, **kwargs)
            if truncate_projected:
                idx0 = self._periodic_two['prjd_atom0']
                idx1 = self._periodic_two['prjd_atom1']
                cls = self._projected_atom.__class__
                df = self._projected_atom[self._projected_atom.index.isin(idx0) |
                                          self._projected_atom.index.isin(idx1)]
                self._projected_atom = cls(df)
        else:
            self._two = _ctb(self, *args, **kwargs)

    def __len__(self):
        return len(self.frame) if self._is('_frame') else 0

    def __init__(self, frame=None, atom=None, two=None, field=None, fields=None,
                 unit_atom=None, projected_atom=None, periodic_two=None,
                 **kwargs):
        '''
        Args:
            frame: Total energies, temperature, cell dimension, ...
            atom: Atomic coordinates, forces, velocties, ...
            two: Interatomic distances, bonds, ...
            field: Field data
        '''
        self._frame = self._enforce_df_type('frame', frame)
        self._atom = self._enforce_df_type('atom', atom)
        self._field = self._reconstruct_field('field', field, fields)
        self._two = self._enforce_df_type('two', two)
        self._unit_atom = self._enforce_df_type('unit_atom', unit_atom)
        self._projected_atom = self._enforce_df_type('projected_atom', projected_atom)
        self._periodic_two = self._enforce_df_type('periodic_two', periodic_two)
        super().__init__(**kwargs)
        ma = self.frame['atom_count'].max() if self._is('_frame') else 0
        nf = len(self)
        if ma == 0 and nf == 0:
            self._test = True
            self.name = 'TestUniverse'
            self._widget.width = 950
            self._widget.gui_width = 350
            self._update_traits()
            self._traits_need_update = False
        elif self.is_periodic and ma < mapfp and nf < mfp and self._atom is not None:
            if self._periodic_two is None:
                self.compute_two_body()
            self._update_traits()
            self._traits_need_update = False
        elif not self.is_periodic and ma < mapf and nf < mf and self._atom is not None:
            if self._two is None:
                self.compute_two_body()
            self._update_traits()
            self._traits_need_update = False
