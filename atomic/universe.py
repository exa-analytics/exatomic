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
from exa import Container, _conf
from atomic.widget import UniverseWidget
from atomic.frame import minimal_frame
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
    def field(self):
        return self._field

    @property
    def two(self):
        if not self._is('_two') and not self._is('_periodic_two'):
            self.compute_two_body()
        elif self._is('_periodic_two'):
            return self._periodic_two
        return self._two

    def field_values(self, field):
        '''
        Retrieve values of a specific field.

        Args:
            field (int): Field index (corresponding to the fields dataframe)

        Returns:
            data: Series or dataframe object containing field values
        '''
        return self._field._fields[field]

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

    def compute_two_body(self, *args, truncate_projected=True, **kwargs):
        '''
        Compute two body properties for the current universe.

        For arguments see :func:`~atomic.two.get_two_body`. Note that this
        operation (like all compute) operations are performed in place.

        Args:
            truncate_projected (bool): Applicable to periodic universes - decreases the size of the projected atom table
        '''
        if self.is_periodic:
            self._periodic_two = _ctb(self, *args, **kwargs)
            if truncate_projected:
                idx0 = self._periodic_two['prjd_atom0']
                idx1 = self._periodic_two['prjd_atom1']
                self._projected_atom = self._projected_atom[self._projected_atom.index.isin(idx0) |
                                                            self._projected_atom.index.isin(idx1)]
        else:
            self._two = _ctb(self, *args, **kwargs)

    def __len__(self):
        return len(self.frame) if self._is('_frame') else 0

    def __init__(self, frame=None, atom=None, two=None, field=None, fields=None,
                 unit_atom=None, projected_atom=None, periodic_two=None,
                 **kwargs):
        self._frame = frame
        self._atom = atom
        self._field = field        # Dataframe containing field dimensions.
        self._fields = fields      # List of field values (Series or DataFrame
        self._two = two            # objects), with index corresponding to _field.
        self._unit_atom = unit_atom
        self._projected_atom = projected_atom
        self._periodic_two = periodic_two
        super().__init__(**kwargs)
        ma = self.frame['atom_count'].max() if self._is('_frame') else 0
        nf = len(self)
        if ma == 0 and nf == 0:
            self._test = True
            self.name = 'TestUniverse'
            self._update_traits()
            self._traits_need_update = False
        elif self.is_periodic and ma < mapfp and nf < mfp:
            if self._periodic_two is None:
                self.compute_two_body()
            self._update_traits()
            self._traits_need_update = False
        elif not self.is_periodic and ma < mapf and nf < mf:
            if self._two is None:
                self.compute_two_body()
            self._update_traits()
            self._traits_need_update = False


#    def compute_bond_count(self, inplace=False):
#        '''
#        Compute the bond count
#        '''
#        s = self.two.compute_bond_count()
#        if inplace:
#            if self._prjd_atom is None:
#                self._atom['bond_count'] = s
#            elif len(self._prjd_atom) > 0:
#                self._prjd_atom['bond_count'] = s
#                s = pd.Series(s.index).map(self.projected_atom['atom'])
#                self._atom['bond_count'] = s
#            else:
#                self._atom['bond_count'] = s
#        else:
#            return s
#
#    def compute_frame_mass(self, inplace=False):
#        '''
#        Compute the mass (of atoms) in each frame and store it in the frame
#        dataframe.
#        '''
#        self.atom.get_element_mass(inplace=True)
#        mass = self.atom.groupby('frame').apply(lambda frame: frame['mass'].sum())
#        del self._atom['mass']
#        if inplace:
#            self._frame['mass'] = mass
#        else:
#            return mass
#
#    def compute_frame_formula(self, inplace=False):
#        '''
#        Compute the (simple) formula of each frame and store it in the frame
#        dataframe.
#        '''
#        def convert(frame):
#            return dict_to_string(frame['symbol'].value_counts().to_dict())
#        formulas = self.atom.groupby('frame').apply(convert)
#        if inplace:
#            self._frame['simple_formula'] = formulas
#        else:
#            return formulas
#
#    def pair_correlation_function(self, *args, **kwargs):
#        '''
#        Args:
#            a (str): First atom type
#            b (str): Second atom type
#            dr (float): Step size (default 0.1 au)
#            rr (float): Two body sample distance (default 11.3)
#
#        See Also:
#            The compute function :func:`~atomic.algorithms.pcf.compute_radial_pair_correlation`
#            and the two body module :mod:`~atomic.two` are useful in
#            understanding how this function works.
#        '''
#        return compute_radial_pair_correlation(self, *args, **kwargs)
#
#
#    @property
#    def two(self):
#        '''
#        '''
#        if self._two is None and self._prjd_two is None and self._atom is not None:
#            self.compute_two_body(inplace=True)
#            self._traits_need_update = True
#        if self._two is None:                  # TEMPORARY HACK UNTIL NONE v empty DF decided
#            if len(self._prjd_two) > 0:
#                return self._prjd_two
#            else:
#                return self._two
#        else:
#            if len(self._two) > 0:
#                return self._two
#            else:
#                return self._prjd_two
