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
    def periodic(self):
        return self.frame.is_periodic()

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
    def field(self):
        return self._field

    @property
    def two(self):
        if not self._is('_two'):
            self.compute_two_body()
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

    def compute_two_body(self, *args, **kwargs):
        '''
        Compute two body properties for the current universe.

        For arguments see :func:`~atomic.two.get_two_body`. Note that this
        operation (like all compute) operations are performed in place.
        '''
        self._two = _ctb(self, *args, **kwargs)

    def _custom_container_traits(self):
        '''
        Create custom traits using multiple (related) dataframes.
        '''
        traits = {}
        if self._is('_two'):
            traits = self.two._get_bond_traits(self.atom['label'])
        return traits

    def __init__(self, frame=None, atom=None, two=None, field=None, **kwargs):
        self._frame = frame
        self._atom = atom
        self._field = field
        self._two = two
        super().__init__(**kwargs)
        if self.frame['atom_count'].max() < 2000 and len(self.frame) < 200:
            self.compute_two_body()
            self._update_traits()

#    def is_variable_cell(self):
#        '''
#        Variable cell universe?
#        '''
#        return self.frame.is_variable_cell()
#
#
#    def compute_minimal_frame(self, inplace=False):
#        '''
#        '''
#        return minimal_frame(self.atom, inplace)
#
#    def compute_cell_magnitudes(self, inplace=False):
#        '''
#        '''
#        return self._frame.get_unit_cell_magnitudes(inplace)
#
#    def compute_unit_atom(self, inplace=False):
#        '''
#        '''
#        return get_unit_atom(self, inplace)
#
#    def compute_projected_atom(self, inplace=False):
#        '''
#        '''
#        return get_projected_atom(self, inplace)
#
#
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
#    # DataFrames are "obscured" from the user via properties
#    @property
#    def frame(self):
#        if self._frame is None:
#            self.compute_minimal_frame(inplace=True)
#            self._framelist = self._frame.index.tolist()
#        return self._frame
#
#    @property
#    def atom(self):
#        return self._atom
#
#    @property
#    def unit_atom(self):
#        '''
#        Primitive atom positions.
#        '''
#        if self._unit_atom is None:
#            self.compute_unit_atom(inplace=True)
#        atom = self.atom.copy()
#        atom.update(self._unit_atom)
#        if 'prjd_atom' in self._unit_atom.columns:
#            atom['prjd_atom'] = self._unit_atom['prjd_atom']
#        return Atom(atom.to_dense())
#
#    @property
#    def projected_atom(self):
#        '''
#        Projected unit atom coordinates generating a 3x3x3 super cell.
#        '''
#        if self._prjd_atom is None:
#            self.compute_projected_atom(inplace=True)
#        return self._prjd_atom
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
#
#    @property
#    def fieldmeta(self):
#        return self._fieldmeta
#
#    @property
#    def fields(self):
#        return self._fields
#
#    def _update_traits(self):
#        '''
#        '''
#        st = dt.now()
#        self._update_df_traits()
#        print('df: ', (dt.now() - st).total_seconds())
#        if self.frame is not None:
#            st = dt.now()
#            self._update_frame_list()
#            print('frame: ', (dt.now() - st).total_seconds())
#        if self._two is not None:
#            st = dt.now()
#            self._update_bond_list()
#            print('blist: ', (dt.now() - st).total_seconds())
#        if self._atom is not None:
#            st = dt.now()
#            self._center = self.atom.groupby('frame').apply(lambda x: x[['x', 'y', 'z']].mean().values).to_json()
#            print('center: ', (dt.now() - st).total_seconds())
#
#    def _update_frame_list(self):
#        '''
#        '''
#        self._framelist = [int(f) for f in self._frame.index.values]
#
#    def _update_bond_list(self):
#        '''
#        Retrieve a Series containing tuples of bonded atom labels
#        '''
#        if self.is_periodic() and len(self.projected_atom) > 0:
#            mapper1 = self.projected_atom['atom']
#            mapper2 = self.atom['label']
#            df = self.two.ix[(self.two['bond'] == True), ['frame', 'prjd_atom0', 'prjd_atom1']]
#            df['atom0'] = df['prjd_atom0'].map(mapper1)
#            df['atom1'] = df['prjd_atom1'].map(mapper1)
#            df['label0'] = df['atom0'].map(mapper2)
#            df['label1'] = df['atom1'].map(mapper2)
#            self._bonds = df.groupby('frame').apply(lambda x: x[['label0', 'label1']].values.astype(np.int64)).to_json()
#        else:
#            mapper = self.atom['label']
#            df = self.two.ix[(self.two['bond'] == True), ['frame', 'atom0', 'atom1']]
#            df['label0'] = df['atom0'].map(mapper)
#            df['label1'] = df['atom1'].map(mapper)
#            self._bonds = df.groupby('frame').apply(lambda x: x[['label0', 'label1']].values.astype(np.int64)).to_json()
#
#    def __len__(self):
#        return len(self.frame)
#
#    def __init__(self, frame=None, atom=None, unit_atom=None, prjd_atom=None,
#                 two=None, prjd_two=None, atomtwo=None, prjd_atomtwo=None,
#                 molecule=None, fieldmeta=None, fields=[], **kwargs):
#        '''
#        The universe container represents all of the atoms, bonds, molecules,
#        orbital/densities, etc. present within an atomistic simulation. See
#        documentation related to each data specific dataframe for more information.
#        '''
#        super().__init__(**kwargs)
#        self._atom = atom
#        if frame is None:
#            self._frame = _min_frame_from_atom(self._atom)
#        else:
#            self._frame = frame
#        self._unit_atom = unit_atom
#        self._prjd_atom = prjd_atom
#        self._two = two
#        self._prjd_two = prjd_two
#        self._atomtwo = atomtwo
#        self._prjd_atomtwo = prjd_atomtwo
#        self._fieldmeta = fieldmeta
#        self._fieldframes = []
#        if hasattr(self._fieldmeta, 'index'):
#            self._fieldframes = [int(obj) for obj in self._fieldmeta['frame'].values]
#        self._fields = fields
#        #if Config.ipynb and len(self._frame) < 1000:    # TODO workup a better solution here!
#        #    self._update_traits()
#
#
#def concat(universes):
#    '''
#    Concatenate a collection of universes.
#    '''
#    raise NotImplementedError()
#
#
#@event.listens_for(Universe, 'after_insert')
#def after_insert(*args, **kwargs):
#    '''
#    '''
#    print('after_insert')
#    print(args)
#    print(kwargs)
#    return args
#
#@event.listens_for(Universe, 'after_update')
#def after_update(*args, **kwargs):
#    '''
#    '''
#    print('after_update')
#    print(args)
#    print(kwargs)
#    return args
#
