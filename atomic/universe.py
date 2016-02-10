# -*- coding: utf-8 -*-
'''
Universe
====================
The atomic container object.
'''
from traitlets import Unicode, List
from sqlalchemy import Column, Integer, ForeignKey
from exa import Container
from exa import _np as np
from exa.config import Config
from atomic.frame import Frame, minimal_frame, _min_frame_from_atom
from atomic.atom import (Atom, UnitAtom, ProjectedAtom,
                         get_unit_atom, get_projected_atom)
from atomic.two import (Two, ProjectedTwo, AtomTwo, ProjectedAtomTwo,
                        get_two_body)
from atomic.formula import dict_to_string
from atomic.algorithms.pcf import compute_radial_pair_correlation


class Universe(Container):
    '''
    A collection of atoms, molecules, electronic data, and other relevant
    information from an atomistic simulation.

    Information about the data specific dataframes can be found in their class
    documentation.

    See Also:
        :mod:`~atomic.frame`, :mod:`~atomic.atom`, :mod:`~atomic.two`, etc.

    Note:
        If a frame dataframe is not provided, a minimal frame will be created.
    '''
    # Relational information
    cid = Column(Integer, ForeignKey('container.pkid'), primary_key=True)
    frame_count = Column(Integer)
    __mapper_args__ = {'polymorphic_identity': 'universe'}
    __dfclasses__ = {'_frame': Frame, '_atom': Atom, '_unit_atom': UnitAtom,
                     '_prjd_atom': ProjectedAtom, '_two': Two, '_prjd_two': ProjectedTwo,
                     '_atomtwo': AtomTwo, '_prjd_atomtwo': ProjectedAtomTwo}

    # DOMWidget settings
    _view_module = Unicode('nbextensions/exa/atomic/universe').tag(sync=True)
    _view_name = Unicode('UniverseView').tag(sync=True)
    _framelist = List().tag(sync=True)
    _center = Unicode().tag(sync=True)
    _atom_type = Unicode('points').tag(sync=True)
    _bonds = Unicode().tag(sync=True)

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
        df = get_two_body(self, inplace=inplace, **kwargs)
        if inplace == True:
            self._update_bond_list()
        return df

    def compute_frame_mass(self, inplace=False):
        '''
        Compute the mass (of atoms) in each frame and store it in the frame
        dataframe.
        '''
        self.atom.get_element_mass(inplace=True)
        mass = self.atom.groupby('frame').apply(lambda frame: frame['mass'].sum())
        del self._atom['mass']
        if inplace:
            self._frame['mass'] = mass
        else:
            return mass

    def compute_frame_formula(self, inplace=False):
        '''
        Compute the (simple) formula of each frame and store it in the frame
        dataframe.
        '''
        def convert(frame):
            return dict_to_string(frame['symbol'].value_counts().to_dict())
        formulas = self.atom.groupby('frame').apply(convert)
        if inplace:
            self._frame['simple_formula'] = formulas
        else:
            return formulas

    def pair_correlation_function(self, **kwargs):
        '''
        See Also:
            :func:`~atomic.algorithms.pcf.compute_radial_pair_correlation`
        '''
        return compute_radial_pair_correlation(self, **kwargs)

    # DataFrames are "obscured" from the user via properties
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
        if 'prjd_atom' in self._unit_atom.columns:
            atom['prjd_atom'] = self._unit_atom['prjd_atom']
        return Atom(atom.to_dense())

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

    def _update_traits(self):
        '''
        '''
        self._update_df_traits()
        if len(self._frame) > 0:
            self._update_frame_list()
        if len(self._two) > 0 or len(self._prjd_two) > 0:
            self._update_bond_list()
        self._center = self.atom.groupby('frame').apply(lambda x: x[['x', 'y', 'z']].mean().values).to_json()

    def _update_frame_list(self):
        '''
        '''
        self._framelist = [int(f) for f in self._frame.index.values]

    def _update_bond_list(self):
        '''
        Retrieve a Series containing tuples of bonded atom labels
        '''
        if self.is_periodic() and len(self.projected_atom) > 0:
            mapper1 = self.projected_atom['atom']
            mapper2 = self.atom['label']
            df = self.two.ix[(self.two['bond'] == True), ['frame', 'prjd_atom0', 'prjd_atom1']]
            df['atom0'] = df['prjd_atom0'].map(mapper1)
            df['atom1'] = df['prjd_atom1'].map(mapper1)
            df['label0'] = df['atom0'].map(mapper2)
            df['label1'] = df['atom1'].map(mapper2)
            self._bonds = df.groupby('frame').apply(lambda x: x[['label0', 'label1']].values.astype(np.int64)).to_json()
        else:
            mapper = self.atom['label']
            df = self.two.ix[(self.two['bond'] == True), ['frame', 'atom0', 'atom1']]
            df['label0'] = df['atom0'].map(mapper)
            df['label1'] = df['atom1'].map(mapper)
            self._bonds = df.groupby('frame').apply(lambda x: x[['label0', 'label1']].values.astype(np.int64)).to_json()

    def __len__(self):
        return len(self._framelist)

    def __init__(self, frame=None, atom=None, unit_atom=None, prjd_atom=None,
                 two=None, prjd_two=None, atomtwo=None, prjd_atomtwo=None,
                 molecule=None, **kwargs):
        '''
        The universe container represents all of the atoms, bonds, molecules,
        orbital/densities, etc. present within an atomistic simulations. See
        documentation related to each data specific dataframe for more information.
        '''
        super().__init__(**kwargs)
        self._atom = Atom(atom)
        if frame is None:
            self._frame = _min_frame_from_atom(self._atom)
        else:
            self._frame = frame
        self._unit_atom = UnitAtom(unit_atom)
        self._prjd_atom = ProjectedAtom(prjd_atom)
        self._two = Two(two)
        self._prjd_two = ProjectedTwo(prjd_two)
        self._atomtwo = AtomTwo(atomtwo)
        self._prjd_atomtwo = ProjectedAtomTwo(prjd_atomtwo)
        if Config.ipynb:
            self._update_traits()


def concat(universes):
    '''
    Concatenate a collection of universes.
    '''
    raise NotImplementedError()
