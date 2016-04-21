# -*- coding: utf-8 -*-
'''
Atom Dataframes
==========================
A dataframe containing the nuclear positions, forces, velocities, symbols, etc.
Examples of data that may exist in this dataframe are given below (note that
the dataframe is not limited to only these record types - rather this provides
a guide for what type of data is required and can be expected).

+-------------------+----------+-------------------------------------------+
| Column            | Type     | Description                               |
+===================+==========+===========================================+
| x                 | float    | position in x (req.)                      |
+-------------------+----------+-------------------------------------------+
| y                 | float    | position in y (req.)                      |
+-------------------+----------+-------------------------------------------+
| z                 | float    | position in z (req.)                      |
+-------------------+----------+-------------------------------------------+
| frame             | category | non-unique integer (req.)                 |
+-------------------+----------+-------------------------------------------+
| symbol            | category | element symbol (req.)                     |
+-------------------+----------+-------------------------------------------+
| fx                | float    | force in x                                |
+-------------------+----------+-------------------------------------------+
| fy                | float    | force in y                                |
+-------------------+----------+-------------------------------------------+
| fz                | float    | force in z                                |
+-------------------+----------+-------------------------------------------+
| vx                | float    | velocity in x                             |
+-------------------+----------+-------------------------------------------+
| vy                | float    | velocity in y                             |
+-------------------+----------+-------------------------------------------+
| vz                | float    | velocity in z                             |
+-------------------+----------+-------------------------------------------+
| label             | category | non-unique integer                        |
+-------------------+----------+-------------------------------------------+

See Also:
    :class:`~atomic.universe.Universe`
'''
import numpy as np
import pandas as pd
from traitlets import Dict, Unicode
from exa import DataFrame, SparseDataFrame
from exa.algorithms import supercell3
from atomic import Isotope


class BaseAtom(DataFrame):
    '''
    Base atom and related datframe.
    '''
    _precision = 2
    _indices = ['atom']
    _columns = ['x', 'y', 'z', 'symbol', 'frame']
    _groupbys = ['frame']
    _categories = {'frame': np.int64, 'label': np.int64, 'symbol': str}

    def _custom_trait_creator(self):
        '''
        Custom trait creator function because traits from the atom table are
        not automatically created via exa.numerical.
        '''
        grps = self.groupby('frame')
        symbols = grps.apply(lambda g: g['symbol'].cat.codes.values)
        symbols = Unicode(symbols.to_json(orient='values')).tag(sync=True)
        symmap = {i: v for i, v in enumerate(self['symbol'].cat.categories)}
        radii = Isotope.symbol_to_radius()[self['symbol'].unique()]
        radii = Dict({i: radii[v] for i, v in symmap.items()}).tag(sync=True)
        colors = Isotope.symbol_to_color()[self['symbol'].unique()]
        colors = Dict({i: colors[v] for i, v in symmap.items()}).tag(sync=True)
        atom_x = grps.apply(lambda g: g['x'].values).to_json(orient='values', double_precision=self._precision)
        atom_x = Unicode(atom_x).tag(sync=True)
        atom_y = grps.apply(lambda g: g['y'].values).to_json(orient='values', double_precision=self._precision)
        atom_y = Unicode(atom_y).tag(sync=True)
        atom_z = grps.apply(lambda g: g['z'].values).to_json(orient='values', double_precision=self._precision)
        atom_z = Unicode(atom_z).tag(sync=True)
        return {'atom_symbols': symbols, 'atom_radii': radii, 'atom_colors': colors,
                'atom_x': atom_x, 'atom_y': atom_y, 'atom_z': atom_z}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._revert_categories()
        self.ix[self['symbol'].isin(['nan', 'NaN', 'none', 'None']), 'symbol'] = None
        self['symbol'].fillna('Dga', inplace=True)
        self._set_categories()


class Atom(BaseAtom):
    '''
    Absolute positions of atoms and their symbol.
    '''
    def get_element_mass(self, inplace=False):
        '''
        Retrieve the mass of each element in the atom dataframe.
        '''
        masses = self['symbol'].astype('O').map(Isotope.symbol_to_mass())
        if inplace:
            self['mass'] = masses
        else:
            return masses

    def reset_label(self):
        '''
        Reset the label column
        '''
        if 'label' in self:
            del self['label']
        nats = self.groupby('frame').size().values
        self['label'] = [i for nat in nats for i in range(nat)]
        self['label'] = self['label'].astype('category')

    def compute_simple_formula(self):
        '''
        Compute the simple formula for each frame.
        '''
        raise NotImplementedError()

    def _compute_unit_atom_static_cell(self, rxyz, oxyz):
        '''
        Given a static unit cell, compute the unit cell coordinates for each
        atom.

        Args:
            rxyz (:class:`~numpy.ndarray`): Unit cell magnitudes
            oxyz (:class:`~numpy.ndarray`): Unit cell origin

        Returns:
            sparse_df (:pandas:`~pandas.SparseDataFrame`): Sparse dataframe of in unit cell positions
        '''
        xyz = self[['x', 'y', 'z']]
        unit = np.mod(xyz, rxyz) + oxyz
        return UnitAtom(unit[unit != xyz].astype(np.float64).to_sparse())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'label' not in self.columns:
            self.reset_label()


class UnitAtom(SparseDataFrame):
    '''
    In unit cell coordinates (sparse) for periodic systems. These coordinates
    are used to update the corresponding :class:`~atomic.atom.Atom` object
    '''
    _indices = ['atom']
    _columns = ['x', 'y', 'z']


class ProjectedAtom(BaseAtom):
    '''
    Projected atom coordinates (e.g. on 3x3x3 supercell). These coordinates are
    typically associated with their corresponding indices in another dataframe.
    '''
    _indices = ['prjd_atom']
    _columns = ['x', 'y', 'z', 'symbol', 'frame', 'atom']
    _traits = []
    _groupbys = ['frame']
    _categories = {'atom': np.int64, 'frame': np.int64, 'label': np.int64, 'symbol': str}

    def _get_custom_traits(self):
        return {}

    def reset_label(self, atom_label):
        '''
        '''
        if 'label' in self:
            del self['label']
        self['label'] = self['atom'].map(atom_label)


class VisualAtom(SparseDataFrame):
    '''
    Akin to :class:`~atomic.atom.UnitAtom`, this class is used to store a special
    set of coordinates used specifically for visualization. Typically these coordinates
    are the unit cell coordinates of a periodic system with select atoms translated
    so as not to break apart molecules across the periodic boundary.
    '''
    _indices = ['atom']
    _columns = ['x', 'y', 'z']

    def _get_custom_traits(self):
        return {}


def compute_unit_atom(universe):
    '''
    Compute the unit cell coordinates of the atoms.

    Args:
        universe (:class:`~atomic.universe.Universe`): Atomic universe

    Returns:
        sparse_df (:pandas:`~pandas.SparseDataFrame`): Sparse dataframe of in unit cell positions
    '''
    if not universe.is_periodic:
        raise TypeError('Is this a periodic universe? Check frame for periodic column.')
    if universe.is_variable_cell:
        raise NotImplementedError('Variable cell simulations not yet supported')
    idx = universe.frame.index[0]
    rxyz = universe.frame.ix[idx, ['rx', 'ry', 'rz']].values
    oxyz = universe.frame.ix[idx, ['ox', 'oy', 'oz']].values
    return universe.atom._compute_unit_atom_static_cell(rxyz, oxyz)


def compute_projected_atom(universe):
    '''
    Computes the 3x3x3 supercell coordinates from the unit cell coordinates.

    Args:
        universe (:class:`~atomic.universe.Universe`): The atomic universe

    Returns:
        two (:class:`~atomic.two.PeriodicTwo`): Two body distances
    '''
    if not universe.is_periodic:
        raise TypeError('Is this a periodic universe? Check frame for periodic column.')
    if universe.is_variable_cell:
        raise NotImplementedError('Variable cell simulations not yet supported')
    return _compute_projected_static(universe)


def _compute_projected_static(universe):
    '''
    Compute the 3x3x3 supercell coordinates given a static unit cell
    '''
    idx = universe.frame.index[0]
    ua = universe.unit_atom
    x = ua['x'].values
    y = ua['y'].values
    z = ua['z'].values
    rx = universe.frame.ix[idx, 'rx']
    ry = universe.frame.ix[idx, 'ry']
    rz = universe.frame.ix[idx, 'rz']
    x, y, z = supercell3(x, y, z, rx, ry, rz)
    df = pd.DataFrame.from_dict({'x': x, 'y': y, 'z': z})
    df['frame'] = pd.Series(ua['frame'].astype(np.int64).values.tolist() * 27, dtype='category')
    df['symbol'] = pd.Series(ua['symbol'].astype(str).values.tolist() * 27, dtype='category')
    df['atom'] = pd.Series(ua.index.values.tolist() * 27, dtype='category')
    return ProjectedAtom(df)


def compute_visual_atom(universe):
    '''
    Creates visually pleasing atomic coordinates (useful for periodic
    systems).

    See Also:
        :func:`~atomic.universe.Universe.compute_vis_atom`
    '''
    def process(grp):
        if len(grp) > 0:
            if not np.all(grp['bond_count'] == grp.iloc[0, -1]):
                return grp.iloc[0, :]

    if not universe.is_periodic:
        raise TypeError('Is this a periodic universe? Check frame for periodic column.')
    universe.compute_projected_bond_count()
    updater = universe.two.ix[universe.two['bond'] == True, ['prjd_atom0', 'prjd_atom1']].stack()
    updater = universe.projected_atom[universe.projected_atom.index.isin(updater.values)]
    dup = updater[updater['atom'].duplicated()]
    dup = updater[updater['atom'].isin(dup['atom'])].sort_values('bond_count', ascending=False)
    updater = updater[~updater.index.isin(dup.index)]
    ddup = dup.groupby('atom').apply(process).dropna()[['x', 'y', 'z']]
    updater = updater.set_index('atom')[['x', 'y', 'z']]
    updater = pd.concat((updater, ddup))
    vis = universe._unit_atom.copy()
    vis.update(updater)
    return VisualAtom(vis)
