# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Atomic Position Data
############################
This module provides a collection of dataframes supporting nuclear positions,
forces, velocities, symbols, etc. (all data associated with atoms as points).
"""
import numpy as np
import pandas as pd
from traitlets import Dict, Unicode
from exa.numerical import DataFrame, SparseDataFrame, Series
from exa.relational.isotope import (symbol_to_color, symbol_to_radius,
                                   symbol_to_element_mass)
from exatomic.error import PeriodicUniverseError
from exatomic.math.distance import minimal_image_counts


class BaseAtom(DataFrame):
    """
    Base atom dataframe; sets some default precision (for traits creation and
    visualization), required columns, and categories.

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
    """
    _precision = {'x': 2, 'y': 2, 'z': 2}
    _index = 'atom'
    _groupby = ('frame', np.int64)
    _categories = {'symbol': str, 'set': np.int64, 'molecule': np.int64}


class Atom(BaseAtom):
    """
    This table contains the absolute coordinates (regardless of boundary
    conditions) of atomic nuclei.
    """
    _traits = ['x', 'y', 'z', 'set']
    _columns = ['x', 'y', 'z', 'symbol']

    def get_element_masses(self):
        """Compute and return element masses from symbols."""
        elem_mass = symbol_to_element_mass()
        return self['symbol'].astype('O').map(elem_mass)

    def get_atom_labels(self):
        """
        Compute and return enumerated atoms.

        Returns:
            labels (:class:`~exa.numerical.Series`): Enumerated atom labels (of type int)
        """
        nats = self.grouped().size().values
        labels = Series([i for nat in nats for i in range(nat)], dtype='category')
        labels.index = self.index
        return labels

    def _custom_traits(self):
        """
        Create creates for the atomic size (using the covalent radius) and atom
        colors (using the common `Jmol`_ color scheme). Note that that data is
        present in the static data (see :mod:`~exa.relational.isotope`).

        .. _Jmol: http://jmol.sourceforge.net/jscolors/
        """
        self._set_categories()
        kwargs = {}
        grps = self.grouped()
        symbols = grps.apply(lambda g: g['symbol'].cat.codes.values)    # Pass integers rather than string symbols
        kwargs['atom_symbols'] = Unicode(symbols.to_json(orient='values')).tag(sync=True)
        symmap = {i: v for i, v in enumerate(self['symbol'].cat.categories)}
        sym2rad = symbol_to_radius()
        radii = sym2rad[self['symbol'].unique()]
        kwargs['atom_radii'] = Dict({i: radii[v] for i, v in symmap.items()}).tag(sync=True)  # (Int) symbol radii
        sym2col = symbol_to_color()
        colors = sym2col[self['symbol'].unique()]    # Same thing for colors
        kwargs['atom_colors'] = Dict({i: colors[v] for i, v in symmap.items()}).tag(sync=True)
        return kwargs


class UnitAtom(SparseDataFrame):
    """
    In unit cell coordinates (sparse) for periodic systems. These coordinates
    are used to update the corresponding :class:`~exatomic.atom.Atom` object
    """
    _index = 'atom'
    _columns = ['x', 'y', 'z']

    @classmethod
    def from_universe(cls, universe):
        """
        """
        if universe.frame.is_periodic():
            xyz = universe.atom[['x', 'y', 'z']].values.astype(np.float64)
            if 'rx' not in universe.frame:
                universe.frame.compute_cell_magnitudes()
            counts = universe.frame['atom_count'].values.astype(np.int64)
            rxyz = universe.frame[['rx', 'ry', 'rz']].values.astype(np.float64)
            oxyz = universe.frame[['ox', 'oy', 'oz']].values.astype(np.float64)
            unit = pd.DataFrame(minimal_image_counts(xyz, rxyz, oxyz, counts), columns=['x', 'y', 'z'])
            unit = unit[unit != xyz].astype(np.float64).to_sparse()
            unit.index = universe.atom.index
            return cls(unit)
        raise PeriodicUniverseError()


class ProjectedAtom(DataFrame):
    """
    Projected atom coordinates (e.g. on 3x3x3 supercell). These coordinates are
    typically associated with their corresponding indices in another dataframe.

    Note:
        This table is computed when periodic two body properties are computed;
        it doesn't have meaning outside of that context.

    See Also:
        :func:`~exatomic.two.compute_periodic_two`.
    """
    _index = 'two'
    _columns = ['x', 'y', 'z']


class VisualAtom(SparseDataFrame):
    """
    """
    _index = 'atom'
    _columns = ['x', 'y', 'z']

    @classmethod
    def from_universe(cls, universe):
        """
        """
        if universe.frame.is_periodic():
            atom = universe.atom[['x', 'y', 'z']].copy()
            atom.update(universe.unit_atom)
            bonded = universe.two[universe.two['bond'] == True]
            prjd = universe.projected_atom.ix[bonded.index.values]
            prjd['atom'] = bonded['atom1'].astype(np.int64)
            prjd.drop_duplicates('atom', inplace=True)
            prjd.set_index('atom', inplace=True)
            atom.update(prjd)
            atom = atom[atom != universe.atom[['x', 'y', 'z']]].to_sparse()
            return cls(atom)
        raise PeriodicUniverseError()


#class VisualAtom(SparseDataFrame):
#    """
#    Akin to :class:`~exatomic.atom.UnitAtom`, this class is used to store a special
#    set of coordinates used specifically for visualization. Typically these coordinates
#    are the unit cell coordinates of a periodic system with select atoms translated
#    so as not to break apart molecules across the periodic boundary.
#    """
#    _indices = ['atom']
#    _columns = ['x', 'y', 'z']
#
#    def _get_custom_traits(self):
#        return {}
#
#
#def compute_unit_atom(universe):
#    """
#    Compute the in-unit-cell exatomic coordiations of a periodic universe.
#
#    Args:
#        universe: Periodic exatomic universe
#
#    Returns:
#        unit_atom (:class:`~exatomic.atom.UnitAtom`): Sparse dataframe of coordinations
#
#    Note:
#        The returned coordinate dataframe is sparse and is used to update the
#        atom dataframe as needed. Note that updating the atom dataframe overwrites
#        the data there, so typically one updates a copy of the atom dataframe.
#    """
#    if not universe.is_periodic:
#        raise TypeError('Is this a periodic universe? Check frame for periodic column.')
#
#def OLD_compute_unit_atom(universe):
#    """
#    Compute the unit cell coordinates of the atoms.
#
#    Args:
#        universe (:class:`~exatomic.universe.Universe`): Atomic universe
#
#    Returns:
#        sparse_df (:pandas:`~pandas.SparseDataFrame`): Sparse dataframe of in unit cell positions
#    """
#    if not universe.is_periodic:
#        raise TypeError('Is this a periodic universe? Check frame for periodic column.')
#    if universe.is_vc:
#        raise NotImplementedError('Variable cell simulations not yet supported')
#    idx = universe.frame.index[0]
#    rxyz = universe.frame.ix[idx, ['rx', 'ry', 'rz']].values
#    oxyz = universe.frame.ix[idx, ['ox', 'oy', 'oz']].values
#    return universe.atom._compute_unit_atom_static_cell(rxyz, oxyz)
#
#
#def compute_projected_atom(universe):
#    """
#    Computes the 3x3x3 supercell coordinates from the unit cell coordinates.
#
#    Args:
#        universe (:class:`~exatomic.universe.Universe`): The exatomic universe
#
#    Returns:
#        two (:class:`~exatomic.two.PeriodicTwo`): Two body distances
#    """
#    if not universe.is_periodic:
#        raise TypeError('Is this a periodic universe? Check frame for periodic column.')
#    if universe.is_vc:
#        raise NotImplementedError('Variable cell simulations not yet supported')
#    return _compute_projected_static(universe)
#
#
#def _compute_projected_static(universe):
#    """
#    Compute the 3x3x3 supercell coordinates given a static unit cell
#    """
#    idx = universe.frame.index[0]
#    ua = universe.unit_atom
#    x = ua['x'].values
#    y = ua['y'].values
#    z = ua['z'].values
#    rx = universe.frame.ix[idx, 'rx']
#    ry = universe.frame.ix[idx, 'ry']
#    rz = universe.frame.ix[idx, 'rz']
#    x, y, z = supercell3d(x, y, z, rx, ry, rz)
#    df = pd.DataFrame.from_dict({'x': x, 'y': y, 'z': z})
#    df['frame'] = pd.Series(ua['frame'].astype(np.int64).values.tolist() * 27, dtype='category')
#    df['symbol'] = pd.Series(ua['symbol'].astype(str).values.tolist() * 27, dtype='category')
#    df['atom'] = pd.Series(ua.index.values.tolist() * 27, dtype='category')
#    return ProjectedAtom(df)
#
#
#def compute_visual_atom(universe):
#    """
#    Creates visually pleasing exatomic coordinates (useful for periodic
#    systems).
#
#    See Also:
#        :func:`~exatomic.universe.Universe.compute_vis_atom`
#    """
#    if not universe.is_periodic:
#        raise TypeError('Is this a periodic universe? Check frame for periodic column.')
#    if 'bond_count' not in universe.projected_atom:
#        universe.compute_projected_bond_count()
#    if not universe._is('molecule'):
#        universe.compute_molecule()
#
#    bonded = universe.two.ix[(universe.two['bond'] == True), ['prjd_atom0', 'prjd_atom1']]
#    updater = universe.projected_atom[universe.projected_atom.index.isin(bonded.stack())]
#    dup_atom = updater.ix[updater['atom'].duplicated(), 'atom']
#    if len(dup_atom) > 0:
#        dup = updater[updater['atom'].isin(dup_atom)].sort_values('bond_count', ascending=False)
#        updater = updater[~updater.index.isin(dup.index)]
#        updater = updater.set_index('atom')[['x', 'y', 'z']]
#        grps = dup.groupby('atom')
#        indices = np.empty((grps.ngroups, ), dtype='O')
#        for i, (atom, grp) in enumerate(grps):
#            if len(grp) > 0:
#                m = universe.atom.ix[atom, 'molecule']
#                diff = grp.index[1] - grp.index[0]
#                atom_m = universe.atom[universe.atom['molecule'] == m]
#                prjd = universe.projected_atom[universe.projected_atom['atom'].isin(atom_m.index)]
#                notidx = grp.index[1]
#                if grp['bond_count'].diff().values[-1] == 0:
#                    updater = pd.concat((atom_m[['x', 'y', 'z']], updater))
#                    updater = updater.reset_index().drop_duplicates('atom').set_index('atom')
#                    indices[i] = []
#                else:
#                    mol = bonded[bonded['prjd_atom0'].isin(prjd.index) |
#                                 bonded['prjd_atom1'].isin(prjd.index)].stack().values
#                    mol = mol[mol != notidx]
#                    mol += diff
#                    indices[i] = mol.tolist() + [grp.index[1]]
#            else:
#                indices[i] = []
#        indices = np.concatenate(indices).astype(np.int64)
#        up = universe.projected_atom[universe.projected_atom.index.isin(indices)]
#        up = up.set_index('atom')[['x', 'y', 'z']]
#        if len(up) > 0:
#            updater = pd.concat((up, updater))
#            updater = updater.reset_index().drop_duplicates('atom').set_index('atom')
#    else:
#        updater = updater.set_index('atom')[['x', 'y', 'z']]
#    vis = universe.atom.copy()[['x', 'y', 'z']]
#    vis.update(updater)
#    vis = vis[vis != universe.atom[['x', 'y', 'z']]].dropna(how='all')
#    vis = VisualAtom(vis.to_sparse())
#    return vis
#
