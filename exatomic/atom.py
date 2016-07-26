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


class ProjectedAtom(SparseDataFrame):
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
            bonded = universe.two.ix[universe.two['bond'] == True, 'atom1'].astype(np.int64)
            prjd = universe.projected_atom.ix[bonded.index].to_dense()
            prjd['atom'] = bonded
            prjd.drop_duplicates('atom', inplace=True)
            prjd.set_index('atom', inplace=True)
            atom.update(prjd)
            atom = atom[atom != universe.atom[['x', 'y', 'z']]].to_sparse()
            return cls(atom)
        raise PeriodicUniverseError()
