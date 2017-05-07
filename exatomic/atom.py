# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Nuclear Coordinates
############################
This module provides data objects targeted at storing
"""
###from exa.core import DataFrame
###
###
###class Atom(DataFrame):
###    """
###    A table of the nuclear coordinates.
###
###    At a minimum this table requires the atom symbol, coordinates in 3D space,
###    and frame (i.e. point in time/SCF iteration/etc.).
###    """
###    _required_columns = ("symbol", "x", "y", "z", "frame")
###    _col_descriptions = {'symbol': "Element symbol", 'x': "Absolute x coordinate",
###                         'y': "Absolute y coordinate", 'z': "Absolute y coordinate",
###                         'frame': "Frame index"}
###


from numbers import Integral
import numpy as np
import pandas as pd
from traitlets import Dict, Unicode
try:
    from exa.core.numerical import DataFrame, SparseDataFrame, Series
    from exa.cms.isotope import (symbol_to_color, symbol_to_radius, symbol_to_znum,
                                    symbol_to_mass)
except ImportError:
    from exa.numerical import DataFrame, SparseDataFrame, Series
    from exa.relational.isotope import (symbol_to_color, symbol_to_radius,
                                        symbol_to_z, symbol_to_element_mass)
    symbol_to_znum = symbol_to_z
    symbol_to_mass = symbol_to_element_mass
from exatomic import Length
from exatomic.error import PeriodicUniverseError
#from exatomic.algorithms.distance import minimal_image_counts
from exatomic.algorithms.geometry import make_small_molecule


class Atom(DataFrame):
    """
    The atom dataframe.

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
    _cardinal = ('frame', np.int64)
    _categories = {'symbol': str, 'set': np.int64, 'molecule': np.int64,
                   'label': np.int64}
    _traits = ['x', 'y', 'z', 'set']
    _columns = ['x', 'y', 'z', 'symbol']

    @property
    def nframes(self):
        """Return the total number of frames in the atom table."""
        return self.frame.cat.as_ordered().max() + 1

    @property
    def last_frame(self):
        """Return the last frame of the atom table."""
        return self[self.frame == self.nframes - 1]

    @property
    def unique_atoms(self):
        """Return unique atom symbols of the last frame."""
        return self.last_frame.symbol.unique()

    def center(self, idx, frame=None):
        if frame is None: frame = self.last_frame.copy()
        else: frame = self[self.frame == frame].copy()
        center = frame.ix[idx]
        for r in ['x', 'y', 'z']:
            if center[r] > 0: frame[r] = frame[r] - center[r]
            else: frame[r] = frame[r] + np.abs(center[r])
        return frame


    def to_xyz(self, tag='symbol', header=False, comments='', columns=None,
               frame=None, units='A'):
        """
        Return atomic data in XYZ format, by default without the first 2 lines.
        If multiple frames are specified, return an XYZ trajectory format. If
        frame is not specified, by default returns the last frame in the table.

        Args
            tag (str): column name to use in place of 'symbol'
            header (bool): if True, return the first 2 lines of XYZ format
            comment (str,list): comment(s) to put in the comment line
            frame (int,iter): frame or frames to return
            units (str): units (default angstroms)

        Returns
            ret (str): XYZ formatted atomic data
        """
        columns = (tag, 'x', 'y', 'z') if columns is None else columns
        frame = self.nframes - 1 if frame is None else frame
        if isinstance(frame, Integral): frame = [frame]
        if not isinstance(comments, list): comments = [comments]
        if len(comments) == 1: comments = comments * len(frame)
        df = self[self['frame'].isin(frame)].copy()
        if tag not in df.columns:
            if tag == 'Z':
                stoz = symbol_to_z()
                df[tag] = df['symbol'].map(stoz)
        df['x'] *= Length['au', units]
        df['y'] *= Length['au', units]
        df['z'] *= Length['au', units]
        grps = df.groupby('frame')
        ret = ''
        formatter = {tag: lambda x: '{:<5}'.format(x)}
        stargs = {'columns': columns, 'header': False,
                  'index': False, 'formatters': formatter}
        t = 0
        for f, grp in grps:
            if not len(grp): continue
            tru = (header or comments[t] or len(frame) > 1)
            hdr = '\n'.join([str(len(grp)), comments[t], '']) if tru else ''
            ret = ''.join([ret, hdr, grp.to_string(**stargs), '\n'])
            t += 1
        return ret

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
        nats = self.cardinal_groupby().size().values
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
        grps = self.cardinal_groupby()
        symbols = grps.apply(lambda g: g['symbol'].cat.codes.values)    # Pass integers rather than string symbols
        kwargs['atom_symbols'] = Unicode(symbols.to_json(orient='values')).tag(sync=True)
        symmap = {i: v for i, v in enumerate(self['symbol'].cat.categories) if v in self.unique_atoms}
        sym2rad = symbol_to_radius()
        radii = sym2rad[self['symbol'].unique()]
        kwargs['atom_radii'] = Dict({i: radii[v] for i, v in symmap.items()}).tag(sync=True)  # (Int) symbol radii
        sym2col = symbol_to_color()
        colors = sym2col[self['symbol'].unique()]    # Same thing for colors
        kwargs['atom_colors'] = Dict({i: colors[v] for i, v in symmap.items()}).tag(sync=True)
        return kwargs

    @classmethod
    def from_small_molecule_data(cls, center=None, ligand=None, distance=None, geometry=None,
                                 offset=None, plane=None, axis=None, domains=None, unit='A'):
        '''
        A minimal molecule builder for simple one-center, homogeneous ligand
        molecules of various general chemistry molecular geometries. If domains
        is not specified and geometry is ambiguous (like 'bent'),
        it just guesses the simplest geometry (smallest number of domains).

        Args
            center (str): atomic symbol of central atom
            ligand (str): atomic symbol of ligand atoms
            distance (float): distance between central atom and any ligand
            geometry (str): molecular geometry
            domains (int): number of electronic domains
            offset (np.array): 3-array of position of central atom
            plane (str): cartesian plane of molecule (eg. for 'square_planar')
            axis (str): cartesian axis of molecule (eg. for 'linear')

        Returns
            exatomic.atom.Atom: Atom table of small molecule
        '''
        return cls(make_small_molecule(center=center, ligand=ligand, distance=distance,
                                       geometry=geometry, offset=offset, plane=plane,
                                       axis=axis, domains=domains, unit=unit))


class UnitAtom(SparseDataFrame):
    """
    In unit cell coordinates (sparse) for periodic systems. These coordinates
    are used to update the corresponding :class:`~exatomic.atom.Atom` object
    """
    _index = 'atom'
    _columns = ['x', 'y', 'z']

#    @classmethod
#    def from_universe(cls, universe):
#        """
#        """
#        if universe.frame.is_periodic():
#            xyz = universe.atom[['x', 'y', 'z']].values.astype(np.float64)
#            if 'rx' not in universe.frame:
#                universe.frame.compute_cell_magnitudes()
#            counts = universe.frame['atom_count'].values.astype(np.int64)
#            rxyz = universe.frame[['rx', 'ry', 'rz']].values.astype(np.float64)
#            oxyz = universe.frame[['ox', 'oy', 'oz']].values.astype(np.float64)
#            unit = pd.DataFrame(minimal_image_counts(xyz, rxyz, oxyz, counts), columns=['x', 'y', 'z'])
#            unit = unit[unit != xyz].astype(np.float64).to_sparse()
#            unit.index = universe.atom.index
#            return cls(unit)
#        raise PeriodicUniverseError()


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
            bonded = universe.atom_two.ix[universe.atom_two['bond'] == True, 'atom1'].astype(np.int64)
            prjd = universe.projected_atom.ix[bonded.index].to_dense()
            prjd['atom'] = bonded
            prjd.drop_duplicates('atom', inplace=True)
            prjd.set_index('atom', inplace=True)
            atom.update(prjd)
            atom = atom[atom != universe.atom[['x', 'y', 'z']]].to_sparse()
            return cls(atom)
        raise PeriodicUniverseError()


class Frequency(DataFrame):
    """
    The Frequency dataframe.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | frame             | category | non-unique integer (req.)                 |
    +-------------------+----------+-------------------------------------------+
    | frequency         | float    | frequency of oscillation (cm-1) (req.)    |
    +-------------------+----------+-------------------------------------------+
    | freqdx            | int      | index of frequency of oscillation (req.)  |
    +-------------------+----------+-------------------------------------------+
    | dx                | float    | atomic displacement in x direction (req.) |
    +-------------------+----------+-------------------------------------------+
    | dy                | float    | atomic displacement in y direction (req.) |
    +-------------------+----------+-------------------------------------------+
    | dz                | float    | atomic displacement in z direction (req.) |
    +-------------------+----------+-------------------------------------------+
    | symbol            | str      | atomic symbol (req.)                      |
    +-------------------+----------+-------------------------------------------+
    | label             | int      | atomic identifier                         |
    +-------------------+----------+-------------------------------------------+
    """
    def displacement(self, freqdx):
        return self[self['freqdx'] == freqdx][['dx', 'dy', 'dz', 'symbol']]

def add_vibrational_mode(uni, freqdx):
    displacements = uni.frequency.displacements(freqdx)
    if not all(displacements['symbol'] == uni.atom['symbol']):
        print('Mismatch in ordering of atoms and frequencies.')
        return
    displaced = []
    frames = []
    # Should these only be absolute values?
    factor = np.abs(np.sin(np.linspace(-4*np.pi, 4*np.pi, 200)))
    for fac in factor:
        moved = uni.atom.copy()
        moved['x'] += displacements['dx'].values * fac
        moved['y'] += displacements['dy'].values * fac
        moved['z'] += displacements['dz'].values * fac
        displaced.append(moved)
        frames.append(uni.frame)
    movie = pd.concat(displaced).reset_index()
    movie['frame'] = np.repeat(range(len(factor)), len(uni.atom))
    uni.frame = pd.concat(frames).reset_index()
    uni.atom = movie
    uni._traits_need_update = True
    uni._update_traits()
