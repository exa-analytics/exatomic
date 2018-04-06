# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Atomic Position Data
############################
This module provides a collection of dataframes supporting nuclear positions,
forces, velocities, symbols, etc. (all data associated with atoms as points).
"""
from numbers import Integral
import numpy as np
import pandas as pd
from exa import DataFrame, SparseDataFrame, Series
from exa.util.units import Length
from exatomic.base import sym2z, sym2mass
from exatomic.algorithms.distance import modv
from exatomic.core.error import PeriodicUniverseError
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
    _index = 'atom'
    _cardinal = ('frame', np.int64)
    _categories = {'symbol': str, 'set': np.int64, 'molecule': np.int64,
                   'label': np.int64}
    _columns = ['x', 'y', 'z', 'symbol']

    #@property
    #def _constructor(self):
    #    return Atom

    @property
    def nframes(self):
        """Return the total number of frames in the atom table."""
        return np.int64(self.frame.cat.as_ordered().max() + 1)

    @property
    def last_frame(self):
        """Return the last frame of the atom table."""
        return self[self.frame == self.nframes - 1]

    @property
    def unique_atoms(self):
        """Return unique atom symbols of the last frame."""
        return self.last_frame.symbol.unique()

    def center(self, idx, frame=None):
        """Return a copy of a single frame of the atom table
        centered around a specific atom index."""
        if frame is None: frame = self.last_frame.copy()
        else: frame = self[self.frame == frame].copy()
        center = frame.ix[idx]
        for r in ['x', 'y', 'z']:
            if center[r] > 0: frame[r] = frame[r] - center[r]
            else: frame[r] = frame[r] + np.abs(center[r])
        return frame


    def to_xyz(self, tag='symbol', header=False, comments='', columns=None,
               frame=None, units='Angstrom'):
        """
        Return atomic data in XYZ format, by default without the first 2 lines.
        If multiple frames are specified, return an XYZ trajectory format. If
        frame is not specified, by default returns the last frame in the table.

        Args:
            tag (str): column name to use in place of 'symbol'
            header (bool): if True, return the first 2 lines of XYZ format
            comment (str, list): comment(s) to put in the comment line
            frame (int, iter): frame or frames to return
            units (str): units (default angstroms)

        Returns:
            ret (str): XYZ formatted atomic data
        """
        # TODO :: this is conceptually a duplicate of XYZ.from_universe
        columns = (tag, 'x', 'y', 'z') if columns is None else columns
        frame = self.nframes - 1 if frame is None else frame
        if isinstance(frame, Integral): frame = [frame]
        if not isinstance(comments, list): comments = [comments]
        if len(comments) == 1: comments = comments * len(frame)
        df = self[self['frame'].isin(frame)].copy()
        if tag not in df.columns:
            if tag == 'Z':
                stoz = sym2z()
                df[tag] = df['symbol'].map(stoz)
        df['x'] *= Length['au', units]
        df['y'] *= Length['au', units]
        df['z'] *= Length['au', units]
        grps = df.groupby('frame')
        ret = ''
        formatter = {tag: '{:<5}'.format}
        stargs = {'columns': columns, 'header': False,
                  'index': False, 'formatters': formatter}
        t = 0
        for _, grp in grps:
            if not len(grp): continue
            tru = (header or comments[t] or len(frame) > 1)
            hdr = '\n'.join([str(len(grp)), comments[t], '']) if tru else ''
            ret = ''.join([ret, hdr, grp.to_string(**stargs), '\n'])
            t += 1
        return ret

    def get_element_masses(self):
        """Compute and return element masses from symbols."""
        return self['symbol'].astype('O').map(sym2mass)

    def get_atom_labels(self):
        """
        Compute and return enumerated atoms.

        Returns:
            labels (:class:`~exa.core.numerical.Series`): Enumerated atom labels (of type int)
        """
        nats = self.cardinal_groupby().size().values
        labels = Series([i for nat in nats for i in range(nat)], dtype='category')
        labels.index = self.index
        return labels

    @classmethod
    def from_small_molecule_data(cls, center=None, ligand=None, distance=None, geometry=None,
                                 offset=None, plane=None, axis=None, domains=None, unit='Angstrom'):
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

    #@property
    #def _constructor(self):
    #    return UnitAtom

    @classmethod
    def from_universe(cls, universe):
        if universe.periodic:
            if "rx" not in universe.frame.columns:
                universe.frame.compute_cell_magnitudes()
            a, b, c = universe.frame[["rx", "ry", "rz"]].max().values
            x = modv(universe.atom['x'].values, a)
            y = modv(universe.atom['y'].values, b)
            z = modv(universe.atom['z'].values, c)
            df = pd.DataFrame.from_dict({'x': x, 'y': y, 'z': z})
            df.index = universe.atom.index
            df = df[universe.atom[['x', 'y', 'z']] != df].to_sparse()
            return cls(df)
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

    #@property
    #def _constructor(self):
    #    return ProjectedAtom


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

    #@property
    #def _constructor(self):
    #    return VisualAtom


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
    #@property
    #def _constructor(self):
    #    return Frequency

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
