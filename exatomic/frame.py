# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Frame Data
######################
The primary "coordinate" for the atomic container (:class:`~exatomic.container.Universe`)
is the "frame". The frame concept can be anything; time, step along a geometry
optimization, different functional, etc. Each frame is distinguished from other
frames by unique atomic coordinates, a different level of theory, etc.
The following is a guide for the typical data found in this dataframe.

+-------------------+----------+-------------------------------------------+
| Column            | Type     | Description                               |
+===================+==========+===========================================+
| atom_count        | int      | non-unique integer (req.)                 |
+-------------------+----------+-------------------------------------------+
| molecule_count    | int      | non-unique integer                        |
+-------------------+----------+-------------------------------------------+
| ox                | float    | unit cell origin point in x               |
+-------------------+----------+-------------------------------------------+
| oy                | float    | unit cell origin point in y               |
+-------------------+----------+-------------------------------------------+
| oz                | float    | unit cell origin point in z               |
+-------------------+----------+-------------------------------------------+
| periodic          | bool     | true if periodic system                   |
+-------------------+----------+-------------------------------------------+
'''
import numpy as np
from traitlets import Float
from exa.numerical import DataFrame
from exa.math.vector.cartesian import magnitude_xyz
from exatomic.error import PeriodicUniverseError


class Frame(DataFrame):
    '''
    Information about the current frame; a frame is a concept that distinguishes
    atomic coordinates along a molecular dynamics simulation, geometry optimization,
    etc.
    '''
    _indices = ['frame']
    _columns = ['atom_count']
    _precision = {'xi': 2, 'xj': 2, 'xk': 2, 'yi': 2, 'yj': 2, 'yk': 2, 'zi': 2,
                  'zj': 2, 'zk': 2, 'ox': 2, 'oy': 2, 'oz': 2}
    _traits = ['xi', 'xj', 'xk', 'yi', 'yj', 'yk', 'zi', 'zj', 'zk',
               'ox', 'oy', 'oz', 'frame']

    @property
    def is_periodic(self, how='all'):
        '''
        Check if any/all frames are periodic.

        Args:
            how (str): Require "any" frames to be periodic ("all" default)

        Returns:
            result (bool): True if any/all frame are periodic
        '''
        if 'periodic' in self:
            if how == 'all':
                if np.all(self['periodic'] == True):
                    return True
            elif how == 'any':
                if np.any(self['periodic'] == True):
                    return True
        return False

    @property
    def is_variable_cell(self, how='all'):
        '''
        Check if the simulation cell (applicable to periodic simulations) varies
        (e.g. variable cell molecular dynamics).
        '''
        if self.is_periodic:
            if 'rx' not in self.columns:
                self.compute_cell_magnitudes()
            rx = self['rx'].min()
            ry = self['ry'].min()
            rz = self['rz'].min()
            if np.all(self['rx'] == rx) and np.all(self['ry'] == ry) and np.all(self['rz'] == rz):
                return False
            else:
                return True
        raise PeriodicUniverseError()

    def compute_cell_magnitudes(self):
        '''
        Compute the magnitudes of the unit cell vectors (rx, ry, rz).
        '''
        self['rx'] = magnitude_xyz(self['xi'].values, self['yi'].values, self['zi'].values)
        self['ry'] = magnitude_xyz(self['xj'].values, self['yj'].values, self['zj'].values)
        self['rz'] = magnitude_xyz(self['xk'].values, self['yk'].values, self['zk'].values)


def compute_frame(universe):
    '''
    Compute (minmal) :class:`~exatomic.frame.Frame` from
    :class:`~exatomic.container.Universe`.

    Args:
        uni (:class:`~exatomic.container.Universe`): Universe with atom table

    Returns:
        frame (:class:`~exatomic.frame.Frame`): Minimal frame table
    '''
    return compute_frame_from_atom(universe.atom)


def compute_frame_from_atom(atom):
    '''
    Compute :class:`~exatomic.frame.Frame` from :class:`~exatomic.atom.Atom`
    (or related).

    Args:
        atom (:class:`~exatomic.atom.Atom`): Atom table

    Returns:
        frame (:class:`~exatomic.frame.Frame`): Minimal frame table
    '''
    frame = atom.groupby('frame').size().to_frame()
    frame.index = frame.index.astype(np.int64)
    frame.columns = ['atom_count']
    return Frame(frame)
