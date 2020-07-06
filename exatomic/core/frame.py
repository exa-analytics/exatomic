# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Frame Data
######################
The primary "coordinate" for the atomic container (:class:`~exatomic.container.Universe`)
is the "frame". The frame concept can be anything; time, step along a geometry
optimization, different functional, etc. Each frame is distinguished from other
frames by unique atomic coordinates, a different level of theory, etc.
"""
import numpy as np
from exa import DataFrame
from exatomic.algorithms.distance import cartmag


class Frame(DataFrame):
    """
    Information about the current frame; a frame is a concept that distinguishes
    atomic coordinates along a molecular dynamics simulation, geometry optimization,
    etc.

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
    """
    _index = 'frame'
    _columns = ['atom_count']

#    @property
#    def _constructor(self):
#        return Frame

    def is_periodic(self, how='all'):
        """
        Check if any/all frames are periodic.

        Args:
            how (str): Require "any" frames to be periodic ("all" default)

        Returns:
            result (bool): True if any/all frame are periodic
        """
        if 'periodic' in self:
            if how == 'all' and np.all(self['periodic'] == True):
                return True
            elif how == 'any' and np.any(self['periodic'] == True):
                return True
        return False

    def is_variable_cell(self, how='all'):
        """
        Check if the simulation cell (applicable to periodic simulations) varies
        (e.g. variable cell molecular dynamics).
        """
        if self.is_periodic:
            if 'rx' not in self.columns:
                self.compute_cell_magnitudes()
            rx = self['rx'].min()
            ry = self['ry'].min()
            rz = self['rz'].min()
            if np.allclose(self['rx'], rx) and np.allclose(self['ry'], ry) and np.allclose(self['rz'], rz):
                return False
            else:
                return True
        raise PeriodicUniverseError()

    def compute_cell_magnitudes(self):
        """
        Compute the magnitudes of the unit cell vectors (rx, ry, rz).
        """
        self['rx'] = cartmag(self['xi'].values, self['yi'].values, self['zi'].values)
        self['ry'] = cartmag(self['xj'].values, self['yj'].values, self['zj'].values)
        self['rz'] = cartmag(self['xk'].values, self['yk'].values, self['zk'].values)

    def orthorhombic(self):
        if "xi" in self.columns and np.allclose(self["xj"], 0.0):
            return True
        return False


def compute_frame(universe):
    """
    Compute (minmal) :class:`~exatomic.frame.Frame` from
    :class:`~exatomic.container.Universe`.

    Args:
        uni (:class:`~exatomic.container.Universe`): Universe with atom table

    Returns:
        frame (:class:`~exatomic.frame.Frame`): Minimal frame table
    """
    return compute_frame_from_atom(universe.atom)


def compute_frame_from_atom(atom):
    """
    Compute :class:`~exatomic.frame.Frame` from :class:`~exatomic.atom.Atom`
    (or related).

    Args:
        atom (:class:`~exatomic.atom.Atom`): Atom table

    Returns:
        frame (:class:`~exatomic.frame.Frame`): Minimal frame table
    """
    frame = atom.cardinal_groupby().size().to_frame()
    frame.index = frame.index.astype(np.int64)
    frame.columns = ['atom_count']
    return Frame(frame)
