# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
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
from traitlets import Float
from exa.numerical import DataFrame
from exa.math.vector.cartesian import magnitude_xyz


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
    _precision = {'xi': 2, 'xj': 2, 'xk': 2, 'yi': 2, 'yj': 2, 'yk': 2, 'zi': 2,
                  'zj': 2, 'zk': 2, 'ox': 2, 'oy': 2, 'oz': 2}
    # Note that adding "frame" below turns the index of this dataframe into a trait
    _traits = ['xi', 'xj', 'xk', 'yi', 'yj', 'yk', 'zi', 'zj', 'zk',
               'ox', 'oy', 'oz', 'frame']

    def create_cubic_lattice(self, a, ox=0.0, oy=0.0, oz=0.0):
        """
        Create a static cubic cell lattice for the current universe.
        """
        self['ox'] = ox
        self['oy'] = oy
        self['oz'] = oz
        for i, xyz in enumerate(["x", "y", "z"]):
            for j, ijk in enumerate(["i", "j", "k"]):
                if i == j:
                    self[xyz+ijk] = a
                else:
                    self[xyz+ijk] = 0.0

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
        self['rx'] = magnitude_xyz(self['xi'].values, self['yi'].values, self['zi'].values).astype(np.float64)
        self['ry'] = magnitude_xyz(self['xj'].values, self['yj'].values, self['zj'].values).astype(np.float64)
        self['rz'] = magnitude_xyz(self['xk'].values, self['yk'].values, self['zk'].values).astype(np.float64)


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
