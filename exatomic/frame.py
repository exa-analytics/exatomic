# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Frame Dataframe
######################
A frame represents a single unique nuclear geometry. Frames are distinguishable
by any type of information, time, level of theory, differences in atomic
structure, etc.

+-------------------+----------+-------------------------------------------+
| Column            | Type     | Description                               |
+===================+==========+===========================================+
| atom_count        | int      | non-unique integer (req.)                 |
+-------------------+----------+-------------------------------------------+
| ox                | float    | unit cell origin point in x               |
+-------------------+----------+-------------------------------------------+
| oy                | float    | unit cell origin point in y               |
+-------------------+----------+-------------------------------------------+
| oz                | float    | unit cell origin point in z               |
+-------------------+----------+-------------------------------------------+
| periodic          | bool     | periodic frame?                           |
+-------------------+----------+-------------------------------------------+

See Also:
    More information on the :class:`~exatomic.frame.Frame` concept can be
    found in :mod:`~exatomic.universe` module's documentation.
'''
import numpy as np
from traitlets import Float
from exa.numerical import DataFrame
from exa.algorithms import vmag3


class Frame(DataFrame):
    '''
    The frame DataFrame contains non-atomic information about each snapshot
    of the :class:`~exatomic.universe.Universe` object.
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
            how (str): Either any (default) or all

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
    def is_vc(self, how='all'):
        '''
        Check if this is a variable unit cell simulation.

        Note:
            Returns false if not periodic
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
        return False


    def compute_cell_magnitudes(self):
        '''
        Compute the magnitudes of the unit cell vectors (rx, ry, rz).
        '''
        xi = self['xi'].values    # Vector component variables are denoted by
        xj = self['xj'].values    # their basis vector ending: _i, _j, _k
        xk = self['xk'].values
        yi = self['yi'].values
        yj = self['yj'].values
        yk = self['yk'].values
        zi = self['zi'].values
        zj = self['zj'].values
        zk = self['zk'].values
        self['rx'] = vmag3(xi, yi, zi)**0.5
        self['ry'] = vmag3(xj, yj, zj)**0.5
        self['rz'] = vmag3(xk, yk, zk)**0.5


def minimal_frame(atom):
    '''
    Create a minmal :class:`~exatomic.frame.Frame` object from a
    :class:`~exatomic.atom.Atom` object.
    '''
    frame = atom.groupby('frame').size().to_frame()
    frame.index = frame.index.astype(np.int64)
    frame.columns = ['atom_count']
    return Frame(frame)
