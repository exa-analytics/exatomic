# -*- coding: utf-8 -*-
'''
Frame Dataframes
==========================
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

See Also:
    More information on the :class:`~atomic.frame.Frame` concept can be
    found in :mod:`~atomic.universe` module's documentation.
'''
import numpy as np
from traitlets import Float
from exa.numerical import DataFrame
from exa.algorithms import vmag3


class Frame(DataFrame):
    '''
    The frame DataFrame contains non-atomic information about each snapshot
    of the :class:`~atomic.universe.Universe` object.
    '''
    xi = Float()  # Static unit cell component
    xj = Float()  # Static unit cell component
    xk = Float()  # Static unit cell component
    yi = Float()  # Static unit cell component
    yj = Float()  # Static unit cell component
    yk = Float()  # Static unit cell component
    zi = Float()  # Static unit cell component
    zj = Float()  # Static unit cell component
    zk = Float()  # Static unit cell component
    ox = Float()  # Static unit cell origin point x
    oy = Float()  # Static unit cell origin point y
    oz = Float()  # Static unit cell origin point z
    _indices = ['frame']
    _columns = ['atom_count']
    _traits = ['xi', 'xj', 'xk', 'yi', 'yj', 'yk', 'zi', 'zj', 'zk',
               'ox', 'oy', 'oz', 'frame']

    @property
    def is_periodic(self):
        '''
        Check if (any) frame is/are periodic.

        Returns:
            result (bool): True if periodic false otherwise
        '''
        if 'periodic' in self.columns:
            if np.any(self['periodic'] == True):
                return True
        return False

    @property
    def is_variable_cell(self):
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
        self['rx'] = vmag3(xi, xj, xk)
        self['ry'] = vmag3(yi, yj, yk)
        self['rz'] = vmag3(zi, zj, zk)


def minimal_frame(atom):
    '''
    Create a minmal :class:`~atomic.frame.Frame` object from a
    :class:`~atomic.atom.Atom` object.
    '''
    atom._revert_categories()
    frame = atom.groupby('frame').count().ix[:, 0].to_frame()
    frame.columns = ['atom_count']
    atom._set_categories()
    return Frame(frame)
