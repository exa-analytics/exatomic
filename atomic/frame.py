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
| nat               | int      | non-unique integer (req.)                 |
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
    _columns = ['nat']
    _traits = ['xi', 'xj', 'xk', 'yi', 'yj', 'yk', 'zi', 'zj', 'zk',
               'ox', 'oy', 'oz', 'frame']

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


def minimal_frame(atom):
    '''
    Create a minmal :class:`~atomic.frame.Frame` object from a
    :class:`~atomic.atom.Atom` object.
    '''
    atom._revert_categories()
    frame = atom.groupby('frame').count().ix[:, 0].to_frame()
    frame.columns = ['nat']
    atom._set_categories()
    return Frame(frame)


#    def get_unit_cell_magnitudes(self, inplace=False):
#        '''
#        Compute the magnitudes of the unit cell vectors.
#
#        Note that this computation adds three new column to the dataframe;
#        'rx', 'ry', and 'rz'.
#        '''
#        req_cols = ['xi', 'xj', 'xk', 'yi', 'yj', 'yk', 'zi', 'zj', 'zk']
#        missing_req = set(req_cols).difference(self.columns)
#        if missing_req:           # Check that we have cell dimensions
#            raise ColumnError(missing_req, self)
#        xi = self['xi'].values    # Vector component variables are denoted by
#        xj = self['xj'].values    # their basis vector ending: _i, _j, _k
#        xk = self['xk'].values
#        yi = self['yi'].values
#        yj = self['yj'].values
#        yk = self['yk'].values
#        zi = self['zi'].values
#        zj = self['zj'].values
#        zk = self['zk'].values
#        rx = mag_3d(xi, xj, xk)
#        ry = mag_3d(yi, yj, yk)
#        rz = mag_3d(zi, zj, zk)
#        if inplace:
#            self['rx'] = rx
#            self['ry'] = ry
#            self['rz'] = rz
#        else:
#            return (rx, ry, rz)
#
#    def is_periodic(self):
#        '''
#        '''
#        if 'periodic' in self.columns:
#            if np.any(self['periodic'] == True):
#                return True
#        return False
#
#
#    def is_variable_cell(self):
#        '''
#        Does the unit cell vary.
#
#        Returns:
#            is_vc (bool): True if variable cell dimension
#        '''
#        if 'rx' not in self.columns:
#            self.get_unit_cell_magnitudes(inplace=True)
#        rx = self['rx'].min()
#        ry = self['ry'].min()
#        rz = self['rz'].min()
#        if np.all(self['rx'] == rx) and np.all(self['ry'] == ry) and np.all(self['rz'] == rz):
#            return False
#        else:
#            return True
#
#
#def minimal_frame(universe, inplace=False):
#    '''
#    Generate the minimal :class:`~atomic.frame.Frame` dataframe given a
#    :class:`~atomic.universe.Universe` with a :class:`~atomic.atom.Atom` dataframe.
#
#    Args:
#        universe (:class:`~atomic.universe.Universe`): Universe with atoms
#        inplace (bool): Attach the frame dataframe to the universe.
#
#    Returns:
#        frame (:class:`~atomic.frame.Frame`): None if inplace is true
#    '''
#    df = Frame()
#    if universe.atom is not None:
#        if 'frame' in universe.atom.columns:
#            df = _min_frame_from_atom(universe.atom)
#    if inplace:
#        universe['_frame'] = df
#    else:
#        return df
#
#
#
