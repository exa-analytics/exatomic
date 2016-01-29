# -*- coding: utf-8 -*-
'''
Atom DataFrame
==========================
'''
from exa import _np as np
from exa.frames import DataFrame
from exa.jitted.broadcasting import mag_3d


class Frame(DataFrame):
    '''
    '''
    __pk__ = ['frame']
    __traits__ = ['xi', 'xj', 'xk', 'yi', 'yj', 'yk', 'zi', 'zj', 'zk',
                  'rx', 'ry', 'rz', 'ox', 'oy', 'oz']

    def get_unit_cell_magnitudes(self, inplace=False):
        '''
        Compute the magnitudes of the unit cell vectors.

        Note that this computation adds three new column to the dataframe;
        'rx', 'ry', and 'rz'.
        '''
        req_cols = ['xi', 'xj', 'xk', 'yi', 'yj', 'yk', 'zi', 'zj', 'zk']
        missing_req = set(req_cols).difference(self.columns)
        if missing_req:           # Check that we have cell dimensions
            raise ColumnError(missing_req, self)
        xi = self['xi'].values    # Vector component variables are denoted by
        xj = self['xj'].values    # their basis vector ending: _i, _j, _k
        xk = self['xk'].values
        yi = self['yi'].values
        yj = self['yj'].values
        yk = self['yk'].values
        zi = self['zi'].values
        zj = self['zj'].values
        zk = self['zk'].values
        rx = mag_3d(xi, xj, xk)
        ry = mag_3d(yi, yj, yk)
        rz = mag_3d(zi, zj, zk)
        if inplace:
            self['rx'] = rx
            self['ry'] = ry
            self['rz'] = rz
        else:
            return (rx, ry, rz)

    def is_periodic(self):
        '''
        '''
        if 'periodic' in self.columns:
            if np.any(self['periodic'] == True):
                return True
        return False


    def is_variable_cell(self):
        '''
        Does the unit cell vary.

        Returns:
            is_vc (bool): True if variable cell dimension
        '''
        if 'rx' not in self.columns:
            self.get_unit_cell_magnitudes(inplace=True)
        rx = self['rx'].min()
        ry = self['ry'].min()
        rz = self['rz'].min()
        if np.all(self['rx'] == rx) and np.all(self['ry'] == ry) and np.all(self['rz'] == rz):
            return False
        else:
            return True


def minimal_frame(atom):
    '''
    Generate the minimal :class:`~atomic.frame.Frame` dataframe given an
    :class:`~atomic.atom.Atom` dataframe.

    Args:
        atoms (:class:`~atomic.atom.Atom`): Atoms dataframe

    Returns:
        frames (:class:`~atomic.frame.Frame`): Frames dataframe
    '''
    df = atom.groupby('frame').count().iloc[:, 0].to_frame()
    df.columns = ['atom_count']
    return Frame(df)
