# -*- coding: utf-8 -*-
'''
Atom DataFrame
==========================
'''
from exa import DataFrame
from exa.jitted.broadcasting import mag_3d


class Frame(DataFrame):
    '''
    Required indexes: frame
    Required columns: atom_count
    '''
    __indices__ = ['frame']
    __columns__ = ['atom_count']

    def cell_mags(self):
        '''
        Compute the magnitudes of the unit cell vectors.

        Note that this computation adds three new column to the dataframe;
        'xr', 'yr', and 'zr'.
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
        self['xr'] = mag_3d(xi, xj, xk)
        self['yr'] = mag_3d(yi, yj, yk)
        self['zr'] = mag_3d(zi, zj, zk)


def minimal_frame_from_atoms(atoms):
    '''
    '''
    df = atoms.groupby(level='frame')['symbol'].count().to_frame()
    df.columns = ['atom_count']
    return df
