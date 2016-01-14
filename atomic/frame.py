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
        Compute the magnitudes of the periodic unit cell.

        Note that this computation adds three new column to the dataframe;
        'xr', 'yr', and 'zr'.
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
        self['xr'] = mag_3d(xi, xj, xk)
        self['yr'] = mag_3d(yi, yj, yk)
        self['zr'] = mag_3d(zi, zj, zk)


def minimal_frame_from_atoms(atoms):
    '''
    '''
    df = atoms.groupby(level='frame')['symbol'].count().to_frame()
    df.columns = ['atom_count']
    df['periodic'] = False
    return df
