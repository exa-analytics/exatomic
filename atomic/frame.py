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
        '''
        xi = self['xi'].values
        xj = self['xj'].values
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
