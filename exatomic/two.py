# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Two Body Properties Table
##################################
This module provides functions for computing interatomic distances and bonds
(i.e. two body properties). This computation depends on the type of boundary
conditions used; free or periodic boundaries. The following table provides a
guide for the types of data found in the two types of two body tables provided
by this module.

+-------------------+----------+---------------------------------------------+
| Column            | Type     | Description                                 |
+===================+==========+=============================================+
| atom0             | integer  | foreign key to :class:`~exatomic.atom.Atom` |
+-------------------+----------+---------------------------------------------+
| atom1             | integer  | foreign key to :class:`~exatomic.atom.Atom` |
+-------------------+----------+---------------------------------------------+
| distance          | float    | distance between atom0 and atom1            |
+-------------------+----------+---------------------------------------------+
| bond              | boolean  | True if bond                                |
+-------------------+----------+---------------------------------------------+
| frame             | category | non-unique integer (req.)                   |
+-------------------+----------+---------------------------------------------+
| symbols           | category | concatenated atomic symbols                 |
+-------------------+----------+---------------------------------------------+
'''
import numpy as np
import pandas as pd
from traitlets import Unicode
from exa.numerical import DataFrame, SparseDataFrame
from exa.relational.isotope import symbols_to_radii
from exatomic.math.distance import free_two_frame, periodic_two_frame


class BaseTwo(DataFrame):
    '''
    Base class for two body properties.

    See Also:
        Two body data are store depending on the boundary conditions of the
        system: :class:`~exatomic.two.FreeTwo` or :class:`~exatomic.two.PeriodicTwo`.
    '''
    _indices = ['two']
    _columns = ['dx', 'dy', 'dz', 'atom0', 'atom1', 'distance', 'frame']
    _groupbys = ['frame']
    _categories = {'frame': np.int64, 'symbols': str, 'atom0': np.int64,
                   'atom1': np.int64}

    def _bond_traits(self, label_mapper):
        '''
        Traits representing bonded atoms are reported as two lists of equal
        length with atom labels.
        '''
        bonded = self.ix[self['bond'] == True, ['atom0', 'atom1', 'frame']]
        lbl0 = bonded['atom0'].map(label_mapper)
        lbl1 = bonded['atom1'].map(label_mapper)
        lbl = pd.concat((lbl0, lbl1), axis=1)
        lbl['frame'] = bonded['frame']
        bond_grps = lbl.groupby('frame')
        grps = self.groupby('frame')
        n = grps.ngroups
        b0 = np.empty((n, ), dtype='O')
        b1 = b0.copy()
        for i, (frame, grp) in enumerate(grps):
            if frame in bond_grps.groups:
                b0[i] = bond_grps.get_group(frame)['atom0'].astype(str).values
                b1[i] = bond_grps.get_group(frame)['atom1'].astype(str).values
            else:
                b0[i] = []
                b1[i] = []
        b0 = Unicode(pd.Series(b0).to_json(orient='values')).tag(sync=True)
        b1 = Unicode(pd.Series(b1).to_json(orient='values')).tag(sync=True)
        return {'two_bond0': b0, 'two_bond1': b1}


class FreeTwo(BaseTwo):
    '''
    Free boundary condition two body properties table.
    '''
    pass


class PeriodicTwo(BaseTwo):
    '''
    Periodic boundary condition two body properties table.
    '''
    pass


def compute_two(universe, bond_extra=0.55, max_distance=19.0):
    '''
    Compute interatomic distances.
    '''
    if universe.frame.is_periodic:
        return compute_periodic_two(universe, bond_extra, max_distance)
    return compute_free_two(universe, bond_extra, max_distance)


def compute_free_two(universe, bond_extra=0.55, max_distance=19.0):
    '''
    Compute free boundary condition two body properties from an input universe.
    '''
    groups = universe.atom.groupby('frame')
    n = groups.ngroups
    dxs = np.empty((n, ), dtype='O')
    dys = np.empty((n, ), dtype='O')
    dzs = np.empty((n, ), dtype='O')
    ds = np.empty((n, ), dtype='O')
    idx0s = np.empty((n, ), dtype='O')
    idx1s = np.empty((n, ), dtype='O')
    fdxs = np.empty((n, ), dtype='O')
    for i, (frame, group) in enumerate(groups):
        x = group['x'].values.astype(np.float64)
        y = group['y'].values.astype(np.float64)
        z = group['z'].values.astype(np.float64)
        idx = group.index.values.astype(np.int64)
        dx, dy, dz, idx0, idx1, fdx, d = free_two_frame(x, y, z, idx, frame)
        dxs[i] = dx
        dys[i] = dy
        dzs[i] = dz
        idx0s[i] = idx0
        idx1s[i] = idx1
        fdxs[i] = fdx
        ds[i] = d
    dxs = np.concatenate(dxs)
    dys = np.concatenate(dys)
    dzs = np.concatenate(dzs)
    idx0s = pd.Series(np.concatenate(idx0s), dtype='category')
    idx1s = pd.Series(np.concatenate(idx1s), dtype='category')
    fdxs = pd.Series(np.concatenate(fdxs), dtype='category')
    ds = np.concatenate(ds)
    two = pd.DataFrame.from_dict({'dx': dxs, 'dy': dys, 'dz': dzs, 'distance': ds,
                                  'frame': fdxs, 'atom0': idx0s, 'atom1': idx1s})
    mapper = universe.atom['symbol'].astype(str)
    two['symbol0'] = two['atom0'].astype(np.int64).map(mapper)
    two['symbol1'] = two['atom1'].astype(np.int64).map(mapper)
    two['symbols'] = (two['symbol0'] + two['symbol1']).astype('category')
    del two['symbol0']
    del two['symbol1']
    mapper = symbols_to_radii()
    two['mbl'] = two['symbols'].astype(str).map(mapper) + bond_extra
    two['bond'] = two['distance'] < two['mbl']
    del two['mbl']
    return FreeTwo(two)


def compute_periodic_two(universe, bond_extra=0.55, max_distance=19.0):
    '''
    Compute periodic two body properties.
    '''
    grps = universe.atom[['x', 'y', 'z', 'frame']].copy()
    grps.update(universe.unit_atom)
    grps = grps.groupby('frame')
    n = grps.ngroups
    dxs = np.empty((n, ), dtype='O')
    dys = np.empty((n, ), dtype='O')
    dzs = np.empty((n, ), dtype='O')
    idx0s = np.empty((n, ), dtype='O')
    idx1s = np.empty((n, ), dtype='O')
    fdxs = np.empty((n, ), dtype='O')
    pxs = np.empty((n, ), dtype='O')
    pys = np.empty((n, ), dtype='O')
    pzs = np.empty((n, ), dtype='O')
    ds = np.empty((n, ), dtype='O')
    for i, (frame, grp) in enumerate(grps):
        ux = grp['x'].values.astype(np.float64)
        uy = grp['y'].values.astype(np.float64)
        uz = grp['z'].values.astype(np.float64)
        idx = grp.index.values.astype(np.int64)
        rx, ry, rz = universe.frame.ix[frame, ['rx', 'ry', 'rz']]
        dx, dy, dz, d, idx0, idx1, fdx, px, py, pz = periodic_two_frame(ux, uy, uz, rx, ry, rz, idx, frame, max_distance)
        dxs[i] = dx
        dys[i] = dy
        dzs[i] = dz
        idx0s[i] = idx0
        idx1s[i] = idx1
        pxs[i] = px
        pys[i] = py
        pzs[i] = pz
        ds[i] = d
        fdxs[i] = fdx
    dxs = np.concatenate(dxs)
    dys = np.concatenate(dys)
    dzs = np.concatenate(dzs)
    ds = np.concatenate(ds)
    idx0s = pd.Series(np.concatenate(idx0s), dtype='category')
    idx1s = pd.Series(np.concatenate(idx1s), dtype='category')
    fdxs = pd.Series(np.concatenate(fdxs), dtype='category')
    pxs = np.concatenate(pxs)
    pys = np.concatenate(pys)
    pzs = np.concatenate(pzs)
    two = pd.DataFrame.from_dict({'dx': dxs, 'dy': dys, 'dz': dzs, 'distance': ds,
                                  'frame': fdxs, 'atom0': idx0s, 'atom1': idx1s})
    two = two.dropna(how='any', axis=0)
    patom = pd.DataFrame.from_dict({'x': pxs, 'y': pys, 'z': pzs})
    patom = patom.dropna(how='all', subset=['x', 'y', 'z'])
    mapper = universe.atom['symbol'].astype(str)
    two['symbol0'] = two['atom0'].astype(np.int64).map(mapper)
    two['symbol1'] = two['atom1'].astype(np.int64).map(mapper)
    two['symbols'] = (two['symbol0'] + two['symbol1']).astype('category')
    del two['symbol0']
    del two['symbol1']
    mapper = symbols_to_radii()
    two['mbl'] = two['symbols'].astype(str).map(mapper) + bond_extra
    two['bond'] = two['distance'] < two['mbl']
    del two['mbl']
    return two, patom


def compute_bond_count(universe):
    '''
    Computes bond count (number of bonds associated with a given atom index).

    Args:
        universe (:class:`~exatomic.universe.Universe`): Atomic universe

    Returns:
        counts (:class:`~numpy.ndarray`): Bond counts

    Note:
        For both periodic and non-periodic universes, counts returned are
        atom indexed. Counts for projected atoms have no meaning/are not
        computed during two body property calculation.
    '''
    stack = universe.two.ix[universe.two['bond'] == True, ['atom0', 'atom1']].stack()
    return stack.value_counts().sort_index()


#def bond_summary_by_label_pairs(universe, *labels, length='A', stdev=False,
#                                stderr=False, variance=False, ncount=False):
#    '''
#    Compute a summary of bond lengths by label pairs
#
#    Args:
#        universe: The atomic container
#        \*labels: Any number of label pairs (e.g. ...paris(uni, (0, 1), (1, 0), ...))
#        length (str): Output length unit (default Angstrom)
#        stdev (bool): Compute the standard deviation of the mean (default false)
#        stderr (bool): Compute the standard error in the mean (default false)
#        variance (bool): Compute the variance in the mean (default false)
#        ncount (bool): Include the data point count (default false)
#
#    Returns:
#        summary (:class:`~pandas.DataFrame`): Bond length dataframe
#    '''
#    l0, l1 = list(zip(*labels))
#    l0 = np.array(l0, dtype=np.int64)
#    l1 = np.array(l1, dtype=np.int64)
#    ids = unordered_pairing(l0, l1)
#    bonded = universe.two[universe.two['bond'] == True].copy()
#    if universe.is_periodic:
#        bonded['atom0'] = bonded['prjd_atom0'].map(universe.projected_atom['atom'])
#        bonded['atom1'] = bonded['prjd_atom1'].map(universe.projected_atom['atom'])
#    bonded['label0'] = bonded['atom0'].map(universe.atom['label'])
#    bonded['label1'] = bonded['atom1'].map(universe.atom['label'])
#    bonded['id'] = unordered_pairing(bonded['label0'].values.astype(np.int64),
#                                     bonded['label1'].values.astype(np.int64))
#    return bonded[bonded['id'].isin(ids)]
#    grps = bonded[bonded['id'].isin(ids)].groupby('id')
#    df = grps['distance'].mean().reset_index()
#    if variance:
#        df['variance'] = grps['distance'].var().reset_index()['distance']
#        df['variance'] *= Length['au', length]
#    if stderr:
#        df['stderr'] = grps['distance'].std().reset_index()['distance']
#        df['stderr'] /= np.sqrt(grps['distance'].size().values[0])
#        df['stderr'] *= Length['au', length]
#    if stdev:
#        df['stdev'] = grps['distance'].std().reset_index()['distance']
#        df['stdev'] *= Length['au', length]
#    if ncount:
#        df['count'] = grps['distance'].size().reset_index()[0]
#    mapper = bonded.drop_duplicates('id').set_index('id')
#    df['symbols'] = df['id'].map(mapper['symbols'])
#    df['distance'] *= Length['au', length]
#    df['label0'] = df['id'].map(mapper['label0'])
#    df['label1'] = df['id'].map(mapper['label1'])
#    del df['id']
#    return df
#
#
#def n_nearest_distances_by_symbols(universe, a, b, n, length='A', stdev=False,
#                                   stderr=False, variance=False, ncount=False):
#    '''
#    Compute a distance summary of the n nearest pairs of symbols, (a, b).
#
#    Args:
#        universe: The atomic universe
#        a (str): Symbol string
#        b (str): Symbol string
#        n (int): Number of distances to include
#        stdev (bool): Compute the standard deviation of the mean (default false)
#        stderr (bool): Compute the standard error in the mean (default false)
#        variance (bool): Compute the variance in the mean (default false)
#        ncount (bool): Include the data point count (default false)
#
#    Returns:
#        summary (:class:`~pandas.DataFrame`): Distance summary dataframe
#    '''
#    def compute(group):
#        return group.sort_values('distance').iloc[:n]
#
#    df = universe.two[universe.two['symbols'].isin([a+b, b+a])]
#    df = df.groupby('frame').apply(compute)
#    df['pair'] = list(range(n)) * (len(df) // n)
#    pvd = df.pivot('frame', 'pair', 'distance')
#    df = pvd.mean(0).reset_index()
#    df.columns = ['pair', 'distance']
#    df['distance'] *= Length['au', length]
#    if stdev:
#        df['stdev'] = pvd.std().reset_index()[0]
#        df['stdev'] *= Length['au', length]
#    if stderr:
#        df['stderr'] = pvd.std().reset_index()[0]
#        df['stderr'] /= np.sqrt(len(pvd))
#        df['stderr'] *= Length['au', length]
#    if variance:
#        df['variance'] = pvd.var().reset_index()[0]
#        df['variance'] *= Length['au', length]
#    if ncount:
#        df['count'] = pvd.shape[0]
#    return df
#
