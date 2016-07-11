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
#import numpy as np
#import pandas as pd
#from traitlets import Unicode
#from scipy.spatial import cKDTree
#from exa import DataFrame
#from exa.algorithms import pdist, unordered_pairing
#from exa.relational.isotope import symbols_to_radii
#from exatomic import Length
#from exatomic import Isotope, Length
#
#
#max_atoms_per_frame = 300
#max_frames = 500
#max_atoms_per_frame_periodic = 200
#max_frames_periodic = 300
#bond_extra = 0.45
#dmin = 0.3
#dmax = 11.3
#
#
#class Two(DataFrame):
#    '''
#    The two body property dataframe includes interatomic distances and bonds.
#    '''
#    _index_prefix = 'atom'
#    _indices = ['two']
#    _columns = ['distance', 'atom0', 'atom1', 'frame']
#    _groupbys = ['frame']
#    _categories = {'frame': np.int64, 'symbols': str, 'atom0': np.int64,
#                   'atom1': np.int64}
#
#    def _get_bond_traits(self, atom):
#        '''
#        Generate bond traits for the notebook widget.
#        '''
#        ip = self._index_prefix
#        a0 = ip + '0'
#        a1 = ip + '1'
#        label_mapper = atom['label']
#        grps = atom.groupby('frame')
#        bonded = self.ix[self['bond'] == True, [a0, a1, 'frame']]
#        label0 = bonded[a0].map(label_mapper)
#        label1 = bonded[a1].map(label_mapper)
#        label = pd.concat((label0, label1), axis=1)
#        label['frame'] = bonded['frame']
#        bgrps = label.groupby('frame')
#        b0 = np.empty((grps.ngroups, ), dtype='O')
#        b1 = np.empty((grps.ngroups, ), dtype='O')
#        for i, (frame, grp) in enumerate(grps):
#            if frame in bgrps.groups:
#                b0[i] = bgrps.get_group(frame)[a0].values.astype(np.int64)
#                b1[i] = bgrps.get_group(frame)[a1].values.astype(np.int64)
#            else:
#                b0[i] = []
#                b1[i] = []
#        b0 = pd.Series(b0).to_json(orient='values')
#        b1 = pd.Series(b1).to_json(orient='values')
#        return {'two_bond0': Unicode(b0).tag(sync=True), 'two_bond1': Unicode(b1).tag(sync=True)}
#
#    def bond_summary(self, length='au'):
#        '''
#        Generate a summary table of bond lengths
#        '''
#        return self[self['bond'] == True].groupby('symbols')['distance'].mean().dropna() * Length['au', length]
#
#
#class PeriodicTwo(Two):
#    '''
#    The two body property dataframe but computed using the periodic algorithm.
#    The atom indices match those present in the projected atom dataframe.
#    '''
#    _index_prefix = 'prjd_atom'
#    _indices = ['pbc_two']
#    _columns = ['distance', 'prjd_atom0', 'prjd_atom1', 'frame']
#    _categories = {'frame': np.int64, 'symbols': str, 'prjd_atom0': np.int64,
#                   'prjd_atom1': np.int64}
#
#    def mapped_atom(self, mapper):
#        '''
#        Maps the projected atom columns back onto their atom indices using a
#        mapper.
#
#        Args:
#            mapper (:class:`~pandas.Series`): Projected atom mapper
#
#        Returns:
#            tup: Tuple of two series objects corresponding to prjd_atom0 and prjd_atom1
#        '''
#        self._revert_categories()
#        b0 = bonds['prjd_atom0'].map(mapper)
#        b1 = bonds['prjd_atom1'].map(mapper)
#        self._set_categories()
#        return b0, b1
#
#
#def compute_two_body(universe, k=None, dmax=dmax, dmin=dmin, bond_extra=bond_extra,
#                     compute_bonds=True, compute_symbols=True, in_mem=False):    # in_mem is undocumented on purpose...
#    '''
#    Compute two body information given a universe.
#
#    Bonds are computed semi-empirically (if requested - default true):
#
#    .. math::
#
#        distance(A, B) < covalent\_radius(A) + covalent\_radius(B) + bond\_extra
#
#    Args:
#        universe (:class:`~exatomic.universe.Universe`): Chemical universe
#        k (int): Number of distances (per atom) to compute (optional)
#        dmax (float): Max distance of interest (larger distances are ignored)
#        dmin (float): Min distance of interest (smaller distances are ignored)
#        bond_extra (float): Extra distance to include when determining bonds (see above)
#        compute_bonds (bool): Compute bonds from distances (default: true)
#        compute_symbols (bool): Compute symbol pairs (default: true)
#
#    Returns:
#        df (:class:`~exatomic.twobody.TwoBody`): Two body property table
#
#    Warning:
#        Computing periodic distances can use a large amount of memory (>16 GB)
#        and take up to 5 minutes (on a modern machine, with universes of more
#        than a few thousand frames)!
#    '''
#    nat = universe.frame['atom_count'].max()
#    nf = len(universe.frame)
#    if universe.is_periodic:
#        if (nat < max_atoms_per_frame_periodic and nf < max_frames_periodic) or in_mem:
#            k = k if k else nat - 1
#            return _periodic_in_mem(universe, k, dmin, dmax, bond_extra, compute_symbols,
#                                    compute_bonds)
#        else:
#            raise NotImplementedError('Out of core two body not implemented')
#    else:
#        if (nat < max_atoms_per_frame and nf < max_frames) or in_mem:
#            return _free_in_mem(universe, dmin, dmax, bond_extra, compute_symbols,
#                                compute_bonds)
#        else:
#            raise NotImplementedError('Out of core two body not implemented')
#
#
#def _free_in_mem(universe, dmin, dmax, bond_extra, compute_symbols,
#                 compute_bonds):
#    '''
#    Free boundary condition two body properties computed in memory.
#
#    Args:
#        universe (:class:`~exatomic.universe.Universe`): The atomic universe
#        dmin (float): Minimum distance of interest
#        dmax (float): Max distance of interest
#        bond_extra (float): Extra distance to add when determining bonds
#        compute_symbols (bool): Compute symbol pairs
#        compute_bonds (bool): Compute (semi-empirical) bonds
#
#    Returns:
#        two (:class:`~exatomic.two.Two`): Two body property dataframe
#    '''
#    atom_groups = universe.atom.groupby('frame')
#    n = atom_groups.ngroups
#    atom0 = np.empty((n, ), dtype='O')
#    atom1 = np.empty((n, ), dtype='O')
#    distance = np.empty((n, ), dtype='O')
#    frames = np.empty((n, ), dtype='O')
#    for i, (frame, atom) in enumerate(atom_groups):
#        xyz = atom[['x', 'y', 'z']].values
#        dists, i0, i1 = pdist(xyz)
#        atom0[i] = atom.iloc[i0].index.values
#        atom1[i] = atom.iloc[i1].index.values
#        distance[i] = dists
#        frames[i] = [frame] * len(dists)
#    distance = np.concatenate(distance).astype(np.float64)
#    atom0 = np.concatenate(atom0).astype(np.int64)
#    atom1 = np.concatenate(atom1).astype(np.int64)
#    frames = np.concatenate(frames).astype(np.int64)
#    df = pd.DataFrame.from_dict({'atom0': atom0, 'atom1': atom1,
#                                 'distance': distance, 'frame': frames})
#    df = df[(df['distance'] > dmin) & (df['distance'] < dmax)].reset_index(drop=True)
#    df['frame'] = df['frame'].astype('category')
#    df['atom0'] = df['atom0'].astype('category')
#    df['atom1'] = df['atom1'].astype('category')
#    if compute_symbols:
#        symbols = universe.atom['symbol'].astype(str)
#        df['symbol0'] = df['atom0'].map(symbols)
#        df['symbol1'] = df['atom1'].map(symbols)
#        del symbols
#        df['symbols'] = df['symbol0'] + df['symbol1']
#        df['symbols'] = df['symbols'].astype('category')
#        del df['symbol0']
#        del df['symbol1']
#    if compute_bonds:
#        df['mbl'] = df['symbols'].astype(str).map(symbols_to_radii)
#        df['mbl'] += bond_extra
#        df['bond'] = df['distance'] < df['mbl']
#        del df['mbl']
#    return Two(df)
#
#
#def _periodic_in_mem(universe, k, dmin, dmax, bond_extra, compute_symbols,
#                     compute_bonds):
#    '''
#    Periodic boundary condition two body properties computed in memory.
#
#    Args:
#        universe (:class:`~exatomic.universe.Universe`): The atomic universe
#        k (int): Number of distances to compute
#        dmin (float): Minimum distance of interest
#        dmax (float): Max distance of interest
#        bond_extra (float): Extra distance to add when determining bonds
#        compute_symbols (bool): Compute symbol pairs
#        compute_bonds (bool): Compute (semi-empirical) bonds
#
#    Returns:
#        two (:class:`~exatomic.two.Two`): Two body property dataframe
#    '''
#    prjd_grps = universe.projected_atom.groupby('frame')
#    unit_grps = universe.unit_atom.groupby('frame')
#    n = prjd_grps.ngroups
#    distances = np.empty((n, ), dtype='O')
#    index1 = np.empty((n, ), dtype='O')
#    index2 = np.empty((n, ), dtype='O')
#    frames = np.empty((n, ), dtype='O')
#    for i, (frame, prjd) in enumerate(prjd_grps):
#        pxyz = prjd[['x', 'y', 'z']]
#        uxyz = unit_grps.get_group(frame)[['x', 'y', 'z']]
#        dists, idxs = cKDTree(pxyz).query(uxyz, k=k)    # Distances computed using k-d tree
#        distances[i] = dists.ravel()
#        index1[i] = prjd.iloc[np.repeat(idxs[:, 0], k)].index.values
#        index2[i] = prjd.iloc[idxs.ravel()].index.values
#        frames[i] = np.repeat(frame, len(index1[i]))
#    distances = np.concatenate(distances)
#    index1 = np.concatenate(index1)
#    index2 = np.concatenate(index2)
#    frames = np.concatenate(frames)
#    df = pd.DataFrame.from_dict({'distance': distances, 'frame': frames,
#                                 'prjd_atom0': index1, 'prjd_atom1': index2})
#    df['prjd_atom0'] = df['prjd_atom0'].astype('category')
#    df['prjd_atom1'] = df['prjd_atom1'].astype('category')
#    df = df[(df['distance'] > dmin) & (df['distance'] < dmax)]#.sort_values('distance')
#    atom = universe.projected_atom['atom']
#    df['atom0'] = df['prjd_atom0'].map(atom)
#    df['atom1'] = df['prjd_atom1'].map(atom)
#    del atom
#    df['id'] = unordered_pairing(df['atom0'].values.astype(np.int64),
#                                 df['atom1'].values.astype(np.int64))
#    df = df.drop_duplicates('id').reset_index(drop=True)
#    del df['id']
#    del df['atom0']
#    del df['atom1']
#    symbols = universe.projected_atom['symbol']
#    df['symbol1'] = df['prjd_atom0'].map(symbols)
#    df['symbol2'] = df['prjd_atom1'].map(symbols)
#    del symbols
#    df['symbols'] = df['symbol1'].astype(str) + df['symbol2'].astype(str)
#    del df['symbol1']
#    del df['symbol2']
#    df['symbols'] = df['symbols'].astype('category')
#    df['mbl'] = df['symbols'].map(symbols_to_radii)
#    if not compute_symbols:
#        del df['symbols']
#    df['mbl'] += bond_extra
#    if compute_bonds:
#        df['bond'] = df['distance'] < df['mbl']
#    del df['mbl']
#    return PeriodicTwo(df)
#
#
#def compute_bond_count(universe):
#    '''
#    Computes bond count (number of bonds associated with a given atom index).
#
#    Args:
#        universe (:class:`~exatomic.universe.Universe`): Atomic universe
#
#    Returns:
#        counts (:class:`~numpy.ndarray`): Bond counts
#
#    Note:
#        For both periodic and non-periodic universes, counts returned are
#        atom indexed. Counts for projected atoms have no meaning/are not
#        computed during two body property calculation.
#    '''
#    if universe.is_periodic:
#        mapper = universe.projected_atom['atom']
#        bonds = universe.two.ix[(universe.two['bond'] == True), ['prjd_atom0', 'prjd_atom1']].stack().astype(np.int64)
#        bonds = bonds.map(mapper)
#        return bonds.value_counts()
#    else:
#        bonds = universe.two.ix[(universe.two['bond'] == True), ['atom0', 'atom1']].stack().value_counts()
#        return bonds
#
#
#def compute_projected_bond_count(universe):
#    '''
#    The projected bond count doesn't have physical meaning but can be useful
#    in certain cases (e.g. visual atom selection).
#    '''
#    if not universe.is_periodic:
#        raise TypeError('Is this a periodic universe? Check frame for periodic column.')
#    bc = universe.two.ix[(universe.two['bond'] == True), ['prjd_atom0', 'prjd_atom1']].stack().value_counts()
#    return bc.astype(np.int64)
#
#
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
