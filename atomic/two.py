# -*- coding: utf-8 -*-
'''
Two Body Properties DataFrame
===============================
This module provides various functions for computing two body properties (e.g.
interatomic distances). While this may seem like a trivial calculation, it is
not; it is a combinatorial problem and fast algorithms for it are an outstanding
problem in computational science.

+-------------------+----------+-------------------------------------------+
| Column            | Type     | Description                               |
+===================+==========+===========================================+
| atom0             | integer  | foreign key to :class:`~atomic.atom.Atom` |
+-------------------+----------+-------------------------------------------+
| atom1             | integer  | foreign key to :class:`~atomic.atom.Atom` |
+-------------------+----------+-------------------------------------------+
| distance          | float    | distance between atom0 and atom1          |
+-------------------+----------+-------------------------------------------+
| bond              | boolean  | True if bond                              |
+-------------------+----------+-------------------------------------------+
| frame             | category | non-unique integer (req.)                 |
+-------------------+----------+-------------------------------------------+
| symbols           | category | concatenated atomic symbols               |
+-------------------+----------+-------------------------------------------+
'''
import numpy as np
import pandas as pd
from traitlets import Unicode
from scipy.spatial import cKDTree
from exa import DataFrame
from exa.algorithms import pdist, unordered_pairing
from atomic import Isotope


max_atoms_per_frame = 1000
max_frames = 2000
max_atoms_per_frame_periodic = 500
max_frames_periodic = 1000
bond_extra = 0.25
dmin = 0.3
dmax = 11.3


class Two(DataFrame):
    '''
    The two body property dataframe includes interatomic distances and bonds.
    '''
    _index_prefix = 'atom'
    _indices = ['two']
    _columns = ['distance', 'atom0', 'atom1', 'frame']
    _groupbys = ['frame']
    _categories = {'frame': np.int64, 'symbols': str}

    def _get_bond_traits(self, labels):
        '''
        Generate bond traits for the notebook widget.
        '''
        df = self.ix[(self['bond'] == True), [self._index_prefix + '0', self._index_prefix + '1', 'frame']].copy()
        df['label0'] = df[self._index_prefix + '0'].map(labels)
        df['label1'] = df[self._index_prefix + '1'].map(labels)
        grps = df.groupby('frame')
        b0 = grps.apply(lambda g: g['label0'].astype(np.int64).values).to_json(orient='values')
        b1 = grps.apply(lambda g: g['label1'].astype(np.int64).values).to_json(orient='values')
        del grps, df
        return {'two_bond0': Unicode(b0).tag(sync=True), 'two_bond1': Unicode(b1).tag(sync=True)}


class PeriodicTwo(Two):
    '''
    The two body property dataframe but computed using the periodic algorithm.
    The atom indices match those present in the projected atom dataframe.
    '''
    _index_prefix = 'prjd_atom'
    _indices = ['pbc_two']
    _columns = ['distance', 'prjd_atom0', 'prjd_atom1', 'frame']

    def mapped_atom(self, mapper):
        '''
        Maps the projected atom columns back onto their atom indices using a
        mapper.

        Args:
            mapper (:class:`~pandas.Series`): Projected atom mapper

        Returns:
            tup: Tuple of two series objects corresponding to prjd_atom0 and prjd_atom1
        '''
        self._revert_categories()
        b0 = bonds['prjd_atom0'].map(mapper)
        b1 = bonds['prjd_atom1'].map(mapper)
        self._set_categories()
        return b0, b1


def compute_two_body(universe, k=None, dmax=dmax, dmin=dmin, bond_extra=bond_extra,
                     compute_bonds=True, compute_symbols=True, in_mem=False):    # in_mem is undocumented on purpose...
    '''
    Compute two body information given a universe.

    Bonds are computed semi-empirically (if requested - default true):

    .. math::

        distance(A, B) < covalent\_radius(A) + covalent\_radius(B) + bond\_extra

    Args:
        universe (:class:`~atomic.universe.Universe`): Chemical universe
        k (int): Number of distances (per atom) to compute (optional)
        dmax (float): Max distance of interest (larger distances are ignored)
        dmin (float): Min distance of interest (smaller distances are ignored)
        bond_extra (float): Extra distance to include when determining bonds (see above)
        compute_bonds (bool): Compute bonds from distances (default: true)
        compute_symbols (bool): Compute symbol pairs (default: true)

    Returns:
        df (:class:`~atomic.twobody.TwoBody`): Two body property table
    '''
    nat = universe.frame['atom_count'].max()
    nf = len(universe.frame)
    if universe.is_periodic:
        if (nat < max_atoms_per_frame_periodic and nf < max_frames_periodic) or in_mem:
            k = k if k else nat - 1
            return _periodic_in_mem(universe, k, dmin, dmax, bond_extra, compute_symbols,
                                    compute_bonds)
        else:
            raise NotImplementedError('Out of core two body not implemented')
    else:
        if (nat < max_atoms_per_frame and nf < max_frames) or in_mem:
            return _free_in_mem(universe, dmin, dmax, bond_extra, compute_symbols,
                                compute_bonds)
        else:
            raise NotImplementedError('Out of core two body not implemented')


def _free_in_mem(universe, dmin, dmax, bond_extra, compute_symbols,
                 compute_bonds):
    '''
    Free boundary condition two body properties computed in memory.

    Args:
        universe (:class:`~atomic.universe.Universe`): The atomic universe
        dmin (float): Minimum distance of interest
        dmax (float): Max distance of interest
        bond_extra (float): Extra distance to add when determining bonds
        compute_symbols (bool): Compute symbol pairs
        compute_bonds (bool): Compute (semi-empirical) bonds

    Returns:
        two (:class:`~atomic.two.Two`): Two body property dataframe
    '''
    atom_groups = universe.atom.groupby('frame')
    n = atom_groups.ngroups
    atom0 = np.empty((n, ), dtype='O')
    atom1 = np.empty((n, ), dtype='O')
    distance = np.empty((n, ), dtype='O')
    frames = np.empty((n, ), dtype='O')
    for i, (frame, atom) in enumerate(atom_groups):
        xyz = atom[['x', 'y', 'z']].values
        dists, i0, i1 = pdist(xyz)
        atom0[i] = atom.iloc[i0].index.values
        atom1[i] = atom.iloc[i1].index.values
        distance[i] = dists
        frames[i] = [frame] * len(dists)
    distance = np.concatenate(distance)
    atom0 = np.concatenate(atom0)
    atom1 = np.concatenate(atom1)
    frames = np.concatenate(frames)
    df = pd.DataFrame.from_dict({'atom0': atom0, 'atom1': atom1,
                               'distance': distance, 'frame': frames})
    df = df[(df['distance'] > dmin) & (df['distance'] < dmax)].reset_index(drop=True)
    df['atom0'] = df['atom0'].astype('category')
    df['atom1'] = df['atom1'].astype('category')
    if compute_symbols:
        symbols = universe.atom['symbol'].astype(str)
        df['symbol0'] = df['atom0'].map(symbols)
        df['symbol1'] = df['atom1'].map(symbols)
        del symbols
        df['symbols'] = df['symbol0'] + df['symbol1']
        df['symbols'] = df['symbols'].astype('category')
        del df['symbol0']
        del df['symbol1']
    if compute_bonds:
        df['mbl'] = df['symbols'].astype(str).map(Isotope.symbols_to_radii())
        df['mbl'] += bond_extra
        df['bond'] = df['distance'] < df['mbl']
        del df['mbl']
    return Two(df)


def _periodic_in_mem(universe, k, dmin, dmax, bond_extra, compute_symbols,
                     compute_bonds):
    '''
    Periodic boundary condition two body properties computed in memory.

    Args:
        universe (:class:`~atomic.universe.Universe`): The atomic universe
        k (int): Number of distances to compute
        dmin (float): Minimum distance of interest
        dmax (float): Max distance of interest
        bond_extra (float): Extra distance to add when determining bonds
        compute_symbols (bool): Compute symbol pairs
        compute_bonds (bool): Compute (semi-empirical) bonds

    Returns:
        two (:class:`~atomic.two.Two`): Two body property dataframe
    '''
    prjd_grps = universe.projected_atom.groupby('frame')
    unit_grps = universe.unit_atom.groupby('frame')
    n = prjd_grps.ngroups
    distances = np.empty((n, ), dtype='O')
    index1 = np.empty((n, ), dtype='O')
    index2 = np.empty((n, ), dtype='O')
    frames = np.empty((n, ), dtype='O')
    for i, (frame, prjd) in enumerate(prjd_grps):
        pxyz = prjd[['x', 'y', 'z']]
        uxyz = unit_grps.get_group(frame)[['x', 'y', 'z']]
        dists, idxs = cKDTree(pxyz).query(uxyz, k=k)    # Distances computed using k-d tree
        distances[i] = dists.ravel()
        index1[i] = prjd.iloc[np.repeat(idxs[:, 0], k)].index.values
        index2[i] = prjd.iloc[idxs.ravel()].index.values
        frames[i] = np.repeat(frame, len(index1[i]))
    distances = np.concatenate(distances)
    index1 = np.concatenate(index1)
    index2 = np.concatenate(index2)
    frames = np.concatenate(frames)
    df = pd.DataFrame.from_dict({'distance': distances, 'frame': frames,
                                  'prjd_atom0': index1, 'prjd_atom1': index2})
    df['prjd_atom0'] = df['prjd_atom0'].astype('category')
    df['prjd_atom1'] = df['prjd_atom1'].astype('category')
    df = df[(df['distance'] > dmin) & (df['distance'] < dmax)]
    atom = universe.projected_atom['atom']
    df['atom0'] = df['prjd_atom0'].map(atom)
    df['atom1'] = df['prjd_atom1'].map(atom)
    del atom
    df['id'] = unordered_pairing(df['atom0'].values, df['atom1'].values)
    df = df.drop_duplicates('id').reset_index(drop=True)
    del df['id']
    del df['atom0']
    del df['atom1']
    symbols = universe.projected_atom['symbol']
    df['symbol1'] = df['prjd_atom0'].map(symbols)
    df['symbol2'] = df['prjd_atom1'].map(symbols)
    del symbols
    df['symbols'] = df['symbol1'].astype(str) + df['symbol2'].astype(str)
    del df['symbol1']
    del df['symbol2']
    df['symbols'] = df['symbols'].astype('category')
    df['mbl'] = df['symbols'].map(Isotope.symbols_to_radii())
    if not compute_symbols:
        del df['symbols']
    df['mbl'] += bond_extra
    if compute_bonds:
        df['bond'] = df['distance'] < df['mbl']
    del df['mbl']
    return PeriodicTwo(df)


def compute_bond_count(universe):
    '''
    Computes bond count (number of bonds associated with a given atom index).

    Args:
        universe (:class:`~atomic.universe.Universe`): Atomic universe

    Returns:
        counts (:class:`~numpy.ndarray`): Bond counts

    Note:
        For both periodic and non-periodic universes, counts returned are
        atom indexed. Counts for projected atoms have no meaning/are not
        computed during two body property calculation.
    '''
    universe.two._revert_categories()
    bonds = universe.two[universe.two['bond'] == True]
    if universe.is_periodic:
        mapper = universe.projected_atom['atom']
        b0 = bonds['prjd_atom0'].map(mapper).value_counts()
        b1 = bonds['prjd_atom1'].map(mapper).value_counts()
    else:
        b0 = bonds.groupby('atom0').size()
        b1 = bonds.groupby('atom1').size()
    bc = b0.add(b1, fill_value=0).astype(np.int64)
    bc.index.names = ['atom']
    universe.two._set_categories()
    return bc
