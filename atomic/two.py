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
| atom0             | integer  | foreign key to :class:`~atomic.atom.Atom` |                      |
+-------------------+----------+-------------------------------------------+
| atom1             | integer  | foreign key to :class:`~atomic.atom.Atom` |                      |
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
from sklearn.neighbors import NearestNeighbors
from exa import DataFrame
from exa.algorithms import pdist
from atomic import Isotope


max_atoms_per_frame = 2000
max_frames = 50
bond_extra = 0.5
dmin = 0.3
dmax = 11.3


class Two(DataFrame):
    '''
    The two body property dataframe includes interatomic distances and bonds.
    '''
    _boundary_type = 0
    _indices = ['two']
    _columns = ['distance', 'atom0', 'atom1', 'frame']
    _groupbys = ['frame']
    _categories = {'frame': np.int64, 'symbols': str}

    def _get_bond_traits(self, labels):
        '''
        Generate bond traits for the notebook widget.
        '''
        df = self[self['bond'] == True].copy()
        df['label0'] = df['atom0'].map(labels)
        df['label1'] = df['atom1'].map(labels)
        grps = df.groupby('frame')
        b0 = grps.apply(lambda g: g['label0'].astype(np.int64).values).to_json(orient='values')
        b1 = grps.apply(lambda g: g['label1'].astype(np.int64).values).to_json(orient='values')
        del grps, df
        return {'two_bond0': Unicode(b0).tag(sync=True), 'two_bond1': Unicode(b1).tag(sync=True)}

    def compute_bond_count(self):
        '''
        Compute the bond count for each atom.

        Returns:
            s (:class:`~pandas.Series`): Bond counts for each atom index
        '''
        bonded = self[self['bond'] == True]
        b0 = bonded.groupby('prjd_atom0').size()
        b1 = bonded.groupby('prjd_atom1').size()
        return b0.add(b1, fill_value=0)


def compute_two_body(universe, k=None, dmax=dmax, dmin=dmin, bond_extra=bond_extra,
                     compute_bonds=True, compute_symbols=True):
    '''
    Compute two body information given a universe.

    Bonds are computed semi-empirically (if requested - default True):

    .. math::

        distance(A, B) < covalent\_radius(A) + covalent\_radius(B) + bond\_extra

    Args:
        universe (:class:`~atomic.universe.Universe`): Chemical universe
        k (int): Number of distances (per atom) to compute (optional)
        dmax (float): Max distance of interest (larger distances are ignored)
        dmin (float): Min distance of interest (smaller distances are ignored)
        bond_extra (float): Extra distance to include when determining bonds (see above)
        compute_bonds (bool): Compute bonds from distances (default True)
        compute_symbols (bool): Compute symbol pairs (default True)

    Returns:
        df (:class:`~atomic.twobody.TwoBody`): Two body property table
    '''
    nat = universe.frame['atom_count'].max()
    nf = len(universe.frame)
    if universe.is_periodic:
        if nat < max_atoms_per_frame and nf < max_frames:
            return _periodic_in_mem()
        else:
            raise NotImplementedError('Out of core two body not implemented')
    else:
        if nat < max_atoms_per_frame and nf < max_frames:
            return _free_in_mem(universe, dmax=dmax, dmin=dmin, bond_extra=bond_extra,
                                compute_bonds=compute_bonds, compute_symbols=compute_symbols)
        else:
            raise NotImplementedError('Out of core two body not implemented')


def _free_in_mem(universe, dmax, dmin, bond_extra, compute_symbols, compute_bonds):
    '''
    Free boundary condition two body properties computed in memory.

    Args:
        universe (:class:`~atomic.universe.Universe`): The atomic universe
        dmax (float): Max distance of interest
        dmin (float): Minimum distance of interest
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
    two = DataFrame.from_dict({'atom0': atom0, 'atom1': atom1,
                               'distance': distance, 'frame': frames})
    two = two[(two['distance'] > dmin) & (two['distance'] < dmax)].reset_index(drop=True)
    two.index.names = ['two']
    if compute_symbols:
        symbols = universe.atom['symbol'].astype(str)
        two['symbol0'] = two['atom0'].map(symbols)
        two['symbol1'] = two['atom1'].map(symbols)
        del symbols
        two['symbols'] = two['symbol0'] + two['symbol1']
        two['symbols'] = two['symbols'].astype('category')
        del two['symbol0']
        del two['symbol1']
    if compute_bonds:
        two['mbl'] = two['symbols'].astype(str).map(Isotope.symbols_to_radii_map)
        two['mbl'] += bond_extra
        two['bond'] = two['distance'] < two['mbl']
        del two['mbl']
    return Two(two)


def _periodic_in_mem(universe, k, dmax, dmin, bond_extra, compute_symbols, compute_bonds):
    '''
    Periodic boundary condition two body properties computed in memory.

    Args:
        universe (:class:`~atomic.universe.Universe`): The atomic universe
        k (int): Number of distances to compute
        dmax (float): Max distance of interest
        dmin (float): Minimum distance of interest
        bond_extra (float): Extra distance to add when determining bonds
        compute_symbols (bool): Compute symbol pairs
        compute_bonds (bool): Compute (semi-empirical) bonds

    Returns:
        two (:class:`~atomic.two.Two`): Two body property dataframe
    '''
    k = k if k else universe.frame['atom_count'].max() - 1
    prjd_grps = universe.projected_atom.groupby('frame')
    unit_grps = universe.unit_atom.groupby('frame')
    n = prjd_grps.ngroups
    nn = k**2
    distances = np.empty((n, nn), dtype=np.float64)
    index1 = np.empty((n, nn), dtype=np.float64)
    index2 = np.empty((n, nn), dtype=np.float64)
    frames = np.empty((n, nn), dtype=np.float64)
    for i, (frame, prjd) in enumerate(prjd_grps):
        pxyz = prjd[['x', 'y', 'z']]
        uxyz = unit_grps.get_group(frame)[['x', 'y', 'z']]
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')    # k-d tree search
        dists, idxs = nn.fit(pxyz).kneighbors(uxyz)
        distances[i, :] = dists.ravel()
        index1[i, :] = prjd.iloc[np.tile(idxs[:, 0], k)].index.values
        index2[i, :] = prjd.iloc[idxs.ravel()].index.values
        frames[i, :] = np.repeat(frame, len(index1[i]))
    distances = distances.ravel()
    index1 = index1.ravel()
    index2 = index2.ravel()
    frames = frames.ravel()
    two = ProjectedTwo.from_dict({'distance': distances, 'frame': frames,
                                  'prjd_atom0': index1, 'prjd_atom1': index2})  # We will use prjd_atom0/2 to deduplicate data
    two = two[(two['distance'] > dmin) & (two['distance'] < dmax)]
    two['id'] = unordered_pairing_function(two['prjd_atom0'].values, two['prjd_atom1'].values)
    two = two.drop_duplicates('id').reset_index(drop=True)
    del two['id']
    symbols = universe.projected_atom['symbol']
    two['symbol1'] = two['prjd_atom0'].map(symbols)
    two['symbol2'] = two['prjd_atom1'].map(symbols)
    del symbols
    two['symbols'] = two['symbol1'].astype('O') + two['symbol2'].astype('O')
    two['symbols'] = two['symbols'].astype('category')
    del two['symbol1']
    del two['symbol2']
    two['mbl'] = two['symbols'].map(Isotope.symbols_to_radii_map)
    two['mbl'] += bond_extra
    two['bond'] = two['distance'] < two['mbl']
    del two['mbl']
    two.index.names = ['prjd_two']
    prjd_two = two.index.tolist() * 2
    prjd_atom = two['prjd_atom0'].tolist() + two['prjd_atom1'].tolist()
    atomtwo = ProjectedAtomTwo.from_dict({'prjd_two': prjd_two, 'prjd_atom': prjd_atom})
    return two, atomtwo
