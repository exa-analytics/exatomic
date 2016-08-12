# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Two Body Properties
##################################
This module provides functions for computing two body properties. A body can be,
for example, an atom (such that two body properties correspond to inter-atomic
distances - bonds), or a molecule (such that two body properties correspond to
distances between molecule centers of mass). The following table provides a
guide for the types of data found in two body tables provided by this module
(specifically for atom two body properties).

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
"""
import numpy as np
import pandas as pd
from traitlets import Unicode
from exa.numerical import DataFrame, SparseDataFrame
from exa.relational.isotope import symbol_to_radius
from exa.math.vector.cartesian import pdist_euc_dxyz_idx
from exatomic.algorithms.distance import periodic_pdist_euc_dxyz_idx


class AtomTwo(DataFrame):
    """
    Interatomic distances generated from the :class:`~exatomic.atom.Atom` table.
    """
    _index = 'two'
    _cardinal = ('frame', np.int64)
    _columns = ['dx', 'dy', 'dz', 'atom0', 'atom1', 'distance']
    _categories = {'symbols': str, 'atom0': np.int64, 'atom1': np.int64}

    def compute_bonds(self, symbols, mapper=None, bond_extra=0.45):
        """
        Update bonds based on radii and an extra factor (in atomic units).

        .. code-block:: Python

            symbols = universe.atom['symbol']
            mapper = {'C': 2.0}    # atomic units - Bohr
            bond_extra = 0.2       # ditto
            # updates universe.atom_two['bond']
            universe.atom_two.compute_bonds(symbols, mapper, bond_extra)

        Args:
            symbols: Series of symbols from the atom table (e.g. uni.atom['symbol'])
            mapper (dict): Dictionary of symbol, radii pairs (see note below)
            bond_extra (float): Extra additive factor to include when computing bonds

        Note:
            If the mapper is none, or is missing data, the computation will use
            default covalent radii available in :class:`~exa.relational.isotope`.
        """
        if mapper is None:
            mapper = symbol_to_radius()
        elif not all(symbol in mapper for symbol in symbols.unique()):
            sym2rad = symbol_to_radius()
            for symbol in symbols.unique():
                if symbol not in mapper:
                    mapper[symbol] = sym2rad[symbol]
        # Note that the mapper is transformed here, but the same name is used...
        mapper = symbols.astype(str).map(mapper)
        radius0 = self['atom0'].map(mapper)
        radius1 = self['atom1'].map(mapper)
        mbl = radius0 + radius1 + bond_extra
        self['bond'] = self['distance'] <= mbl

    def _bond_traits(self, label_mapper):
        """
        Traits representing bonded atoms are reported as two lists of equal
        length with atom labels.
        """
        bonded = self.ix[self['bond'] == True, ['atom0', 'atom1', 'frame']]
        lbl0 = bonded['atom0'].map(label_mapper)
        lbl1 = bonded['atom1'].map(label_mapper)
        lbl = pd.concat((lbl0, lbl1), axis=1)
        lbl['frame'] = bonded['frame']
        bond_grps = lbl.groupby('frame')
        frames = self['frame'].unique().astype(np.int64)
        b0 = np.empty((len(frames), ), dtype='O')
        b1 = b0.copy()
        for i, frame in enumerate(frames):
            try:
                b0[i] = bond_grps.get_group(frame)['atom0'].astype(np.int64).values
                b1[i] = bond_grps.get_group(frame)['atom1'].astype(np.int64).values
            except Exception:
                b0[i] = []
                b1[i] = []
        b0 = Unicode(pd.Series(b0).to_json(orient='values')).tag(sync=True)
        b1 = Unicode(pd.Series(b1).to_json(orient='values')).tag(sync=True)
        return {'two_bond0': b0, 'two_bond1': b1}


class MoleculeTwo(DataFrame):
    """
    """
    pass


def compute_atom_two(universe, mapper=None, bond_extra=0.45):
    """
    Compute interatomic distances and determine bonds.

    Args:
        universe: An atomic universe
        mapper (dict): Dictionary of symbol, radii pairs (see note below)
        bond_extra (float): Extra additive factor to include when computing bonds
    """
    if universe.frame.is_periodic():
        return compute_periodic_two_si(universe, mapper, bond_extra)
    return compute_free_two_si(universe, mapper, bond_extra)


def compute_free_two_si(universe, mapper=None, bond_extra=0.45):
    """
    Serial, in memory computation of two body properties for free boundary
    condition systems.
    """
    n = universe.frame['atom_count'].astype(np.int64)
    n = (n*(n - 1)//2).sum()
    dx = np.empty((n, ), dtype=np.float64)
    dy = np.empty((n, ), dtype=np.float64)
    dz = np.empty((n, ), dtype=np.float64)
    distance = np.empty((n, ), dtype=np.float64)
    atom0 = np.empty((n, ), dtype=np.int64)
    atom1 = np.empty((n, ), dtype=np.int64)
    fdx = np.empty((n, ), dtype=np.int64)
    start = 0
    stop = 0
    for frame, group in universe.atom.cardinal_groupby():
        x = group['x'].values.astype(np.float64)
        y = group['y'].values.astype(np.float64)
        z = group['z'].values.astype(np.float64)
        idx = group.index.values.astype(np.int64)
        dxx, dyy, dzz, dist, a0, a1 = pdist_euc_dxyz_idx(x, y, z, idx)
        stop += len(dxx)
        dx[start:stop] = dxx
        dy[start:stop] = dyy
        dz[start:stop] = dzz
        atom0[start:stop] = a0
        atom1[start:stop] = a1
        distance[start:stop] = dist
        fdx[start:stop] = frame
        start = stop
    atom0 = pd.Series(atom0, dtype='category')
    atom1 = pd.Series(atom1, dtype='category')
    fdx = pd.Series(fdx, dtype='category')
    two = pd.DataFrame.from_dict({'dx': dx, 'dy': dy, 'dz': dz, 'distance': distance,
                                  'atom0': atom0, 'atom1': atom1, 'frame': fdx})
    two = AtomTwo(two)
    two.compute_bonds(universe.atom['symbol'], mapper=mapper)
    return two


def compute_periodic_two_si(universe, mapper=None, bond_extra=0.45):
    """
    Compute periodic two body properties.
    """
    grps = universe.atom[['x', 'y', 'z', 'frame']].copy()
    grps['frame'] = grps['frame'].astype(np.int64)
    grps.update(universe.unit_atom)
    grps = grps.groupby('frame')
    n = grps.ngroups
    dx = np.empty((n, ), dtype=np.ndarray)
    dy = np.empty((n, ), dtype=np.ndarray)
    dz = np.empty((n, ), dtype=np.ndarray)
    atom0 = np.empty((n, ), dtype=np.ndarray)
    atom1 = np.empty((n, ), dtype=np.ndarray)
    distance = np.empty((n, ), dtype=np.ndarray)
    fdx = np.empty((n, ), dtype=np.ndarray)
    px = np.empty((n, ), dtype=np.ndarray)
    py = np.empty((n, ), dtype=np.ndarray)
    pz = np.empty((n, ), dtype=np.ndarray)
    start = 0
    stop = 0
    for i, (frame, grp) in enumerate(grps):
        ux = grp['x'].values.astype(np.float64)
        uy = grp['y'].values.astype(np.float64)
        uz = grp['z'].values.astype(np.float64)
        sidx = grp.index.values.astype(np.int64)
        rx, ry, rz = universe.frame.ix[frame, ['rx', 'ry', 'rz']]
        dxx, dyy, dzz, d, a0, a1, pxx, pyy, pzz = periodic_pdist_euc_dxyz_idx(ux, uy, uz, rx, ry, rz, sidx)
        nnn = len(dxx)
        stop += nnn
        dx[i] = dxx
        dy[i] = dyy
        dz[i] = dzz
        distance[i] = d
        atom0[i] = a0
        atom1[i] = a1
        px[i] = pxx
        py[i] = pyy
        pz[i] = pzz
        fdx[i] = [frame for j in range(nnn)]
        start = stop
    dx = np.concatenate(dx)
    dy = np.concatenate(dy)
    dz = np.concatenate(dz)
    distance = np.concatenate(distance)
    px = np.concatenate(px)
    py = np.concatenate(py)
    pz = np.concatenate(pz)
    atom0 = pd.Series(np.concatenate(atom0), dtype='category')
    atom1 = pd.Series(np.concatenate(atom1), dtype='category')
    fdx = pd.Series(np.concatenate(fdx), dtype='category')
    two = pd.DataFrame.from_dict({'dx':dx, 'dy': dy, 'dz': dz, 'distance': distance,
                                  'atom0': atom0, 'atom1': atom1, 'frame': fdx})
    patom = pd.DataFrame.from_dict({'x': px, 'y': py, 'z': pz})
    two = AtomTwo(two)
    two.compute_bonds(universe.atom['symbol'], mapper=mapper)
    return two, patom


def compute_bond_count(universe):
    """
    Computes bond count (number of bonds associated with a given atom index).

    Args:
        universe (:class:`~exatomic.universe.Universe`): Atomic universe

    Returns:
        counts (:class:`~numpy.ndarray`): Bond counts

    Note:
        For both periodic and non-periodic universes, counts returned are
        atom indexed. Counts for projected atoms have no meaning/are not
        computed during two body property calculation.
    """
    stack = universe.atom_two.ix[universe.atom_two['bond'] == True, ['atom0', 'atom1']].stack()
    return stack.value_counts().sort_index()


def compute_molecule_two(universe):
    """
    """
    raise NotImplementedError()





#def bond_summary_by_label_pairs(universe, *labels, length='A', stdev=False,
#                                stderr=False, variance=False, ncount=False):
#    """
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
#    """
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
#    """
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
#    """
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
