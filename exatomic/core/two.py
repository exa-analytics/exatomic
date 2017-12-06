# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Atomic Two Body
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
from exa import DataFrame
#from exa.util.units import Length
from exatomic.base import sym2radius
from exatomic.algorithms.distance import (pdist_ortho, pdist_ortho_nv, pdist,
                                          pdist_nv)


class AtomTwo(DataFrame):
    """Interatomic distances."""
    _index = "two"
    #_cardinal = ("frame", np.int64)
    _columns = ["atom0", "atom1", "dr"]
    _categories = {'symbols': str, 'atom0': np.int64, 'atom1': np.int64}

    @property
    def bonded(self):
        return self[self['bond'] == True]


class MoleculeTwo(DataFrame):
    pass


def compute_atom_two(universe, dmax=8.0, vector=False, bonds=True, **kwargs):
    """
    Compute interatomic distances and determine bonds.

    .. code-block:: python

        atom_two = compute_atom_two(uni, dmax=4.0)    # Max distance of interest as 4 bohr
        atom_two = compute_atom_two(uni, vector=True) # Return distance vector components as well as distance
        atom_two = compute_atom_two(uni, bonds=False) # Don't compute bonds
        # Compute bonds with custom covalent radii (atomic units)
        atom_two = compute_atom_two(unit, H=10.0, He=20.0, Li=30.0, bond_extra=100.0)

    Args:
        universe (:class:`~exatomic.core.universe.Universe`): Universe object with atom table
        dmax (float): Maximum distance of interest
        vector (bool): Compute distance vector (needed for angles)
        bonds (bool): Compute bonds (default True)
        kwargs: Additional keyword arguments for :func:`~exatomic.core.two._compute_bonds`
    """
    if universe.periodic:
        if universe.orthorhombic and vector:
            atom_two = compute_pdist_ortho(universe, dmax=dmax)
        elif universe.orthorhombic:
            atom_two = compute_pdist_ortho_nv(universe, dmax=dmax)
        else:
            raise NotImplementedError("Only supports orthorhombic cells")
    elif vector:
        atom_two = compute_pdist(universe, dmax=dmax)
    else:
        atom_two = compute_pdist_nv(universe, dmax=dmax)
    if bonds:
        _compute_bonds(universe.atom, atom_two, **kwargs)
    return atom_two


def compute_pdist(universe, dmax=8.0):
    """
    Compute interatomic distances for atoms in free boundary conditions.

    Does return distance vector.
    """
    dxs = []
    dys = []
    dzs = []
    drs = []
    atom0s = []
    atom1s = []
    for fdx, group in universe.atom.groupby("frame"):
        if len(group) > 0:
            values = pdist(group['x'].values.astype(float),
                           group['y'].values.astype(float),
                           group['z'].values.astype(float),
                           group.index.values.astype(int), dmax)
            dxs.append(values[0])
            dys.append(values[1])
            dzs.append(values[2])
            drs.append(values[3])
            atom0s.append(values[4])
            atom1s.append(values[5])
    dxs = np.concatenate(dxs)
    dys = np.concatenate(dys)
    dzs = np.concatenate(dzs)
    drs = np.concatenate(drs)
    atom0s = np.concatenate(atom0s)
    atom1s = np.concatenate(atom1s)
    return AtomTwo.from_dict({'dx': dxs, 'dy': dys, 'dz': dzs, 'dr': drs,
                              'atom0': atom0s, 'atom1': atom1s})


def compute_pdist_nv(universe, dmax=8.0):
    """
    Compute interatomic distances for atoms in free boundary conditions.

    Does not return distance vector.
    """
    drs = []
    atom0s = []
    atom1s = []
    for fdx, group in universe.atom.groupby("frame"):
        if len(group) > 0:
            values = pdist_nv(group['x'].values.astype(float),
                              group['y'].values.astype(float),
                              group['z'].values.astype(float),
                              group.index.values.astype(int), dmax)
            drs.append(values[0])
            atom0s.append(values[1])
            atom1s.append(values[2])
    drs = np.concatenate(drs)
    atom0s = np.concatenate(atom0s)
    atom1s = np.concatenate(atom1s)
    return AtomTwo.from_dict({'dr': drs, 'atom0': atom0s, 'atom1': atom1s})


def compute_pdist_ortho(universe, dmax=8.0):
    """
    Compute interatomic distances between atoms in an orthorhombic
    periodic cell.

    Args:
        universe (:class:`~exatomic.core.universe.Universe`): A universe
        bonds (bool): Compute bonds as well as distances
        bond_extra (float): Extra factor to use when determining bonds
        dmax (float): Maximum distance of interest
        rtol (float): Relative tolerance (float equivalence)
        atol (float): Absolute tolerance (float equivalence)
        radii (kwargs): Custom (covalent) radii to use when determining bonds
    """
    if "rx" not in universe.frame.columns:
        universe.frame.compute_cell_magnitudes()
    dxs = []
    dys = []
    dzs = []
    drs = []
    atom0s = []
    atom1s = []
    prjs = []
    atom = universe.atom[["x", "y", "z", "frame"]].copy()
    atom.update(universe.unit_atom)
    for fdx, group in atom.groupby("frame"):
        if len(group) > 0:
            a, b, c = universe.frame.loc[fdx, ["rx", "ry", "rz"]]
            values = pdist_ortho(group['x'].values.astype(float),
                                 group['y'].values.astype(float),
                                 group['z'].values.astype(float),
                                 a, b, c,
                                 group.index.values.astype(int), dmax)
            dxs.append(values[0])
            dys.append(values[1])
            dzs.append(values[2])
            drs.append(values[3])
            atom0s.append(values[4])
            atom1s.append(values[5])
            prjs.append(values[6])
    dxs = np.concatenate(dxs)
    dys = np.concatenate(dys)
    dzs = np.concatenate(dzs)
    drs = np.concatenate(drs)
    atom0s = np.concatenate(atom0s)
    atom1s = np.concatenate(atom1s)
    prjs = np.concatenate(prjs)
    return AtomTwo.from_dict({'dx': dxs, 'dy': dys, 'dz': dzs, 'dr': drs,
                              'atom0': atom0s, 'atom1': atom1s, 'projection': prjs})


def compute_pdist_ortho_nv(universe, dmax=8.0):
    """
    Compute interatomic distances between atoms in an orthorhombic
    periodic cell.

    Args:
        universe (:class:`~exatomic.core.universe.Universe`): A universe
        bonds (bool): Compute bonds as well as distances
        bond_extra (float): Extra factor to use when determining bonds
        dmax (float): Maximum distance of interest
        rtol (float): Relative tolerance (float equivalence)
        atol (float): Absolute tolerance (float equivalence)
        radii (kwargs): Custom (covalent) radii to use when determining bonds
    """
    if "rx" not in universe.frame.columns:
        universe.frame.compute_cell_magnitudes()
    drs = []
    atom0s = []
    atom1s = []
    prjs = []
    atom = universe.atom[["x", "y", "z", "frame"]].copy()
    atom.update(universe.unit_atom)
    for fdx, group in atom.groupby("frame"):
        if len(group) > 0:
            a, b, c = universe.frame.loc[fdx, ["rx", "ry", "rz"]]
            values = pdist_ortho_nv(group['x'].values.astype(float),
                                    group['y'].values.astype(float),
                                    group['z'].values.astype(float),
                                    a, b, c,
                                    group.index.values.astype(int), dmax)
            drs.append(values[0])
            atom0s.append(values[1])
            atom1s.append(values[2])
            prjs.append(values[3])
    drs = np.concatenate(drs)
    atom0s = np.concatenate(atom0s)
    atom1s = np.concatenate(atom1s)
    prjs = np.concatenate(prjs)
    return AtomTwo.from_dict({'dr': drs, 'atom0': atom0s, 'atom1': atom1s,
                              'projection': prjs})


def _compute_bonds(atom, atom_two, bond_extra=0.45, **radii):
    """
    Compute bonds inplce.

    Args:
        bond_extra (float): Additional amount for determining bonds
        radii: Custom radii to use for computing bonds
    """
    atom['symbol'] = atom['symbol'].astype('category')
    radmap = {sym: sym2radius[sym] for sym in atom['symbol'].cat.categories}
    radmap.update(radii)
    maxdr = (atom_two['atom0'].map(atom['symbol']).map(radmap) +
             atom_two['atom1'].map(atom['symbol']).map(radmap) + bond_extra)
    atom_two['bond'] = False
    atom_two.loc[atom_two['dr'] <= maxdr, 'bond'] = True


def _compute_bond_count(atom, atom_two):
    """
    Compute bond counts inplace.
    """
    if "bond" not in atom_two.columns:
        _compute_bonds(atom, atom_two)
    bonded = atom_two.loc[atom_two['bond'] == True, ["atom0", "atom1"]].stack()
    atom['bond_count'] = bonded.value_counts().sort_index()

#def compute_free_two_si(universe, mapper=None, bond_extra=0.45):
#    """
#    Serial, in memory computation of two body properties for free boundary
#    condition systems.
#    """
#    n = universe.frame['atom_count'].astype(np.int64)
#    n = (n*(n - 1)//2).sum()
#    dx = np.empty((n, ), dtype=np.float64)
#    dy = np.empty((n, ), dtype=np.float64)
#    dz = np.empty((n, ), dtype=np.float64)
#    distance = np.empty((n, ), dtype=np.float64)
#    atom0 = np.empty((n, ), dtype=np.int64)
#    atom1 = np.empty((n, ), dtype=np.int64)
#    fdx = np.empty((n, ), dtype=np.int64)
#    start = 0
#    stop = 0
#    for frame, group in universe.atom.cardinal_groupby():
#        x = group['x'].values.astype(np.float64)
#        y = group['y'].values.astype(np.float64)
#        z = group['z'].values.astype(np.float64)
#        idx = group.index.values.astype(np.int64)
#        dxx, dyy, dzz, dist, a0, a1 = pdist_euc_dxyz_idx(x, y, z, idx)
#        stop += len(dxx)
#        dx[start:stop] = dxx
#        dy[start:stop] = dyy
#        dz[start:stop] = dzz
#        atom0[start:stop] = a0
#        atom1[start:stop] = a1
#        distance[start:stop] = dist
#        fdx[start:stop] = frame
#        start = stop
#    atom0 = pd.Series(atom0, dtype='category')
#    atom1 = pd.Series(atom1, dtype='category')
#    fdx = pd.Series(fdx, dtype='category')
#    two = pd.DataFrame.from_dict({'dx': dx, 'dy': dy, 'dz': dz, 'distance': distance,
#                                  'atom0': atom0, 'atom1': atom1, 'frame': fdx})
#    two = AtomTwo(two)
#    two.compute_bonds(universe.atom['symbol'], mapper=mapper)
#    return two
#
#
#def compute_periodic_two_si(universe, mapper=None, bond_extra=0.45):
#    """
#    Compute periodic two body properties.
#    """
#    grps = universe.atom[['x', 'y', 'z', 'frame']].copy()
#    grps['frame'] = grps['frame'].astype(np.int64)
#    grps.update(universe.unit_atom)
#    grps = grps.groupby('frame')
#    n = grps.ngroups
#    dx = np.empty((n, ), dtype=np.ndarray)
#    dy = np.empty((n, ), dtype=np.ndarray)
#    dz = np.empty((n, ), dtype=np.ndarray)
#    atom0 = np.empty((n, ), dtype=np.ndarray)
#    atom1 = np.empty((n, ), dtype=np.ndarray)
#    distance = np.empty((n, ), dtype=np.ndarray)
#    fdx = np.empty((n, ), dtype=np.ndarray)
#    px = np.empty((n, ), dtype=np.ndarray)
#    py = np.empty((n, ), dtype=np.ndarray)
#    pz = np.empty((n, ), dtype=np.ndarray)
#    start = 0
#    stop = 0
#    for i, (frame, grp) in enumerate(grps):
#        ux = grp['x'].values.astype(np.float64)
#        uy = grp['y'].values.astype(np.float64)
#        uz = grp['z'].values.astype(np.float64)
#        sidx = grp.index.values.astype(np.int64)
#        rx, ry, rz = universe.frame.ix[frame, ['rx', 'ry', 'rz']]
#        dxx, dyy, dzz, d, a0, a1, pxx, pyy, pzz = periodic_pdist_euc_dxyz_idx(ux, uy, uz, rx, ry, rz, sidx)
#        nnn = len(dxx)
#        stop += nnn
#        dx[i] = dxx
#        dy[i] = dyy
#        dz[i] = dzz
#        distance[i] = d
#        atom0[i] = a0
#        atom1[i] = a1
#        px[i] = pxx
#        py[i] = pyy
#        pz[i] = pzz
#        fdx[i] = [frame for j in range(nnn)]
#        start = stop
#    dx = np.concatenate(dx)
#    dy = np.concatenate(dy)
#    dz = np.concatenate(dz)
#    distance = np.concatenate(distance)
#    px = np.concatenate(px)
#    py = np.concatenate(py)
#    pz = np.concatenate(pz)
#    atom0 = pd.Series(np.concatenate(atom0), dtype='category')
#    atom1 = pd.Series(np.concatenate(atom1), dtype='category')
#    fdx = pd.Series(np.concatenate(fdx), dtype='category')
#    two = pd.DataFrame.from_dict({'dx':dx, 'dy': dy, 'dz': dz, 'distance': distance,
#                                  'atom0': atom0, 'atom1': atom1, 'frame': fdx})
#    patom = pd.DataFrame.from_dict({'x': px, 'y': py, 'z': pz})
#    two = AtomTwo(two)
#    two.compute_bonds(universe.atom['symbol'], mapper=mapper)
#    return two, patom
#


def compute_molecule_two(universe):
    raise NotImplementedError()


#def bond_summary_by_label_pairs(universe, *labels, **kwargs):
#   """
#   Compute a summary of bond lengths by label pairs
#
#   Args:
#       universe: The atomic container
#       \*labels: Any number of label pairs (e.g. ...paris(uni, (0, 1), (1, 0), ...))
#       length (str): Output length unit (default Angstrom)
#       stdev (bool): Compute the standard deviation of the mean (default false)
#       stderr (bool): Compute the standard error in the mean (default false)
#       variance (bool): Compute the variance in the mean (default false)
#       ncount (bool): Include the data point count (default false)
#
#   Returns:
#       summary (:class:`~pandas.DataFrame`): Bond length dataframe
#   """
#   length = kwargs.pop("length", "Angstrom")
#   stdev = kwargs.pop("stdev", False)
#   stderr = kwargs.pop("stderr", False)
#   variance = kwargs.pop("variance", False)
#   ncount = kwargs.pop("ncount", False)
#   l0, l1 = list(zip(*labels))
#   l0 = np.array(l0, dtype=np.int64)
#   l1 = np.array(l1, dtype=np.int64)
#   ids = unordered_pairing(l0, l1)
#   bonded = universe.two[universe.two['bond'] == True].copy()
#   if universe.periodic:
#       bonded['atom0'] = bonded['prjd_atom0'].map(universe.projected_atom['atom'])
#       bonded['atom1'] = bonded['prjd_atom1'].map(universe.projected_atom['atom'])
#   bonded['label0'] = bonded['atom0'].map(universe.atom['label'])
#   bonded['label1'] = bonded['atom1'].map(universe.atom['label'])
#   bonded['id'] = unordered_pairing(bonded['label0'].values.astype(np.int64),
#                                    bonded['label1'].values.astype(np.int64))
#   return bonded[bonded['id'].isin(ids)]
#   grps = bonded[bonded['id'].isin(ids)].groupby('id')
#   df = grps['distance'].mean().reset_index()
#   if variance:
#       df['variance'] = grps['distance'].var().reset_index()['distance']
#       df['variance'] *= Length['au', length]
#   if stderr:
#       df['stderr'] = grps['distance'].std().reset_index()['distance']
#       df['stderr'] /= np.sqrt(grps['distance'].size().values[0])
#       df['stderr'] *= Length['au', length]
#   if stdev:
#       df['stdev'] = grps['distance'].std().reset_index()['distance']
#       df['stdev'] *= Length['au', length]
#   if ncount:
#       df['count'] = grps['distance'].size().reset_index()[0]
#   mapper = bonded.drop_duplicates('id').set_index('id')
#   df['symbols'] = df['id'].map(mapper['symbols'])
#   df['distance'] *= Length['au', length]
#   df['label0'] = df['id'].map(mapper['label0'])
#   df['label1'] = df['id'].map(mapper['label1'])
#   del df['id']
#   return df


#def n_nearest_distances_by_symbols(universe, a, b, n, length='Angstrom', stdev=False,
#                                  stderr=False, variance=False, ncount=False):
#   """
#   Compute a distance summary of the n nearest pairs of symbols, (a, b).
#
#   Args:
#       universe: The atomic universe
#       a (str): Symbol string
#       b (str): Symbol string
#       n (int): Number of distances to include
#       stdev (bool): Compute the standard deviation of the mean (default false)
#       stderr (bool): Compute the standard error in the mean (default false)
#       variance (bool): Compute the variance in the mean (default false)
#       ncount (bool): Include the data point count (default false)
#
#   Returns:
#       summary (:class:`~pandas.DataFrame`): Distance summary dataframe
#   """
#   def compute(group):
#       return group.sort_values('distance').iloc[:n]
#
#   df = universe.two[universe.two['symbols'].isin([a+b, b+a])]
#   df = df.groupby('frame').apply(compute)
#   df['pair'] = list(range(n)) * (len(df) // n)
#   pvd = df.pivot('frame', 'pair', 'distance')
#   df = pvd.mean(0).reset_index()
#   df.columns = ['pair', 'distance']
#   df['distance'] *= Length['au', length]
#   if stdev:
#       df['stdev'] = pvd.std().reset_index()[0]
#       df['stdev'] *= Length['au', length]
#   if stderr:
#       df['stderr'] = pvd.std().reset_index()[0]
#       df['stderr'] /= np.sqrt(len(pvd))
#       df['stderr'] *= Length['au', length]
#   if variance:
#       df['variance'] = pvd.var().reset_index()[0]
#       df['variance'] *= Length['au', length]
#   if ncount:
#       df['count'] = pvd.shape[0]
#   return df
