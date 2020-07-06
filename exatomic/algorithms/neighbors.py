# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Neighbor Selection Algorithms
###############################
This module provides algoirthms for selecting nearest neighbors, e.g. nearest
solvent molecules to a solute molecule. Because two body properties do not always
represent the desired molecules (i.e. bonds appear where they are not desired),
these algorithms are not completely black box.

Before performing a search, check that the molecule table is computed as desired
and classified (if necessary): see :func:`~exatomic.two.BaseTwo.compute_bonds`
and :func:`~exatomic.molecule.Molecule.classify`.
"""
import numpy as np
import pandas as pd
import numba as nb
from collections import defaultdict
from IPython.display import display
from ipywidgets import FloatProgress
from exatomic.base import nbpll
from exatomic.core.atom import Atom
from exatomic.core.universe import Universe


@nb.jit(nopython=True, parallel=nbpll)
def _worker(idx, x, y, z, a):
    """
    Generate a 3x3x3 'super' cell from a cubic unit cell.

    Args:
        idx (array): Array of index values
        x (array): Array of x coordinates
        y (array): Array of y coordinates
        z (array): Array of z coordinates
        a (float): Cubic unit cell dimension
    """
    n = len(x)
    idxs = np.empty((27*n, ), dtype=np.int64)
    prj = idxs.copy()
    px = np.empty((27*n, ), dtype=np.float64)
    py = px.copy()
    pz = px.copy()
    p = 0
    m = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                for l in range(n):
                    idxs[m] = idx[l]
                    px[m] = x[l] + i*a
                    py[m] = y[l] + j*a
                    pz[m] = z[l] + k*a
                    prj[m] = p
                    m += 1
                p += 1
    return idxs, px, py, pz, prj


def _create_super_universe(u, a):
    """
    Generate a 3x3x3 super cell from a cubic periodic universe

    Args:
        u (:class:`~exatomic.core.universe.Universe`): Universe
        a (float): Cubic unit cell dimension

    Returns:
        uni (:class:`~exatomic.core.universe.Universe`): Universe of 3x3x3x super cell
    """
    adxs = []
    xs = []
    ys = []
    zs = []
    prjs = []
    fdxs = []
    grps = u.atom.groupby("frame")
    for fdx, atom in grps:
        adx, x, y, z, prj = _worker(atom.index.values.astype(np.int64),
                                    atom['x'].values.astype(np.float64),
                                    atom['y'].values.astype(np.float64),
                                    atom['z'].values.astype(np.float64), a)
        adxs.append(adx)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        prjs.append(prj)
        fdxs += [fdx]*len(adx)
    adxs = np.concatenate(adxs)
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    zs = np.concatenate(zs)
    prjs = np.concatenate(prjs)
    # Overwrite the last 'atom' group because that value doesn't need to exist anymore
    atom = pd.DataFrame.from_dict({'atom': adxs, 'x': xs, 'y': ys, 'z': zs, 'prj': prjs})
    atom['frame'] = fdxs
    atom['symbol'] = atom['atom'].map(u.atom['symbol'])
    atom['label'] = atom['atom'].map(u.atom['label'])
    atom = Atom(atom)
    return Universe(atom=atom)


def periodic_nearest_neighbors_by_atom(uni, source, a, sizes, **kwargs):
    """
    Determine nearest neighbor molecules to a given source (or sources) and
    return the data as a dataframe.

    Warning:
        For universes with more than about 250 atoms, consider using the
        slower but more memory efficient
        :func:`~exatomic.algorithms.neighbors.periodic_nearest_neighbors_by_atom_large`.

    For a simple cubic periodic system with unit cell dimension ``a``,
    clusters can be generated as follows. In the example below, additional
    keyword arguments have been included as they are almost always required
    in order to correctly identify molecular units semi-empirically.

    .. code-block:: python

        periodic_nearest_neighbors_by_atom(u, [0], 40.0, [0, 5, 10, 50],
                                           dmax=40.0, C=1.6, O=1.6)

    Argument descriptions can be found below. The additional keyword arguments,
    ``dmax``, ``C``, ``O``, are passed directly to the two body computation used
    to determine (semi-empirically) molecular units. Note that although molecules
    are computed, neighboring molecular units are determine by an atom to atom
    criteria.

    Args:
        uni (:class:`~exatomic.core.universe.Universe`): Universe
        source (int, str, list): Integer label or string symbol of source atom
        a (float): Cubic unit cell dimension
        sizes (list): List of slices to create
        kwargs: Additional keyword arguments to be passed to atom two body calculation

    Returns:
        dct (dict): Dictionary of sliced universes and nearest neighbor table

    See Also:
        Sliced universe construction can be facilitated by
        :func:`~exatomic.algorithms.neighbors.construct`.
    """
    def sorter(group, source_atom_idxs):
        s = group[['atom0', 'atom1']].stack()
        return s[~s.isin(source_atom_idxs)].reset_index()

    if "label" not in uni.atom.columns:
        uni.atom['label'] = uni.atom.get_atom_labels()
    dct = defaultdict(list)
    grps = uni.atom.groupby("frame")
    ntot = len(grps)
    fp = FloatProgress(description="Slicing:")
    display(fp)
    for i, (fdx, atom) in enumerate(grps):
        if len(atom) > 0:
            uu = _create_super_universe(Universe(atom=atom.copy()), a)
            uu.compute_atom_two(**kwargs)
            uu.compute_molecule()
            if isinstance(source, (int, np.int32, np.int64)):
                source_atom_idxs = uu.atom[(uu.atom.index.isin([source])) &
                                           (uu.atom['prj'] == 13)].index.values
            elif isinstance(source, (list, tuple)):
                source_atom_idxs = uu.atom[uu.atom['label'].isin(source) &
                                           (uu.atom['prj'] == 13)].index.values
            else:
                source_atom_idxs = uu.atom[(uu.atom['symbol'] == source) &
                                           (uu.atom['prj'] == 13)].index.values
            source_molecule_idxs = uu.atom.loc[source_atom_idxs, 'molecule'].unique().astype(int)
            uu.atom_two['frame'] = uu.atom_two['atom0'].map(uu.atom['frame'])
            nearest_atoms = uu.atom_two[(uu.atom_two['atom0'].isin(source_atom_idxs)) |
                                        (uu.atom_two['atom1'].isin(source_atom_idxs))].sort_values("dr")[['frame', 'atom0', 'atom1']]
            nearest = nearest_atoms.groupby("frame").apply(sorter, source_atom_idxs=source_atom_idxs)
            del nearest['level_1']
            nearest.index.names = ['frame', 'idx']
            nearest.columns = ['two', 'atom']
            nearest['molecule'] = nearest['atom'].map(uu.atom['molecule'])
            nearest = nearest[~nearest['molecule'].isin(source_molecule_idxs)]
            nearest = nearest.drop_duplicates('molecule', keep='first')
            nearest.reset_index(inplace=True)
            nearest['frame'] = nearest['frame'].astype(int)
            nearest['molecule'] = nearest['molecule'].astype(int)
            dct['nearest'].append(nearest)
            for nn in sizes:
                atm = []
                for j, fdx in enumerate(nearest['frame'].unique()):
                    mdxs = nearest.loc[nearest['frame'] == fdx, 'molecule'].tolist()[:nn]
                    mdxs.append(source_molecule_idxs[j])
                    atm.append(uu.atom[uu.atom['molecule'].isin(mdxs)][['symbol', 'x', 'y', 'z', 'frame']].copy())
                dct[nn].append(pd.concat(atm, ignore_index=True))
        fp.value = i/ntot*100
    dct['nearest'] = pd.concat(dct['nearest'], ignore_index=True)
    for nn in sizes:
        dct[nn] = Universe(atom=pd.concat(dct[nn], ignore_index=True))
    fp.close()
    return dct


def periodic_nearest_neighbors_by_atom_large(uni, source, a, sizes, **kwargs):
    """
    Determine nearest neighbor molecules to a given source (or sources) and
    return the data as a dataframe.

    Tip:
        This function performs the same operation as
        :func:`~exatomic.algorithms.neighbors.periodic_nearest_neighbors_by_atom`,
        but is meant for universes containing more than about 250 atoms
        per frame (the referenced function will be faster for smaller universes).

    For a simple cubic periodic system with unit cell dimension ``a``,
    clusters can be generated as follows. In the example below, additional
    keyword arguments have been included as they are almost always required
    in order to correctly identify molecular units semi-empirically.

    .. code-block:: python

        periodic_nearest_neighbors_by_atom_ooc(u, [0], 40.0, [0, 5, 10, 50],
                                           dmax=40.0, C=1.6, O=1.6)

    Argument descriptions can be found below. The additional keyword arguments,
    ``dmax``, ``C``, ``O``, are passed directly to the two body computation used
    to determine (semi-empirically) molecular units. Note that although molecules
    are computed, neighboring molecular units are determine by an atom to atom
    criteria.

    Args:
        uni (:class:`~exatomic.core.universe.Universe`): Universe
        source (int, str, list): Integer label or string symbol of source atom
        a (float): Cubic unit cell dimension
        sizes (iterable): List of slices to create
        kwargs: Additional keyword arguments to be passed to atom two body calculation

    Returns:
        dct (dict): Dictionary of sliced universes and nearest neighbor table

    See Also:
        Sliced universe construction can be facilitated by
        :func:`~exatomic.algorithms.neighbors.construct`.
    """
    if "label" not in uni.atom.columns:
        uni.atom['label'] = uni.atom.get_atom_labels()
    if not isinstance(sizes, list):
        raise TypeError("Argument sizes must be iterable of ints.")
    dct = defaultdict(list)
    grps = uni.atom.groupby("frame")
    ntot = len(grps)
    fp = FloatProgress(description="Slicing:")
    display(fp)
    for i, (fdx, atom) in enumerate(grps):
        if len(atom) > 0:
            uu = Universe(atom=atom.copy())
            uu.frame = uni.frame.loc[[fdx], :]
            uu.compute_atom_two(**kwargs)
            uu.compute_molecule()
            if isinstance(source, (int, np.int32, np.int64)):
                source_atom_idxs = [source]
            elif isinstance(source, np.ndarray):
                source_atom_idxs = source.tolist()
            elif isinstance(source, (list, tuple)):
                source_atom_idxs = uu.atom[uu.atom['label'].isin(source) |
                                           uu.atom['symbol'].isin(source)].index.astype(int).tolist()
            else:
                source_atom_idxs = uu.atom[uu.atom['symbol'] == source].index.astype(int).tolist()
            source_molecule_idxs = uu.atom.loc[source_atom_idxs, 'molecule'].unique().astype(int).tolist()
            # Identify the nearest molecules
            nearest_atoms = uu.atom_two[uu.atom_two['atom0'].isin(source_atom_idxs) |
                                        uu.atom_two['atom1'].isin(source_atom_idxs)].sort_values('dr')[['atom0', 'atom1']].copy()
            nearest_atoms['molecule0'] = nearest_atoms['atom0'].map(uu.atom['molecule'])
            nearest_atoms['molecule1'] = nearest_atoms['atom1'].map(uu.atom['molecule'])
            nearest_molecules = nearest_atoms[['molecule0', 'molecule1']].stack()
            nearest_molecules = nearest_molecules[~nearest_molecules.isin(source_molecule_idxs)].drop_duplicates(keep='first')
            # Build the appropriate universes
            for nn in sizes:
                atom1 = uu.atom.loc[uu.atom['molecule'].isin(nearest_molecules.iloc[:nn].tolist()+source_molecule_idxs),
                                   ['symbol', 'x', 'y', 'z']]
                adxs, x, y, z, prj = _worker(atom1.index.values.astype(int),
                                             atom1['x'].values.astype(float),
                                             atom1['y'].values.astype(float),
                                             atom1['z'].values.astype(float), a)
                patom = pd.DataFrame.from_dict({'atom': adxs, 'x': x, 'y': y, 'z': z, 'prj': prj})
                patom['frame'] = patom['atom'].map(uu.atom['frame'])
                patom['symbol'] = patom['atom'].map(uu.atom['symbol'])
                sliced_u = Universe(atom=patom)
                sliced_u.compute_atom_two(dmax=a)
                sliced_u.compute_molecule()
                source_adxs1 = sliced_u.atom[(sliced_u.atom['prj'] == 13) & sliced_u.atom['atom'].isin(source_atom_idxs)].index
                source_mdxs1 = sliced_u.atom.loc[source_adxs1, 'molecule'].unique().tolist()
                nearest_atoms1 = sliced_u.atom_two[sliced_u.atom_two['atom0'].isin(source_adxs1) |
                                                   sliced_u.atom_two['atom1'].isin(source_adxs1)].sort_values('dr')[['atom0', 'atom1']].copy()
                nearest_atoms1['molecule0'] = nearest_atoms1['atom0'].map(sliced_u.atom['molecule'])
                nearest_atoms1['molecule1'] = nearest_atoms1['atom1'].map(sliced_u.atom['molecule'])
                nearest_molecules1 = nearest_atoms1[['molecule0', 'molecule1']].stack()
                nearest_molecules1 = nearest_molecules1[~nearest_molecules1.isin(source_mdxs1)].drop_duplicates(keep='first')
                # Its fine to overwrite atom1 above since the uu.atom slice is not necessarily clustered
                atom1 = sliced_u.atom.loc[sliced_u.atom['molecule'].isin(nearest_molecules1.iloc[:nn].tolist()+source_mdxs1)].copy()
                dct[nn].append(atom1)
            index = nearest_molecules.index.get_level_values(0)
            nearest_molecules = nearest_molecules.to_frame()
            nearest_molecules.columns = ['molecule']
            nearest_molecules['frame'] = fdx
            nearest_molecules['atom0'] = nearest_atoms.loc[index, 'atom0'].values
            nearest_molecules['atom1'] = nearest_atoms.loc[index, 'atom1'].values
            dct['nearest'].append(nearest_molecules)
        fp.value = i/ntot*100
    dct['nearest'] = pd.concat(dct['nearest'], ignore_index=True)
    for nn in sizes:
        dct[nn] = Universe(atom=pd.concat(dct[nn], ignore_index=True))
    fp.close()
    return dct
