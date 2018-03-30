# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
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
from exatomic.base import nbtgt, nbpll
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
    for i, (fdx, atom) in enumerate(grps):
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

    Args:
        uni (:class:`~exatomic.core.universe.Universe`): Universe
        source (int, str): Integer label or string symbol of source atom
        a (float): Cubic unit cell dimension
        sizes (list): List of slices to create
        kwargs: Additional keyword arguments to be passed to atom two body computation (such as covalent radii)

    Returns:
        dct (dict): Dictionary of sliced universes and nearest neighbor table

    Note:
        This function expects cubic periodic unit cells.

    See Also:
        Sliced universe construction can be facilitated by
        :func:`~exatomic.algorithms.neighbors.construct`.
    """
    def sorter(group, source_atom_idxs):
        s = group[['atom0', 'atom1']].stack()
        return s[~s.isin(source_atom_idxs)].reset_index()

    dct = defaultdict(list)
    grps = uni.atom.groupby("frame")
    n = len(grps)
    fp = FloatProgress(description="Slicing:")
    display(fp)
    for i, (fdx, atom) in enumerate(grps):
        uu = _create_super_universe(Universe(atom=atom), a)
        uu.compute_atom_two(**kwargs)
        uu.compute_molecule()
        if isinstance(source, (int, np.int32, np.int64)):
            source_atom_idxs = uu.atom[(uu.atom['label'] == source) &
                                       (uu.atom['prj'] == 13)].index.values
        else:
            source_atom_idxs = uu.atom[(uu.atom['symbol'] == source) &
                                       (uu.atom['prj'] == 13)].index.values
        source_molecule_idxs = uu.atom.loc[source_atom_idxs, 'molecule'].unique()
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
            for i, fdx in enumerate(nearest['frame'].unique()):
                mdxs = nearest.loc[nearest['frame'] == fdx, 'molecule'].tolist()[:nn]
                mdxs.append(source_molecule_idxs[i])
                atm.append(uu.atom[uu.atom['molecule'].isin(mdxs)][['symbol', 'x', 'y', 'z', 'frame']].copy())
            dct[nn].append(pd.concat(atm, ignore_index), ignore_index=True)
        fp.value = i/n*100
    fp.close()
    dct['nearest'] = pd.concat(dct['nearest'], ignore_index=True)
    for nn in sizes:
        dct[nn] = Universe(atom=pd.concat(dct[nn], ignore_index=True))
    return dct


#def nearest_molecules(universe, n, sources, restrictions=None, how='atom',
#                      free_boundary=True, center=(0, 0, 0)):
#    """
#    Select nearest molecules to a source or sources.
#
#    .. code-block:: Python
#
#        source = 'analyte'    # By molecule classification  (Molecule)
#        source = 1            # By atom label (Atom)
#        source = 'C'          # By atom symbol (Atom)
#        nearest_molecules(uni, 5, source)
#
#    .. code-block:: Python
#
#        sources = ['solute', 'C']   # Can mix and match..
#        nearest_molecules(uni, 5, sources)    # Nearest neighbors to 'C' atoms on 'solute' molecules
#
#    Args:
#        universe (:class:`~exatomic.container.Universe`): An atomic universe
#        n (int or list): Number(s) of neighbors to select to each source (see note)
#        sources: Source molecules/atoms from which to search for neighbors
#        restrictions: Restrict neighbors (non-source molecules/atoms) by atom symbol or molecule classification
#        how (str): Search by atom to atom distance ('atom') or molecule center of mass ('com')
#        free_boundary (bool): Convert to free boundary conditions (if periodic system - default true)
#        center (array): Center the result on the given point (default (0, 0, 0))
#
#    Returns:
#        unis (dict): Dictionary of number of neighbors keys, universe values
#    """
#    #source_atoms, other_atoms, source_molecules, other_molecules, n = _slice_atoms_molecules(universe, sources, restrictions, n)
#    source_atoms, other_atoms, source_molecules, _, n = _slice_atoms_molecules(universe, sources, restrictions, n)
#    ordered_molecules, ordered_twos = _compute_neighbors_by_atom(universe, source_atoms, other_atoms, source_molecules)
#    unis = {}
#    if free_boundary == True:
#        for nn in n:
#            unis[nn] = _build_free_universe(universe, ordered_molecules,
#                                            ordered_twos, nn, source_atoms,
#                                            source_molecules)
#    else:
#        raise NotImplementedError()
#    return unis
#
#
#def _slice_atoms_molecules(universe, sources, restrictions, n):
#    """
#    Initial check of the unvierse data and argument types and creation of atom
#    and molecule table slices.
#    """
#    if not isinstance(sources, list):
#        sources = [sources]
#    if not isinstance(restrictions, list) and restrictions is not None:
#        restrictions = [restrictions]
#    if isinstance(n, (int, np.int32, np.int64)):
#        n = [n]
#    labels = universe.atom.get_atom_labels()
#    del_label = False
#    if 'label' not in universe.atom.columns:
#        del_label = True
#        universe.atom['label'] = labels
#    labels = labels.unique()
#    symbols = universe.atom['symbol'].unique()
#    classification = []
#    if 'classification' in universe.molecule.columns:
#        classification = universe.molecule['classification'].unique()
#    if all(source in labels for source in sources):
#        source_atoms = universe.atom[universe.atom['label'].isin(sources)]
#        mdx = source_atoms['molecule'].astype(np.int64)
#        source_molecules = universe.molecule[universe.molecule.index.isin(mdx)]
#    elif all(source in symbols for source in sources):
#        source_atoms = universe.atom[universe.atom['symbol'].isin(sources)]
#        mdx = source_atoms['molecule'].astype(np.int64)
#        source_molecules = universe.molecule[universe.molecule.index.isin(mdx)]
#    elif all(source in classification for source in sources):
#        source_molecules = universe.molecule[universe.molecule['classification'].isin(sources)]
#        source_atoms = universe.atom[universe.atom['molecule'].isin(source_molecules.index)]
#    else:
#        classif = [source for source in sources if source in classification]
#        syms = [source for source in sources if source in symbols]
#        lbls = [source for source in sources if source in labels]
#        source_molecules = universe.molecule[universe.molecule['classification'].isin(classif)]
#        source_atoms = universe.atom[universe.atom['molecule'].isin(source_molecules.index)]
#        if len(syms) > 0:
#            source_atoms = source_atoms[source_atoms['symbol'].isin(syms)]
#        if len(lbls) > 0:
#            source_atoms = source_atoms[source_atoms['label'].isin(lbls)]
#    other_molecules = universe.molecule[~universe.molecule.index.isin(source_molecules.index)]
#    other_atoms = universe.atom[~universe.atom.index.isin(source_atoms.index)]
#    if restrictions is not None:
#        if all(other in labels for other in restrictions):
#            other_atoms = other_atoms[other_atoms['label'].isin(restrictions)]
#            mdx = other_atoms['molecule'].astype(np.int64)
#            other_molecules = other_molecules[other_molecules.index.isin(mdx)]
#        elif all(other in symbols for other in restrictions):
#            other_atoms = other_atoms[other_atoms['symbol'].isin(restrictions)]
#            mdx = other_atoms['molecule'].astype(np.int64)
#            other_molecules = other_molecules[other_molecules.index.isin(mdx)]
#        elif all(other in classification for other in restrictions):
#            other_molecules = other_molecules[other_molecules['classification'].isin(restrictions)]
#            other_atoms = other_atom[other_atoms['molecule'].isin(other_molecules.index)]
#        else:
#            classif = [other for other in restrictions if other in classification]
#            syms = [other for other in restrictions if other in symbols]
#            lbls = [other for other in restrictions if other in labels]
#            other_molecules = other_molecules[other_molecules['classification'].isin(classif)]
#            other_atoms = other_atoms[other_atoms['molecule'].isin(other_molecules.index)]
#            if len(syms) > 0:
#                other_atoms = other_atoms[other_atoms['symbol'].isin(syms)]
#            if len(lbls) > 0:
#                other_atoms = other_atoms[other_atoms['label'].isin(lbls)]
#    if del_label:
#        del universe.atom['label']
#    return source_atoms, other_atoms, source_molecules, other_molecules, n
#
#
#def _compute_neighbors_by_atom(universe, source_atoms, other_atoms, source_molecules):
#    """
#    """
#    universe.atom_two._revert_categories()
#    two = universe.atom_two.loc[(universe.atom_two['atom0'].isin(source_atoms.index.values) &
#                                 universe.atom_two['atom1'].isin(other_atoms.index.values)) |
#                                (universe.atom_two['atom1'].isin(source_atoms.index.values) &
#                                 universe.atom_two['atom0'].isin(other_atoms.index.values)),
#                                ['atom0', 'atom1', 'dr', 'frame']].sort_values('dr')
#    mapper = universe.atom['molecule'].astype(np.int64)
#    groups = two['atom0'].map(mapper).to_frame()
#    groups.columns = ['molecule0']
#    groups['molecule1'] = two['atom1'].map(mapper)
#    groups['frame'] = two['frame']
#    universe.atom_two._set_categories()
#    groups = groups.groupby('frame')
#    n = groups.ngroups
#    ordered_molecules = np.empty((n, ), dtype=np.ndarray)
#    ordered_twos = np.empty((n, ), dtype=np.ndarray)
#    for i, (frame, group) in enumerate(groups):
#        series = group[['molecule0', 'molecule1']].stack().drop_duplicates().reset_index(level=1, drop=True)
#        series = series[~series.isin(source_molecules.index)]
#        ordered_molecules[i] = series.values
#        ordered_twos[i] = series.index.values
#    return ordered_molecules, ordered_twos
#
#
#def _compute_neighbors_by_com(universe, source_molecules, other_molecules):
#    """
#    """
#    raise NotImplementedError()
#
#
#def _build_free_universe(universe, ordered_molecules, ordered_twos, n,
#                         source_atoms, source_molecules):
#    """
#    """
#    molecule = np.concatenate([mcules[:n] for mcules in ordered_molecules])
#    molecule = np.concatenate((molecule, source_molecules.index.tolist()))
#    molecule = universe.molecule[universe.molecule.index.isin(molecule)].copy()
#    atom = universe.atom[universe.atom['molecule'].isin(molecule.index)].copy()
#    atom_two = universe.atom_two[(universe.atom_two['atom0'].isin(atom.index) &
#                                  universe.atom_two['atom1'].isin(atom.index))].copy()
#    frame = universe.frame[universe.frame.index.isin(atom['frame'])].copy()
#    frame['periodic'] = False
#    uni = universe.__class__(atom=atom, molecule=molecule, frame=frame, atom_two=atom_two)
##    if universe.frame.is_periodic():
##        uni.atom.update(universe.visual_atom)
##        if 'cx' not in uni.molecule.columns:
##            uni.compute_molecule_com()
##        uni.atom._revert_categories()
##        mapper = uni.atom.drop_duplicates('molecule').set_index('molecule')['frame']
##        uni.atom._set_categories()
##        uni.molecule['frame'] = uni.molecule.index.map(lambda x: mapper[x])
##        sources = source_atoms.groupby('frame')
##        groups = uni.molecule.groupby('frame')
##        n = groups.ngroups
##        dx = np.empty((n, ), dtype=np.ndarray)
##        dy = np.empty((n, ), dtype=np.ndarray)
##        dz = np.empty((n, ), dtype=np.ndarray)
##        index = np.empty((n, ), dtype=np.ndarray)
##        for i, (frame, group) in enumerate(groups):
##            cx = group['cx'].values
##            cy = group['cy'].values
##            cz = group['cz'].values
##            ccx, ccy, ccz = sources.get_group(frame)[['x', 'y', 'z']].mean().values
##    #        ccx, ccy, ccz = mcules.ix[mcules['classification'] == 'solute', ['cx', 'cy', 'cz']].values[0]
##            rx, ry, rz = uni.frame.ix[frame, ['rx', 'ry', 'rz']].values
##            dxf, dyf, dzf = _compute(cx, cy, cz, rx, ry, rz, ccx, ccy, ccz)
##            dx[i] = dxf
##            dy[i] = dyf
##            dz[i] = dzf
##            index[i] = group.index.values
##        del uni.molecule['frame']
##        dx = np.concatenate(dx)
##        dy = np.concatenate(dy)
##        dz = np.concatenate(dz)
##        index = np.concatenate(index)
##        df = pd.DataFrame.from_dict({'x': dx, 'y': dy, 'z': dz, 'molecule': index})
##        df.set_index('molecule', inplace=True)
##        for molecule in df.index:
##            dx, dy, dz = df.ix[molecule].values
##            uni.atom.ix[uni.atom['molecule'] == molecule, 'x'] += dx
##            uni.atom.ix[uni.atom['molecule'] == molecule, 'y'] += dy
##            uni.atom.ix[uni.atom['molecule'] == molecule, 'z'] += dz
#    return uni
#
#
#def _build_universe(universe, ordered_molecules, ordered_twos, n):
#    """
#    """
#    raise NotImplementedError()
#    # TODO CONVERT TO A GENERIC AND COMPLETE SLICER
##    molecules = np.concatenate([m[:n] for m in ordered_molecules])
##    twos = np.concatenate([t[:n] for t in ordered_twos])
##    atom = universe.atom[universe.atom['molecule'].isin(molecules)].copy().sort_index()
##    two = universe.atom_two[universe.atom_two['atom0'].isin(atom.index) &
##                            universe.atom_two['atom1'].isin(atom.index)].copy().sort_index()
##    projected_atom = universe.projected_atom.ix[two.index.values].copy().sort_index()
##    visual_atom = universe.visual_atom.ix[atom.index].copy().sort_index()
##    molecule = universe.molecule.ix[molecules].copy().sort_index()
##    frame = universe.frame.copy().sort_index()
##    uni = Universe(atom=atom, atom_two=two, molecule=molecule, frame=frame,
##                   visual_atom=visual_atom, projected_atom=projected_atom)
##    return uni
