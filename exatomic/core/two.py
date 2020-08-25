# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
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
import pandas as pd
from IPython.display import display
from ipywidgets import FloatProgress
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

#    @property
#    def _constructor(self):
#        return AtomTwo

    @property
    def bonded(self):
        return self[self['bond'] == True]


class MoleculeTwo(DataFrame):
    @property
    def _constructor(self):
        return MoleculeTwo


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
    for _, group in universe.atom.groupby("frame"):
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
    #for fdx, group in universe.atom.groupby("frame"):
    for _, group in universe.atom.groupby("frame"):
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
    radmap = {sym: sym2radius[sym][0] for sym in atom['symbol'].cat.categories}
    radmap.update(radii)
    maxdr = (atom_two['atom0'].map(atom['symbol']).map(radmap).astype(float) +
             atom_two['atom1'].map(atom['symbol']).map(radmap).astype(float) + bond_extra)
    atom_two['bond'] = np.where(atom_two['dr'] <= maxdr, True, False)


def _compute_bond_count(atom, atom_two):
    """
    Compute bond counts inplace.
    """
    if "bond" not in atom_two.columns:
        _compute_bonds(atom, atom_two)
    bonded = atom_two.loc[atom_two['bond'] == True, ["atom0", "atom1"]].stack()
    atom['bond_count'] = bonded.value_counts().sort_index()


def compute_atom_two_out_of_core(hdfname, uni, a, **kwargs):
    """
    Perform an out of core periodic two body calculation for a simple cubic
    unit cell with dimension a.

    All data will be saved to and HDF5 file with the given filename. Key
    structure is per frame, i.e. ``frame_fdx/atom_two``.

    Args:
        hdfname (str): HDF file name
        uni (:class:`~exatomic.core.universe.Universe`): Universe
        a (float): Simple cubic unit cell dimension
        kwargs: Keyword arguments for bond computation (i.e. covalent radii)

    See Also:
        :func:`~exatomic.core.two._compute_bonds`
    """
    store = pd.HDFStore(hdfname, mode="a")
    unit_atom = uni.atom[['symbol', 'x', 'y', 'z', 'frame']].copy()
    unit_atom['symbol'] = unit_atom['symbol'].astype(str)
    unit_atom['frame'] = unit_atom['frame'].astype(int)
    unit_atom.update(uni.unit_atom)
    grps = unit_atom.groupby("frame")
    n = len(grps)
    fp = FloatProgress(description="AtomTwo to HDF:")
    display(fp)
    for i, (fdx, atom) in enumerate(grps):
        v = pdist_ortho(atom['x'].values, atom['y'].values,
                        atom['z'].values, a, a, a,
                        atom.index.values, a)
        tdf = pd.DataFrame.from_dict({'frame': np.array([fdx]*len(v[0]), dtype=int),
                                      'dx': v[0], 'dy': v[1], 'dz': v[2], 'dr': v[3],
                                       'atom0': v[4], 'atom1': v[5], 'projection': v[6]})
        _compute_bonds(uni.atom[uni.atom['frame'] == fdx], tdf, **kwargs)
        store.put("frame_"+str(fdx) + "/atom_two", tdf)
        fp.value = i/n*100
    store.close()
    fp.close()


def compute_molecule_two(universe):
    raise NotImplementedError()
