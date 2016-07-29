# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
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
import warnings


def get_nearest_molecules(universe, n, sources, restrictions=None, how='atom'):
    """
    Select nearest molecules to a source or sources.

    .. code-block:: Python

        source = 'analyte'    # By molecule classification  (Molecule)
        source = 1            # By atom label (Atom)
        source = 'C'          # By atom symbol (Atom)
        get_nearest_molecules(uni, 5, source)

    .. code-block:: Python

        sources = ['solute', 'C']   # Can mix and match..
        get_nearest_molecules(uni, 5, sources)    # Nearest neighbors to 'C' atoms on 'solute' molecules

    Args:
        universe (:class:`~exatomic.container.Universe`): An atomic universe
        n (int): Numver of neighbors to select to each source
        sources: See examples above or note below
    """
    # Perform some simple checks
    if 'classification' not in universe.molecule:
        raise KeyErrror("Column 'classification' not in the molecule table, please classify")
    if len(universe.molecule[universe.molecule['classification'].isnull()]) > 0:
        warnings.warn("Unclassified molecules present in the molecule table, unclassificed .")
    if not isinstance(sources, list):
        sources = [sources]
    if not isinstance(restrictions, list) and restrictions is not None:
        restrictions = [restrictions]
    # Determine the type of selection method
    symbols = universe.atom['symbol'].unique()
    classification = universe.molecule['classification'].unique()
    src_all_in_symbols = all(source in symbols for source in sources)
    src_all_in_classif = all(source in classification for source in sources)
    src_any_in_symbols = any(source in symbols for source in sources)
    src_any_in_classif = any(source in classification for source in sources)
    if restrictions is not None:
        rst_all_in_symbols = all(restriction in symbols for restriction in restrictions)
        rst_all_in_classif = all(restriction in classification for restriction in restrictions)
        rst_any_in_symbols = any(restriction in symbols for restriction in restrictions)
        rst_any_in_classif = any(restriction in classification for restriction in restrictions)
    # Based on the selection method get the correct sections of the molecule and atom tables
    if src_all_in_symbols:
        return get_nearest_molecules_source_by_symbols(universe, n, sources)
    elif src_all_in_classif:
        return get_nearest_molecules_source_by_classif(universe, n, sources)
    else:
        raise NotImplementedError()
    # The following loop performs the selection
    mapper = u.atom['molecule']
    groups = u.two.groupby('frame')
other_molecules = np.empty((groups.ngroups, ), dtype=np.ndarray)
twos = np.empty((groups.ngroups, ), dtype=np.ndarray)

h = 0
for frame, group in groups:
    g = group[(group['atom0'].isin(source_atoms.index) & group['atom1'].isin(other_atoms.index)) |
              (group['atom1'].isin(source_atoms.index) & group['atom0'].isin(other_atoms.index))]
    srtd = g.sort_values('distance')
    molecule0 = srtd['atom0'].map(mapper)
    molecule1 = srtd['atom1'].map(mapper)
    nn = len(molecule0)*2
    molecules = np.empty((nn, ), dtype=np.int64)
    index = np.empty((nn, ), dtype=np.int64)
    molecules[0::2] = molecule0.values
    molecules[1::2] = molecule1.values
    index[0::2] = molecule0.index.values
    index[1::2] = molecule1.index.values
    molecules = pd.Series(molecules)
    molecules.index = index
    molecules = molecules.drop_duplicates(keep='first')
    molecules = molecules[~molecules.isin(source_molecules.index)]
    molecules = molecules.iloc[:n]
    other_molecules[h] = molecules.values
    twos[h] = molecules.index.values
    h += 1
other_molecules = np.concatenate(other_molecules)
molecules = np.concatenate((other_molecules, source_molecules.index.values))
twos = np.concatenate(twos)


def _get_nearest_molecules_source_by_classif(universe, sources):
    """
    """
    source_molecules = universe.molecule[universe.molecule['classification'].isin(sources)]
    other_molecules = universe.molecule[~universe.molecule.index.isin(source_molecules.index)]
    source_atoms = u.atom[u.atom['molecule'].isin(source_molecules.index)]
    other_atoms = u.atom[~u.atom.index.isin(source_atoms.index)]
    return sources, n


def _get_nearest_molecules_source_by_symbols(universe, n, sources):
    """
    """
    source_atoms = universe.atom[universe.atom['symbol'].isin(sources)]
    other_atoms = universe.atom[~universe.atom.index.isin(source_atoms)]
    source_molecules = universe.molecule[universe.molecule[sources[0]] > 0]
    if len(sources) > 1:
        for source in sources[1:]:
            source_molecules = source_molecules[source_molecules[source] > 0]
    other_molecules = universe.molecule[~universe.molecule.index.isin(source_molecules.index)]
    return source_atoms, other_atoms, source_molecules, other_molecules

def _index_slicer(universe, source_molecules, source_atoms, other_molecules, other_atoms):
    """
    """
    pass
