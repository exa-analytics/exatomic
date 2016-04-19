# -*- coding: utf-8 -*-
'''
Molecule Data
===============================
'''
import numpy as np
import pandas as pd
from networkx import Graph
from networkx.algorithms.components import connected_components
from itertools import combinations
from exa import DataFrame
from atomic import Isotope


class Molecule(DataFrame):
    '''
    Description of molecules in the atomic universe.
    '''
    _index = ['molecule']
    _groupbys = ['frame']
    _categories = {'frame': np.int64, 'formula': str}


def compute_molecule(universe):
    '''
    Cluster atoms into molecules.

    The algorithm is to create a network graph containing every atom (in every
    frame as nodes and bonds as edges). Using this connectivity information,
    one can perform a (breadth first) traversal of the network graph to cluster
    all nodes (whose indices correspond to physical atoms).

    Args:
        universe (:class:`~atomic.universe.Universe`): Atomic universe

    Returns:
        objs (tuple): Molecule indices (for atom dataframe(s)) and molecule dataframe

    Warning:
        This function will modify (in place) a few tables of the universe!
    '''
    if 'bond_count' not in universe.atom:    # The bond count is used to find single atoms;
        universe.compute_bond_count()        # single atoms are treated as molecules.
    b0 = None
    b1 = None
    bonded = universe.two[universe.two['bond'] == True]
    if universe.is_periodic:
        mapper = universe.projected_atom['atom']
        b0 = bonded['prjd_atom0'].map(mapper)
        b1 = bonded['prjd_atom1'].map(mapper)
    else:
        b0 = bonded['atom0']
        b1 = bonded['atom1']
    graph = Graph()
    graph.add_edges_from(zip(b0.values, b1.values))
    mapper = {}
    for i, molecule in enumerate(connected_components(graph)):
        for atom in molecule:
            mapper[atom] = i
    n = max(mapper.values()) + 1
    idxs = universe.atom[universe.atom['bond_count'] == 0].index
    for i, index in enumerate(idxs):
        mapper[index] = i + n
    # Set the molecule indices
    universe.atom['molecule'] = universe.atom.index.map(lambda idx: mapper[idx])
    # Now compute molecule table
    universe.atom['mass'] = universe.atom['symbol'].map(Isotope.symbol_to_mass())
    universe.atom['xm'] = universe.atom['x'].mul(universe.atom['mass'])
    universe.atom['ym'] = universe.atom['y'].mul(universe.atom['mass'])
    universe.atom['zm'] = universe.atom['z'].mul(universe.atom['mass'])
    molecules = universe.atom.groupby('molecule')
    molecule = molecules['symbol'].value_counts().unstack().fillna(0).astype(np.int64)
    molecule.columns.name = None
    molecule['mass'] = molecules['mass'].sum()
    molecule['cx'] = molecules['xm'].sum() / molecule['mass']
    molecule['cy'] = molecules['ym'].sum() / molecule['mass']
    molecule['cz'] = molecules['zm'].sum() / molecule['mass']
    del universe.atom['mass']
    del universe.atom['xm']
    del universe.atom['ym']
    del universe.atom['zm']
    frame = universe.atom[['molecule', 'frame']].drop_duplicates('molecule')
    frame = frame.set_index('molecule')['frame'].astype(np.int64)
    molecule['frame'] = frame.astype('category')
    return Molecule(molecule)
