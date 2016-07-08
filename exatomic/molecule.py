# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Molecule Table
###################
'''
import numpy as np
import pandas as pd
from networkx import Graph
from networkx.algorithms.components import connected_components
from itertools import combinations
from exa.numerical import DataFrame
from exa.relational.isotope import symbol_to_element_mass
from exa import DataFrame
from exatomic import Isotope
from exatomic.formula import string_to_dict


class Molecule(DataFrame):
    '''
    Description of molecules in the atomic universe.
    '''
    _index = ['molecule']
    _groupbys = ['frame']
    _categories = {'frame': np.int64, 'formula': str, 'classification': object}

    def classify(self, *classifiers, overwrite=False):
        '''
        Classify molecules into arbitrary categories.

        .. code-block:: Python

            u.molecule.classify(('Na', 'solute'), ('H(2)O(1)', 'solvent'))

        Args:
            classifiers: Any number of tuples of the form ("identifier", "label", exact) (see below)
            overwrite (bool): If true, overwrite existing ClassificationError

        Note:
            A classifier has 3 parts, "identifier", e.g. "H(2)O(1)", "label", e.g.
            "solvent", and exact (true or false). If exact is false (default),
            classification is greedy and (in this example) molecules with formulas
            "H(1)O(1)", "H(3)O(1)", etc. would get classified as "solvent". If,
            instead, exact were set to true, those molecules would remain
            unclassified.

        See Also:
            :func:`~exatomic.algorithms.nearest.compute_nearest_molecules`
        '''
        for c in classifiers:
            n = len(c)
            if n != 3 and n != 2:
                raise ClassificationError()
        if 'classification' not in self:
            self['classification'] = None
        for classifier in classifiers:
            identifier = string_to_dict(classifier[0])
            classification = classifier[1]
            exact = classifier[2] if len(classifier) == 3 else False
            this = self if overwrite else self[self['classification'].isnull()]
            for symbol, count in identifier.items():
                this = this[this[symbol] == count] if exact else this[this[symbol] >= 1]
            if len(this) > 0:
                self.ix[self.index.isin(this.index), 'classification'] = classification
            else:
                raise KeyError('No records found for {}, with identifier {}.'.format(classification, identifier))
        self['classification'] = self['classification'].astype('category')

    def compute_atom_count(self):
        '''
        Compute the molecular atom count.
        '''
        symbols = [col for col in self.columns if len(col) < 3 and col[0].istitle()]
        self['atom_count'] = self[symbols].sum(axis=1)


def compute_molecule(universe):
    '''
    Cluster atoms into molecules.

    The algorithm is to create a network graph containing every atom (in every
    frame as nodes and bonds as edges). Using this connectivity information,
    one can perform a (breadth first) traversal of the network graph to cluster
    all nodes (whose indices correspond to physical atoms).

    Args:
        universe (:class:`~exatomic.universe.Universe`): Atomic universe

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
    n = 1
    if len(mapper.values()) > 0:
        n += max(mapper.values())
    else:
        n -= 1
    idxs = universe.atom[universe.atom['bond_count'] == 0].index
    for i, index in enumerate(idxs):
        mapper[index] = i + n
    # Set the molecule indices
    universe.atom['molecule'] = universe.atom.index.map(lambda idx: mapper[idx])
    # Now compute molecule table
    universe.atom['mass'] = universe.atom['symbol'].map(symbol_to_element_mass)
    # The coordinates of visual_atom represent grouped molecules for
    # periodic calculations and absolute coordinates for free boundary conditions.
    molecules = universe.atom.groupby('molecule')
    molecule = molecules['symbol'].value_counts().unstack().fillna(0).astype(np.int64)
    molecule.columns.name = None
    molecule['frame'] = universe.atom.drop_duplicates('molecule').set_index('molecule')['frame']
    molecule['mass'] = molecules['mass'].sum()
    del universe.atom['mass']
    frame = universe.atom[['molecule', 'frame']].drop_duplicates('molecule')
    frame = frame.set_index('molecule')['frame'].astype(np.int64)
    molecule['frame'] = frame.astype('category')
    return Molecule(molecule)


def compute_molecule_com(universe):
    '''
    Compute molecules' center of mass.
    '''
    universe.atom['mass'] = universe.atom['symbol'].map(symbol_to_element_mass)
    universe.atom['xm'] = universe.visual_atom['x'].mul(universe.atom['mass'])
    universe.atom['ym'] = universe.visual_atom['y'].mul(universe.atom['mass'])
    universe.atom['zm'] = universe.visual_atom['z'].mul(universe.atom['mass'])
    molecules = universe.atom.groupby('molecule')
    molecule = (molecules['xm'].sum() / universe.molecule['mass']).to_frame()
    molecule.index.names = ['molecule']
    molecule.columns = ['cx']
    molecule['cy'] = molecules['ym'].sum() / universe.molecule['mass']
    molecule['cz'] = molecules['zm'].sum() / universe.molecule['mass']
    del universe.atom['xm']
    del universe.atom['ym']
    del universe.atom['zm']
    del universe.atom['mass']
    return molecule
