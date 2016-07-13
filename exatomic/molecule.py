# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Molecule Table
###################
'''
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.components import connected_components
from exa.numerical import DataFrame
from exa.relational.isotope import symbol_to_element_mass
from exatomic.formula import string_to_dict


class Molecule(DataFrame):
    '''
    Description of molecules in the atomic universe.
    '''
    _index = ['molecule']
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
    Cluster atoms into molecules and create the :class:`~exatomic.molecule.Molecule`
    table.

    Args:
        universe: Atomic universe

    Returns:
        molecule: Molecule table

    Warning:
        This function modifies the universe's atom (:class:`~exatomic.atom.Atom`)
        table in place!
    '''
    nodes = universe.atom.index.values
    bonded = universe.two.ix[universe.two['bond'] == True, ['atom0', 'atom1']]
    edges = zip(bonded['atom0'].astype(np.int64), bonded['atom1'].astype(np.int64))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    # generate molecule indices for the atom table
    mapper = {}
    i = 0
    for k, v in g.degree().items():    # First handle single atom "molecules"
        if v == 0:
            mapper[k] = i
            i += 1
    for seht in connected_components(g):    # Second handle multi atom molecules
        for adx in seht:
            mapper[adx] = i
        i += 1
    universe.atom['molecule'] = universe.atom.index.map(lambda x: mapper[x])
    sym2mass = symbol_to_element_mass()
    universe.atom['mass'] = universe.atom['symbol'].map(sym2mass)
    grps = universe.atom.groupby('molecule')
    molecule = grps['symbol'].value_counts().unstack().fillna(0).astype(np.int64)
    molecule.columns.name = None
    molecule['mass'] = grps['mass'].sum()
    universe.atom['molecule'] = universe.atom['molecule'].astype('category')
    return molecule


#def compute_molecule_com(universe):
#    '''
#    Compute molecules' center of mass.
#    '''
#    universe.atom['mass'] = universe.atom['symbol'].map(symbol_to_element_mass)
#    universe.atom['xm'] = universe.visual_atom['x'].mul(universe.atom['mass'])
#    universe.atom['ym'] = universe.visual_atom['y'].mul(universe.atom['mass'])
#    universe.atom['zm'] = universe.visual_atom['z'].mul(universe.atom['mass'])
#    molecules = universe.atom.groupby('molecule')
#    molecule = (molecules['xm'].sum() / universe.molecule['mass']).to_frame()
#    molecule.index.names = ['molecule']
#    molecule.columns = ['cx']
#    molecule['cy'] = molecules['ym'].sum() / universe.molecule['mass']
#    molecule['cz'] = molecules['zm'].sum() / universe.molecule['mass']
#    del universe.atom['xm']
#    del universe.atom['ym']
#    del universe.atom['zm']
#    del universe.atom['mass']
#    return molecule
#
