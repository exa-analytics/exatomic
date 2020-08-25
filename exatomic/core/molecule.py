# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Molecule Table
###################
"""
import numpy as np
import pandas as pd
import networkx as nx
import warnings
from networkx.algorithms.components import connected_components
from exa import DataFrame
from exatomic.base import sym2mass
from exatomic.formula import string_to_dict, dict_to_string


class Molecule(DataFrame):
    """
    Description of molecules in the atomic universe.
    """
    _index = 'molecule'
    _categories = {'frame': np.int64, 'formula': str, 'classification': object}

    #@property
    #def _constructor(self):
    #    return Molecule

    def classify(self, *classifiers):
        """
        Classify molecules into arbitrary categories.

        .. code-block:: Python

            u.molecule.classify(('solute', 'Na'), ('solvent', 'H(2)O(1)'))

        Args:
            classifiers: Any number of tuples of the form ('label', 'identifier', exact) (see below)

        Note:
            A classifier has 3 parts, "label", e.g. "solvent", "identifier", e.g.
            "H(2)O(1)", and exact (true or false). If exact is false (default),
            classification is greedy and (in this example) molecules with formulas
            "H(1)O(1)", "H(3)O(1)", etc. would get classified as "solvent". If,
            instead, exact were set to true, those molecules would remain
            unclassified.

        Warning:
            Classifiers are applied in the order passed; where identifiers overlap,
            the latter classification is used.

        See Also:
            :func:`~exatomic.algorithms.nearest.compute_nearest_molecules`
        """
        for c in classifiers:
            n = len(c)
            if n != 3 and n != 2:
                raise ClassificationError()
        self['classification'] = None
        for classifier in classifiers:
            identifier = string_to_dict(classifier[0])
            classification = classifier[1]
            exact = classifier[2] if len(classifier) == 3 else False
            this = self
            for symbol, count in identifier.items():
                this = this[this[symbol] == count] if exact else this[this[symbol] >= 1]
            if len(this) > 0:
                self.ix[self.index.isin(this.index), 'classification'] = classification
            else:
                raise KeyError('No records found for {}, with identifier {}.'.format(classification, identifier))
        self['classification'] = self['classification'].astype('category')
        if len(self[self['classification'].isnull()]) > 0:
            warnings.warn("Unclassified molecules remaining...")

    def get_atom_count(self):
        """
        Compute the number of atoms per molecule.
        """
        symbols = self._get_symbols()
        return self[symbols].sum(axis=1)

    def get_formula(self, as_map=False):
        """
        Compute the string representation of the molecule.
        """
        symbols = self._get_symbols()
        mcules = self[symbols].to_dict(orient='index')
        ret = map(dict_to_string, mcules.values())
        if as_map:
            return ret
        return list(ret)

    def _get_symbols(self):
        """
        Helper method to get atom symbols.
        """
        return [col for col in self if len(col) < 3 and col[0].istitle()]


def compute_molecule(universe):
    """
    Cluster atoms into molecules and create the :class:`~exatomic.molecule.Molecule`
    table.

    Args:
        universe: Atomic universe

    Returns:
        molecule: Molecule table

    Warning:
        This function modifies the universe's atom (:class:`~exatomic.atom.Atom`)
        table in place!
    """
    nodes = universe.atom.index.values
    bonded = universe.atom_two.loc[universe.atom_two['bond'] == True, ['atom0', 'atom1']]
    edges = zip(bonded['atom0'].astype(np.int64), bonded['atom1'].astype(np.int64))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    # generate molecule indices for the atom table
    mapper = {}
    i = 0
    for k, v in g.degree():    # First handle single atom "molecules"
        if v == 0:
            mapper[k] = i
            i += 1
    for seht in connected_components(g):    # Second handle multi atom molecules
        for adx in seht:
            mapper[adx] = i
        i += 1
    universe.atom['molecule'] = universe.atom.index.map(lambda x: mapper[x])
    universe.atom['mass'] = universe.atom['symbol'].map(sym2mass).astype(float)
    grps = universe.atom.groupby('molecule')
    molecule = grps['symbol'].value_counts().unstack().fillna(0).astype(np.int64)
    molecule.columns.name = None
    molecule['mass'] = grps['mass'].sum()
    universe.atom['molecule'] = universe.atom['molecule'].astype('category')
    del universe.atom['mass']
    return molecule


def compute_molecule_count(universe):
    """
    """
    if 'molecule' not in universe.atom.columns:
        universe.compute_molecule()
    universe.atom._revert_categories()
    mapper = universe.atom.drop_duplicates('molecule').set_index('molecule')['frame']
    universe.atom._set_categories()
    universe.molecule['frame'] = universe.molecule.index.map(lambda x: mapper[x])
    molecule_count = universe.molecule.groupby('frame').size()
    del universe.molecule['frame']
    return molecule_count


def compute_molecule_com(universe):
    """
    Compute molecules' centers of mass.
    """
    if 'molecule' not in universe.atom.columns:
        universe.compute_molecule()
    mass = universe.atom.get_element_masses()
    if universe.frame.is_periodic():
        xyz = universe.atom[['x', 'y', 'z']].copy()
        xyz.update(universe.visual_atom)
    else:
        xyz = universe.atom[['x', 'y', 'z']]
    xm = xyz['x'].mul(mass)
    ym = xyz['y'].mul(mass)
    zm = xyz['z'].mul(mass)
    #rm = xm.add(ym).add(zm)
    df = pd.DataFrame.from_dict({'xm': xm, 'ym': ym, 'zm': zm, 'mass': mass,
                                 'molecule': universe.atom['molecule']})
    groups = df.groupby('molecule')
    sums = groups.sum()
    cx = sums['xm'].div(sums['mass'])
    cy = sums['ym'].div(sums['mass'])
    cz = sums['zm'].div(sums['mass'])
    return cx, cy, cz
