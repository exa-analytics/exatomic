# -*- coding: utf-8 -*-
'''
Molecule Information DataFrame
===============================
Molecules are collections of bonded atoms.
'''
import networkx as nx
from networkx.algorithms.components.connected import connected_components
from itertools import product
from exa import DataFrame, Config
from exa import _np as np
from exa import _pd as pd
if Config.numba:
    from exa.jitted.iteration import repeat_i8
else:
    import numpy.repeat as repeat_i8
from atomic import Isotope
from atomic.tools import formula_dict_to_string

class Molecule(DataFrame):
    '''
    '''
    __indices__ = ['frame', 'molecule']
    __columns__ = ['formula', 'mass', 'cx', 'cy', 'cz']


class PeriodicMolecule(DataFrame):
    '''
    '''
    __indices__ = ['frame', 'molecule']
    __columns__ = ['cx', 'cy', 'cz', 'mass']


def _periodic_molecules(universe):
    '''
    '''
    grouped_bonded = universe.twobody.ix[(universe.twobody['bond'] == True), ['super_atom1', 'super_atom2']].groupby(level='frame')
    n = grouped_bonded.ngroups
    formulas = np.empty((n, ), dtype='O')
    frame_indices = np.empty((n, ), dtype='O')
    molecule_indices = np.empty((n, ), dtype='O')
    masses = np.empty((n, ), dtype='O')
    cx = np.empty((n, ), dtype='O')
    cy = np.empty((n, ), dtype='O')
    cz = np.empty((n, ), dtype='O')
    universe.super_atoms['super_molecule'] = None
    for i, (fdx, group) in enumerate(grouped_bonded):
        graph = nx.Graph()
        pairs = group.values
        for pair in pairs:
            graph.add_nodes_from(pair)
        graph.add_edges_from(pairs)
        single_atoms_index = universe.atoms.ix[(universe.atoms['bond_count'] == 0)].index.values
        super_single = set(universe.super_atoms.ix[single_atoms_index].index.get_level_values('super_atom'))
        grouped_indices = [list(mol) for mol in connected_components(graph)] + [[single] for single in super_single]
        m = len(grouped_indices)
        current_formulas = np.empty((m, ), dtype='O')
        current_masses = np.empty((m, ), dtype='f8')
        current_cx = np.empty((m, ), dtype='f8')
        current_cy = np.empty((m, ), dtype='f8')
        current_cz = np.empty((m, ), dtype='f8')
        for j, super_atom_indices in enumerate(grouped_indices):
            indices = list(product([fdx], super_atom_indices))
            universe.super_atoms.ix[indices, 'super_molecule'] = j
            xyz = universe.super_atoms.ix[indices, ['x', 'y', 'z']].values
            atom_indices = universe.super_atoms.ix[indices, 'atom'].values.tolist()
            indices = list(product([fdx], atom_indices))
            symbols = universe.atoms.ix[indices, 'symbol']
            atoms_masses = symbols.map(Isotope.symbol_mass).values
            current_formulas[j] = formula_dict_to_string(symbols.value_counts().to_dict())
            current_masses[j] = atoms_masses.sum()
            com = (xyz * atoms_masses[:, np.newaxis]).sum(axis=0) / current_masses[j]
            current_cx[j], current_cy[j], current_cz[j] = com
        formulas[i] = current_formulas
        molecule_indices[i] = range(m)
        frame_indices[i] = repeat_i8(fdx, m)
        masses[i] = current_masses
        cx[i] = current_cx
        cy[i] = current_cy
        cz[i] = current_cz

    molecules = pd.DataFrame.from_dict({
        'molecule': np.concatenate(molecule_indices),
        'formula': np.concatenate(formulas),
        'mass': np.concatenate(masses),
        'cx': np.concatenate(cx),
        'cy': np.concatenate(cy),
        'cz': np.concatenate(cz),
        'frame': np.concatenate(frame_indices)
    })
    molecules['molecule'] = molecules['molecule'].astype(np.int64)
    molecules.set_index(['frame', 'molecule'], inplace=True)
    return PeriodicMolecule(molecules)
