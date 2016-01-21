# -*- coding: utf-8 -*-
'''
Nearest Neighbors
===============================================
'''
from collections import defaultdict
from exa import _pd as pd
from exa.algorithms.itertools import unique_everseen


IDX = pd.IndexSlice


def get_formula_dict_from_string(formula):
    '''
    Args
        formula (str): Atomic style chemical formula
    Returns
        formula_dict (dict): Symbols and corresponding counts
    '''
    split = [item for sublist in [s.split(')') for s in formula.split('(')] for item in sublist]
    split = [value for value in split if value != '']
    obj = defaultdict(int)
    for i in range(0, len(split), 2):
        obj[split[i]] += int(split[i + 1])
    return obj


def get_nearest_neighbors(uni, solvent_count, solute_identifier,
                          solvent_identifier=None, include_ions=True):
    '''
    '''
    # Convert string (partial) formulas into dictionaries
    solute_dict = get_formula_dict_from_string(solute_identifier)
    solvent_dict = {}
    if solvent_identifier is not None:
        solvent_dict = get_formula_dict_from_string(solvent_identifier)
    solvent_symbols = list(solvent_dict.keys())

    # Figure out the solute ids in each frame
    def find_molecules(frame, keys):
        '''
        '''
        keys = list(keys)
        indexes = []
        for key in keys:
            indexes += frame.loc[frame.formula.str.contains(key)].index.get_level_values('molecule').tolist()
        return set(indexes)

    def find_solvents(frame, solute_molecule_ids):
        '''
        '''
        fdx = frame.index[0][0]
        sm = solute_molecule_ids.loc[fdx]
        if include_ions and len(solvent_dict) > 0:
            return frame.loc[~frame.index.isin(sm, 'molecule')].index.get_level_values('molecule').tolist()
        else:
            bad_idx = []
            solvs = frame.loc[~frame.index.isin(sm, 'molecule')]
            for atom in solvent_dict.keys():
                bad_idx += solvs.loc[~solvs.formula.str.contains(atom)].index.get_level_values('molecule').tolist()
            return solvs.loc[~solvs.index.isin(bad_idx, 'molecule')]

    grpd_molecule = uni.molecules.groupby(level='frame')
    solute_molecule_ids = grpd_molecule.apply(find_molecules, keys=solute_dict.keys())
    solvent_molecule_ids = grpd_molecule.apply(find_solvents, solute_molecule_ids)

    # Second convert the molecule_ids into onebody ids in order to get twobody distances (atom to atom)
    def map_molecule_to_one(frame, one, exclude=True):
        '''
        '''
        fdx = frame.index[0]
        indexes = []
        for molecule_id in frame.values:
            #t = one.loc[IDX[fdx, :, molecule_id], IDX[:]]
            t = one.ix[fdx, :]
            t = t.ix[t['super_molecule'] == molecule_id]
            if exclude:
                t = t.loc[~t.symbol.isin(solvent_symbols)]
            indexes += t.index.get_level_values('super_atom').tolist()
        return indexes

    solute_one_ids = solute_molecule_ids.groupby(level='frame').apply(map_molecule_to_one, one=uni.super_atoms)
    solute_one_ids_to_extract = solute_molecule_ids.groupby(level='frame').apply(map_molecule_to_one, one=uni.super_atoms, exclude=False)
    print(solute_one_ids)
    print(solute_one_ids_to_extract)
    solvent_one_ids = solvent_molecule_ids.groupby(level='frame').apply(map_molecule_to_one, one=uni.super_atoms, exclude=False)

    # Figure out the max solvent atom count and current solvent atom counts
    def count_atoms(frame, one, keys):
        '''
        '''
        fdx = frame.index[0]
        ones = frame.values[0]
        obj = one.loc[IDX[fdx, ones, :]]
        obj = obj.loc[obj.symbol.isin(keys), 'symbol'].value_counts()
        if len(obj) == 0:
            obj = pd.Series(dict([(key, 0.0) for key in keys]))
        return obj

    solvent_atom_counts = solute_one_ids_to_extract.groupby(level='frame').apply(count_atoms, one=uni.atoms, keys=solvent_dict.keys())
    solvent_atom_counts = solvent_atom_counts.to_frame().unstack()
    solvent_atom_counts.columns = solvent_atom_counts.columns.levels[1].tolist()
    max_atom_counts = solvent_atom_counts.copy()
    for key, value in solvent_dict.items():
        max_atom_counts.loc[IDX[:, key]] = value * solvent_count

    # Time to get the one body indexes
    def order_molecules(fdx):
        '''
        '''
        solute_ones = solute_one_ids[fdx]
        solute_molecules = solute_molecule_ids[fdx]
        solvent_ones = solvent_one_ids[fdx]
        tbp = uni.twobody.loc[IDX[fdx, :, :, :], IDX['distance']]
        tbp = tbp.loc[(tbp.index.isin(solute_ones, 'one1') & tbp.index.isin(solvent_ones, 'one2')) |
                      (tbp.index.isin(solute_ones, 'one2') & tbp.index.isin(solvent_ones, 'one1'))]
        tbp = tbp.sort_values().reset_index(['one1', 'one2'])[['one1', 'one2']]
        reone = uni.atoms.reset_index('molecule')['molecule']
        tbp = tbp.applymap(lambda f: reone.loc[IDX[fdx, f]])
        tbp = list(unique_everseen([i for values in tbp.values for i in values if i not in solute_molecules]))
        return tbp

    def get_counts(mid, fdx, keys):
        '''
        '''
        symbols = uni.atoms.loc[IDX[fdx, :, mid]]
        for key in keys:
            symbols = symbols.loc[symbols.symbol.isin(keys)]
        obj = symbols['symbol'].value_counts()
        if len(obj) == 0:
            obj = pd.Series(dict([(key, 0.0) for key in keys]))
        return obj

    def get_ones(mid, fdx):
        '''
        '''
        return uni.atoms.loc[IDX[fdx, :, mid], IDX[:]].index.get_level_values('one').tolist()

    def iterate(fdx, mids, sac, mac):
        '''
        '''
        one_ids = []
        # While loops are dangerous
        keep = []
        for mid in mids:
            if np.any(sac < mac):
                sac = sac.add(get_counts(mid, fdx, sac.keys()), fill_value=0)
                one_ids += get_ones(mid, fdx)
                keep.append(mid)
            else:
                break
        return one_ids, keep

    def get(frame, solvent_one_ids, solvent_atom_counts, max_atom_counts):
        '''
        '''
        fdx = frame.index[0]
        sac = solvent_atom_counts.loc[fdx]
        mac = max_atom_counts.loc[fdx]
        one_ids = frame.values[0]
        mids = list(solute_molecule_ids[fdx])
        if np.any(sac < mac):
            # convert solvent atoms to ordered molecules by atom to atom distance
            solv_mids = order_molecules(fdx)    # This is heavy
            ods, mds = iterate(fdx, solv_mids, sac, mac)
            one_ids += ods
            mids += mds
        return one_ids, mids

    one_ids = solute_one_ids_to_extract.groupby(level='frame').apply(get, solvent_one_ids, solvent_atom_counts, max_atom_counts)

    def one_getter(frame):
        '''
        '''
        fdx = frame.index[0][0]
        odx = one_ids.loc[fdx][0]
        obj = frame.loc[fdx]
        return obj.loc[obj.index.isin(odx, 'one')]

    def two_getter(frame):
        '''
        '''
        fdx = frame.index[0][0]
        odx = one_ids.loc[fdx][0]
        obj = frame.loc[fdx]
        return obj.loc[frame.index.isin(odx, 'one1') & frame.index.isin(odx, 'one2')]

    def molecule_getter(frame):
        '''
        '''
        fdx = frame.index[0][0]
        mdx = one_ids.loc[fdx][1]
        obj = frame.loc[fdx]
        return obj.loc[frame.index.isin(mdx, 'molecule')]

    # THE COUNTS NEED TO BE UPDATED
    kwargs = {'name': uni.name, 'description': uni.description, 'frames': uni.frames.copy(), 'meta': uni.meta}
    kwargs['atoms'] = uni.atoms.groupby(level='frame').apply(one_getter).copy()
    kwargs['twobody'] = uni.twobody.groupby(level='frame').apply(two_getter).copy()
    kwargs['molecules'] = uni.molecules.groupby(level='frame').apply(molecule_getter).copy()
    kwargs['molecules'].loc[:, 'label'] = 'solvent'
    for key in solute_dict:
        kwargs['molecules'].loc[kwargs['molecules']['formula'].str.contains(key), 'label'] = 'analyte'
    #kwargs['update_counts'] = True
    return kwargs
