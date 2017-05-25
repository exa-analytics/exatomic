# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Output Parser
#####################
Multiple frames are not currently supported
'''
import re
import numpy as np
import pandas as pd
from io import StringIO

from exa.relational.isotope import symbol_to_z
from exatomic.algorithms.basis import lmap, enum_cartesian
from exatomic.basis import BasisSet
from exatomic import Length
from .editor import Editor

symbol_to_z = symbol_to_z()

class Output(Editor):


    def parse_atom(self):
        # TODO : only supports single frame, gets last atomic positions
        start = stop = self.find(_re_bso_00, keys_only=True)[0] + 2
        while self[stop].strip(): stop += 1
        atom = self.pandas_dataframe(start, stop, 7)
        atom.drop([0, 2, 3], axis=1, inplace=True)
        atom.columns = ['symbol', 'x', 'y', 'z']
        for c in ['x', 'y', 'z']: atom[c] *= Length['A', 'au']
        atom['Z'] = atom['symbol'].map(symbol_to_z)
        atom['frame'] = 0
        self.atom = atom


    def parse_basis_set(self):
        # Find the basis set
        start = self.find(_re_bas_00, keys_only=True)[-1] + 3
        stopa = self.find_next(_re_bas_01, start=start, keys_only=True)
        stopb = self.find_next(_re_bas_02, start=start, keys_only=True)
        try: stop = min(stopa, stopb)
        except TypeError: stop = stopa
        # Grab everything
        df = pd.read_fwf(StringIO('\n'.join(self[start:stop])),
                         widths=[4, 2, 12, 4],
                         names=['n', 'L', 'alpha', 'symbol'])
        # Where atom types change
        idxs = [0] + df['n'][df['n'] == '---'].index.tolist() + [df.shape[0]]
        sets, shells = [], []
        for i, (start, stop) in enumerate(zip(idxs, idxs[1:])):
            sets.append(np.repeat(i - 1, stop - start))
            shells.append(np.arange(-1, stop - start - 1))
        df['set'] = np.concatenate(sets)
        df['shell'] = np.concatenate(shells)
        # Atom table basis set map
        basmap = df['symbol'].dropna()
        basmap = basmap[basmap.str.endswith(')')].str.strip(')')
        basmap = {val: df['set'][key] + 1 for
                  key, val in basmap.to_dict().items()}
        # Discard the garbage
        drop = df['n'].str.strip().str.isnumeric().fillna(False)
        df.drop(drop[drop == False].index, inplace=True)
        df.drop('symbol', axis=1, inplace=True)
        # Clean up the series
        df['alpha'] = df['alpha'].astype(np.float64)
        df['n'] = df['n'].astype(np.int64)
        df['L'] = df['L'].str.lower().map(lmap)
        df['d'] = np.sqrt((2 * df['L'] + 1) / (4 * np.pi))
        df['r'] = df['n'] - (df['L'] + 1)
        df['frame'] = 0
        self.basis_set = BasisSet(df, gaussian=False, spherical=False)
        self.atom['set'] = self.atom['symbol'].map(basmap)


    def parse_basis_set_order(self):
        # All the columns we need
        data = {'center': [], 'symbol': [],
                  'seht': [],  'shell': [],
                     'L': [],      'l': [],
                     'm': [],      'n': [],
                     'r': [], 'prefac': []}
        sets = self.basis_set.groupby('set')
        # Iterate over atoms
        for center, symbol, seht in zip(self.atom.index,
                                        self.atom['symbol'],
                                        self.atom['set']):
            # Per basis set
            bas = sets.get_group(seht).groupby('L')
            for L, grp in bas:
                # Iterate over cartesians
                for l, m, n in enum_cartesian[L]:
                    # Wonky normalization in ADF
                    prefac = 0
                    if L == 2: prefac = 0 if any(i == L for i in (l, m, n)) else np.sqrt(L + 1)
                    elif L == 3: prefac = np.sqrt(5 * sum((i == 1 for i in (l, m, n))))
                    # Pre-exponential factors (shell kind of pointless for STOs)
                    for shell, r in zip(grp['shell'], grp['r']):
                        for key in data.keys(): data[key].append(eval(key))
        data['set'] = data.pop('seht')
        data['frame'] = 0
        self.basis_set_order = pd.DataFrame.from_dict(data)


    def parse_orbital(self):
        found = self.find(_re_orb_00, _re_orb_01, keys_only=True)
        # Open shell vs. closed shell
        cols = {
            _re_orb_00: ['symmetry', 'vector', 'spin', 'occupation', 'energy', 'eV'],
            _re_orb_01: ['vector', 'occupation', 'energy', 'eV', 'dE']}
        key = _re_orb_00 if found[_re_orb_00] else _re_orb_01
        start = stop = found[key][-1] + 5
        while self[stop].strip(): stop += 1
        df = self.pandas_dataframe(start, stop, cols[key])
        df['vector'] -= 1
        df['spin'] = df.spin.map({'A': 0, 'B': 1})
        df.sort_values(by=['spin', 'energy'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['frame'] = df['group'] = 0
        self.orbital = df


    def parse_contribution(self):
        # MO contribution by percentage
        found = self.find(_re_con_00, keys_only=True)
        starts = [i + 3 for i in found]
        widths = [12, 6, 6, 6, 11, 6, 10, 12, 6, 6, 3]
        names = ['eV', 'occupation', 'vector', 'sym', '%', 'SFO',
                 'angmom', 'eV(sfo)', 'occ(sfo)', 'atom', 'symbol']
        dfs = []
        # Prints for both spins
        for i, start in enumerate(starts):
            stop = start
            while self[stop].strip(): stop += 1
            dfs.append(pd.read_fwf(StringIO('\n'.join(self[start:stop])),
                                   delim_whitespace=True, widths=widths,
                                   names=names))
            dfs[-1]['spin'] = i
        dfs = pd.concat(dfs).reset_index(drop=True)
        # Maybe a better way to do this
        def _snan(x):
            return np.nan if isinstance(x, str) and x.isspace() else x
        dfs = dfs.applymap(_snan)
        dfs.fillna(method='ffill', inplace=True)
        # Clean up
        dfs['symbol'] = dfs['symbol'].str.strip()
        dfs['angmom'] = dfs['angmom'].str.strip()
        dfs['angmom'].update(dfs['angmom'].map({'S': 'S:'}))
        dfs[['L', 'ml']] = dfs['angmom'].str.extract('(.*):(.*)', expand=True)
        dfs['%'] = dfs['%'].str.replace('%', '')
        dfs['%'].update(dfs['%'].map({"    ******": np.inf})
        dfs['%'] = dfs['%'].astype(np.float64)
        dfs['occupation'] = dfs['occupation'].astype(np.float64)
        dfs['vector'] = dfs['vector'].astype(np.int64) - 1
        dfs['eV'] = dfs['eV'].astype(np.float64)
        dfs['atom'] -= 1
        self.contribution = dfs


    def parse_excitation(self):
        found = self.find_next(_re_exc_00, keys_only=True)
        if not found: return
        # First table of interest here
        start = found + 4
        stop = self.find_next(_re_exc_01, keys_only=True) - 3
        adf = self.pandas_dataframe(start, stop, 9)
        adf.drop(3, axis=1, inplace=True)
        adf[0] = adf[0].str[:-1].astype(np.int64) - 1
        adf[1] = adf[1].map({'Alph': 0, 'Beta': 1})
        adf[[2, 'occsym']] = adf[2].str.extract('([0-9]*)(.*)', expand=True)
        adf[[4, 'virtsym']] = adf[4].str.extract('([0-9]*)(.*)', expand=True)
        adf[2] = adf[2].astype(np.int64)
        adf[4] = adf[4].astype(np.int64)
        adf.rename(columns={0: 'excitation', 1: 'spin', 2: 'occ', 4: 'virt',
                            5: 'weight', 6: 'TDMx', 7: 'TDMy', 8: 'TDMz'},
                            inplace=True)
        # Second one here
        start = stop + 5
        stop = start
        while self[stop].strip(): stop += 1
        cols = _re_exc_01.split()
        df = self.pandas_dataframe(start, stop + 1, cols)
        df.drop(cols[0], axis=1, inplace=True)
        df.columns = ['energy', 'eV', 'osc', 'symmetry']
        # Expand the second table to fit the original
        for col in df.columns: adf[col] = adf.excitation.map(df[col])
        adf['occ'] -= 1
        adf['virt'] -= 1
        adf['frame'] = adf['group'] = 0
        self.excitation = adf


    def parse_momatrix(self):
        found = self.regex(_re_mo_00, _re_mo_01, _re_mo_02,
                           flags=re.IGNORECASE, keys_only=True)
        if not found[_re_mo_00] or not found[_re_mo_01]: return
        if found[_re_mo_02]:
            thresh = found[_re_mo_00][0]
            rowmajor = 'rows' in self[thresh]
            starts = np.array([i for i in found[_re_mo_01] if i > thresh]) + 1
            nchi = starts[1] - starts[0] - 3
            ncol = len(self[starts[0] + 1].split()) - 1
            if len(starts) % 2: os = False
            else:
                anchor = starts[len(starts)//2 - 1] + nchi
                sail = starts[len(starts)//2]
                os = True if self.find('SPIN 2', start=anchor, stop=sail) else False
            blocks = [starts] if not os else [starts[:len(starts)//2],
                                              starts[len(starts)//2:]]
            data = pd.DataFrame()
            for i, block in enumerate(blocks):
                stop = block[-1] + nchi
                skips = [i + j for i in list(block[1:] - block[0] - 3) for j in range(3)]
                name = 'coef' if not i else 'coef{}'.format(i)
                col = self.pandas_dataframe(block[0], stop, ncol + 1,
                                            skiprows=skips).drop(0, axis=1,
                                            ).unstack().dropna().reset_index(drop=True)
                data[name] = col
            norb = len(data.index) // nchi
            data['orbital'] = np.concatenate([np.repeat(range(i, norb, ncol), nchi)
                                              for i in range(ncol)])
            data['chi'] = np.tile(range(nchi), norb)
            data['frame'] = 0
            if rowmajor:
                data.rename(columns={'orbital': 'chi', 'chi': 'orbital'}, inplace=True)
                data.sort_values(by=['orbital', 'chi'], inplace=True)
            self.momatrix = data
        else:
            print('Symmetrized calcs not supported yet.')


# Atom
_re_bso_00 = 'Atoms in this Fragment     Cart. coord.s (Angstrom)'
# Basis Set
_re_bas_00 = '(Slater-type)  F U N C T I O N S'
_re_bas_01 = 'BAS: List of all Elementary Cartesian Basis Functions'
_re_bas_02 = 'Frozen Core Shells'
# Orbital
_re_orb_00 = 'Orbital Energies, both Spins'
_re_orb_01 = 'Orbital Energies, per Irrep and Spin'
# Contribution
_re_con_00 = 'E(eV)  Occ       MO           %     SFO (first member)   E(eV)  Occ   Fragment'
# Excitation
_re_exc_00 = '(sum=1) transition dipole moment'
_re_exc_01 = ' no.     E/a.u.        E/eV      f           Symmetry'
# MOMatrix
_re_mo_00 = 'Eigenvectors .* in BAS representation'
_re_mo_01 = 'row '
_re_mo_02 = 'nosym'
