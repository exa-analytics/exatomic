# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Output Parser
#####################
Multiple frames are not currently supported
'''
import numpy as np
import pandas as pd

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
        for c in ['x', 'y', 'z']:
            atom[c] *= Length['A', 'au']
        atom['Z'] = atom['symbol'].map(symbol_to_z)
        atom['frame'] = 0
        self.atom = atom

    def parse_basis_set(self):
        dfs, basmap = [], {}
        start = self.find(_re_bas_00, keys_only=True)[-1] + 3
        stop = self.find_next(_re_bas_01, start=start, keys_only=True)
        skips, sets, shells = [], [], []
        seht, shell = -1, 0
        for i, ln in enumerate(self[start:stop]):
            if 'Atom' in ln:
                seht += 1
                shell = 0
                basmap[ln.split('(')[-1][:-1]] = seht
            if len(ln.split()) == 3:
                sets.append(seht)
                shells.append(shell)
                shell += 1
            else:
                skips.append(i)
        df = self.pandas_dataframe(start, stop, ['n', 'L', 'alpha'],
                                   skiprows=skips)
        df['L'] = df['L'].str.lower().map(lmap)
        df['d'] = np.sqrt((2 * df['L'] + 1) / (4 * np.pi))
        df['shell'] = shells
        df['set'] = sets
        df['frame'] = 0
        df['r'] = df['n'] - (df['L'] + 1)
        self.basis_set = BasisSet(df, gaussian=False, spherical=False)
        self.atom['set'] = self.atom['symbol'].map(basmap)

    def parse_basis_set_order(self):
        data = {'center': [], 'symbol': [],
                  'seht': [],  'shell': [],
                     'L': [],      'l': [],
                     'm': [],      'n': [],
                     'r': [], 'prefac': []}
        sets = self.basis_set.groupby('set')
        for center, symbol, seht in zip(self.atom.index,
                                        self.atom['symbol'],
                                        self.atom['set']):
            bas = sets.get_group(seht)
            for L, shell, N, pre in zip(bas['L'], bas['shell'],
                                        bas['N'], bas['n']):
                for l, m, n in enum_cartesian[L]:
                    if any(i == L for i in (l, m, n)):
                        prefac = N
                    else:
                        prefac = N * np.sqrt(pre)
                    for key in data.keys():
                        data[key].append(eval(key))
        data['set'] = data.pop('seht')
        self.basis_set_order = pd.DataFrame.from_dict(data)

    def parse_excitation(self):
        found = self.find_next(_re_exc_00, keys_only=True)
        if not found: return
        start = found + 4
        stop = self.find_next(_re_exc_01, keys_only=True) - 3
        adf = self.pandas_dataframe(start, stop, 9)
        adf.drop([0, 3, 5, 6, 7, 8], axis=1, inplace=True)
        start = stop + 5
        stop = start + adf.shape[0]
        cols = _re_exc_01.split()
        df = self.pandas_dataframe(start, stop, cols)
        df.drop(cols[0], axis=1, inplace=True)
        df.columns = ['energy', 'eV', 'osc', 'symmetry']
        df['spin'] = adf[1].map({'Alph': 0, 'Beta': 1})
        df[['occ', 'occsym']] = adf[2].str.extract('([0-9]*)(.*)', expand=True)
        df[['virt', 'virtsym']] = adf[4].str.extract('([0-9]*)(.*)', expand=True)
        df['frame'] = 0
        df['group'] = 0
        self.excitation = df

    def parse_momatrix(self):
        found = self.find(_re_mos_00, keys_only=True)
        if not found: return
        starts = np.array(found) + 1
        ncol = len(self[starts[0] + 1].split()) - 1
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
            col = 'coef' if not i else 'coef{}'.format(i)
            data[col] = self.pandas_dataframe(block[0], stop, ncol + 1,
                                              skiprows=skips).drop(0, axis=1,
                                              ).unstack().dropna().reset_index(drop=True)
        norb = len(data.index) // nchi
        print(data.shape)
        print(norb)
        print(nchi)
        data['chi'] = np.tile(range(nchi), norb)
        data['orbital'] = np.concatenate([np.repeat(range(i, norb, ncol), nchi)
                                          for i in range(ncol)])
        data['frame'] = 0
        self.momatrix = data


_re_bso_00 = 'Atoms in this Fragment     Cart. coord.s (Angstrom)'
_re_atm_00 = 'Coordinates'
_re_bas_00 = '(Slater-type)  F U N C T I O N S'
_re_bas_01 = 'BAS: List of all Elementary Cartesian Basis Functions'
_re_exc_00 = '(sum=1) transition dipole moment'
_re_exc_01 = ' no.     E/a.u.        E/eV      f           Symmetry'
_re_mos_00 = 'row'
