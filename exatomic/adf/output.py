# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Output Parser
#####################
Multiple frames are not currently supported
'''
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
        start = stop = self.find(_re_atm_00, keys_only=True)[-1] + 2
        while '<' not in self[stop]: stop += 1
        columns = ['tag', 'x', 'y', 'z']
        atom = self.pandas_dataframe(start, stop, columns)
        atom['symbol'] = atom['tag'].str.extract('.*\.(.*)', expand=False)
        for c in ['x', 'y', 'z']:
            atom[c] *= Length['A', 'au']
        atom['Z'] = atom['symbol'].map(symbol_to_z)
        atom['frame'] = 0
        self.atom = atom

    def parse_basis_set(self):
        frpi = 4 * np.pi
        ds = {0: np.sqrt(1/frpi), 1: np.sqrt(3/frpi),
              2: np.sqrt(5/frpi), 3: np.sqrt(7/frpi),
              4: np.sqrt(9/frpi), 5: np.sqrt(11/frpi)}
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
        df['shell'] = shells
        df['set'] = sets
        df['frame'] = 0
        df['d'] = df['L'].map(ds)
        self.basis_set = BasisSet(df, gaussian=False, spherical=False)
        self.atom['set'] = self.atom['symbol'].map(basmap)

    def parse_basis_set_order(self):
        data = {'center': [], 'symbol': [],
                  'seht': [],  'shell': [],
                     'L': [],      'l': [],
                     'm': [],      'n': []}
        sets = self.basis_set.groupby('set')
        for center, symbol, seht in zip(self.atom.index,
                                        self.atom['symbol'],
                                        self.atom['set']):
            bas = sets.get_group(seht)
            for L, shell in zip(bas['L'], bas['shell']):
                for l, m, n in enum_cartesian[L]:
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
        stop = starts[-1] + nchi
        skips = [i + j for i in list(starts[1:] - starts[0] - 3)
                       for j in range(3)]
        os = len(starts) * ncol > nchi * 1.5
        if os:
            brk = len(starts) // 2 - 1
            midskips = list(range(starts[brk] + nchi - starts[0],
                                  starts[brk + 1] - starts[0] - 3))
            skips = skips[:brk * 3] + midskips + skips[brk * 3:]
        data = self.pandas_dataframe(starts[0], stop, ncol + 1,
                                     skiprows=skips)
        data = data.drop(0, axis=1).unstack().dropna().values
        norb = data.shape[0] // (nchi * 2) if os else data.shape[0] // nchi
        arlen = data.shape[0]//2 if os else data.shape[0]
        cycle = np.ceil(nchi / ncol).astype(np.int64)
        chis = np.tile(range(nchi), norb)
        orbs = np.concatenate([np.repeat(range(i, norb, ncol), nchi)
                              for i in range(cycle)])
        self.momatrix = pd.DataFrame.from_dict({
                'coef': data[:arlen], 'chi': chis,
                'orbital': orbs[:arlen], 'frame': 0})
        if os: self.momatrix['coef1'] = data[arlen:]


_re_atm_00 = 'Coordinates'
_re_bas_00 = '(Slater-type)  F U N C T I O N S'
_re_bas_01 = 'BAS: List of all Elementary Cartesian Basis Functions'
_re_exc_00 = '(sum=1) transition dipole moment'
_re_exc_01 = ' no.     E/a.u.        E/eV      f           Symmetry'
_re_mos_00 = 'row'
