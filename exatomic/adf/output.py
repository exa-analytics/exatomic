# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
ADF Composite Output
#########################
This module provides the primary (user facing) output parser.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from collections import defaultdict
import re
import six
import numpy as np
import pandas as pd
from io import StringIO
from exa.util.units import Length
from exa import TypedMeta
from exatomic.base import sym2z
from exatomic.algorithms.basis import lmap, enum_cartesian
from exatomic.algorithms.numerical import dfac21
from ..core.atom import Atom
from exatomic.core.basis import BasisSet, BasisSetOrder
from ..core.orbital import Orbital, Excitation, MOMatrix
from .editor import Editor


class OutMeta(TypedMeta):
    atom = Atom
    basis_set = BasisSet
    basis_set_order = BasisSetOrder
    orbital = Orbital
    contribution = pd.DataFrame
    excitation = Excitation
    momatrix = MOMatrix
    sphr_momatrix = MOMatrix


class Output(six.with_metaclass(OutMeta, Editor)):
    """The ADF output parser."""
    def parse_atom(self):
        # TODO : only supports single frame, gets last atomic positions
        _re_atom_00 = 'Atoms in this Fragment     Cart. coord.s (Angstrom)'
        start = stop = self.find(_re_atom_00, keys_only=True)[0] + 2
        while self[stop].strip(): stop += 1
        atom = self.pandas_dataframe(start, stop, 7)
        atom.drop([0, 2, 3], axis=1, inplace=True)
        atom.columns = ['symbol', 'x', 'y', 'z']
        for c in ['x', 'y', 'z']: atom[c] *= Length['Angstrom', 'au']
        atom['Z'] = atom['symbol'].map(sym2z)
        atom['frame'] = 0
        self.atom = atom


    def parse_basis_set(self):
        # Find the basis set
        _re_bas_00 = '(Slater-type)  F U N C T I O N S'
        _re_bas_01 = 'Atom Type'
        start = self.find(_re_bas_00, keys_only=True)[-1] + 3
        starts = self.find(_re_bas_01, start=start, keys_only=True)
        lines = []
        for ext in starts:
            for i in range(4):
                lines.append(start + ext + i)
            stop = start + ext + 4
            while self[stop].strip():
                lines.append(stop)
                stop += 1
        df = pd.read_fwf(StringIO('\n'.join([self[i] for i in lines])),
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
        self.basis_set = BasisSet(df)
        self.meta['spherical'] = False
        self.atom['set'] = self.atom['symbol'].map(basmap)


    def parse_basis_set_order(self):
        # All the columns we need
        data = defaultdict(list)
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
                    for shell, r in zip(grp['shell'], grp['r']):
                        data['center'].append(center)
                        data['symbol'].append(symbol)
                        data['shell'].append(shell)
                        data['seht'].append(seht)
                        data['L'].append(L)
                        data['l'].append(l)
                        data['m'].append(m)
                        data['n'].append(n)
                        data['r'].append(r)
        data['set'] = data.pop('seht')
        data['frame'] = 0
        self.basis_set_order = pd.DataFrame.from_dict(data)
        self.basis_set_order['prefac'] = (self.basis_set_order['L'].apply(dfac21) /
                                          (self.basis_set_order['l'].apply(dfac21) *
                                           self.basis_set_order['m'].apply(dfac21) *
                                           self.basis_set_order['n'].apply(dfac21))
                                          ).apply(np.sqrt)

    def parse_orbital(self):
        _re_orb_00 = 'Orbital Energies, both Spins'
        _re_orb_01 = 'Orbital Energies, per Irrep and Spin'
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
        if 'spin' in cols[key]:
            df['spin'] = df.spin.map({'A': 0, 'B': 1})
            df.sort_values(by=['spin', 'energy'], inplace=True)
        else:
            df.sort_values(by='energy', inplace=True)
            df['spin'] = 0
        df.reset_index(drop=True, inplace=True)
        df['frame'] = df['group'] = 0
        self.orbital = df


    def parse_contribution(self):
        _re_con_00 = ('E(eV)  Occ       MO           %     '
                      'SFO (first member)   E(eV)  Occ   Fragment')
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
        dfs = dfs.applymap(lambda x: np.nan if (isinstance(x, six.string_types)
                                     and x.isspace()) else x)
        dfs.fillna(method='ffill', inplace=True)
        # Clean up
        dfs['symbol'] = dfs['symbol'].str.strip()
        dfs['angmom'] = dfs['angmom'].str.strip()
        dfs['angmom'].update(dfs['angmom'].map({'S': 'S:'}))
        dfs[['L', 'ml']] = dfs['angmom'].str.extract('(.*):(.*)', expand=True)
        dfs['%'] = dfs['%'].str.replace('%', '')
        dfs['%'].update(dfs['%'].map({"    ******": np.inf}))
        dfs['%'] = dfs['%'].astype(np.float64)
        dfs['occupation'] = dfs['occupation'].astype(np.float64)
        dfs['vector'] = dfs['vector'].astype(np.int64) - 1
        dfs['eV'] = dfs['eV'].astype(np.float64)
        dfs['atom'] -= 1
        self.contribution = dfs


    def parse_excitation(self):
        # Excitation
        _re_exc_00 = '(sum=1) transition dipole moment'
        _re_exc_01 = ' no.     E/a.u.        E/eV      f           Symmetry'
        found = self.find_next(_re_exc_00, keys_only=True)
        if not found: return
        # First table of interest here
        start = found + 4
        stop = self.find_next(_re_exc_01, keys_only=True) - 3
        os = len(self[start].split()) == 9
        todrop = ['occ:', 'virt:']
        cols = ['excitation', 'occ', 'drop', 'virt', 'weight', 'TDMx', 'TDMy', 'TDMz']
        if os: cols.insert(1, 'spin')
        if os: todrop = ['occ', 'virt']
        adf = self.pandas_dataframe(start, stop, cols)
        adf.drop('drop', axis=1, inplace=True)
        s1 = set(adf[cols[1]][adf[cols[1]] == 'NTO'].index)
        s2 = set(adf['excitation'][adf['excitation'].isin(todrop)].index)
        adf.drop(s1 | s2, axis=0, inplace=True)
        adf['excitation'] = adf['excitation'].str[:-1].astype(np.int64) - 1
        if os: adf['spin'] = adf['spin'].map({'Alph': 0, 'Beta': 1})
        adf[['occ', 'occsym']] = adf['occ'].str.extract('([0-9]*)(.*)', expand=True)
        adf[['virt', 'virtsym']] = adf['virt'].str.extract('([0-9]*)(.*)', expand=True)
        adf['occ'] = adf['occ'].astype(np.int64) - 1
        adf['virt'] = adf['virt'].astype(np.int64) - 1
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
        adf['frame'] = adf['group'] = 0
        self.excitation = adf


    def parse_momatrix(self):
        _re_mo_00 = 'Eigenvectors .* in BAS representation'
        _re_mo_01 = 'row '
        _re_mo_02 = 'nosym'
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
                skips = [k + j for k in list(block[1:] - block[0] - 3) for j in range(3)]
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

    def parse_sphr_momatrix(self, verbose=False):
        """
        Parser localized momatrix (if present).

        If the ``locorb`` keyword is used in ADF, an additional momatrix is
        printed after localization is performed. Parsing this table allows
        for visualization of these orbitals.

        Note:
            The attr :attr:`~exatomic.adf.output._re_loc_mo` is used for parsing this
            section.
        """
        _re_loc_mo = ("Localized MOs expanded in CFs+SFOs",
                      "SFO contributions (%) per Localized Orbital")
        found = self.find(*_re_loc_mo)
        if len(found[_re_loc_mo[0]]) == 0:
            if verbose:
                print("No localization performed.")
            return    # Nothing to parse
        start = found[_re_loc_mo[0]][0][0] + 8
        stop = found[_re_loc_mo[1]][0][0] - 4
        # Parse the localized momatrix as a whole block of text
        df = pd.read_fwf(StringIO("\n".join(self[start:stop])),
                         widths=(16, 9, 9, 9, 9, 9, 9, 9, 9), header=None)
        del df[0]
        # Identify the eigenvectors and (un)stack them correctly
        n = df[df[1].isnull()].index[0]   # number of basis functions
        m = np.ceil(df.shape[0]/n).astype(int)  # number of printed blocks of text
        # idx - indexes of "lines" (rows) that don't contain coefficients
        idx = [(n+5)*j + i - 5 for j in range(1, m) for i in range(0, 5)]
        df = df[~df.index.isin(idx)]
        coefs = []
        for i in range(0, df.shape[0]//n+1):
            d = df.iloc[n*(i-1):n*i, :]
            coefs.append(d.unstack().dropna().values.astype(float))
        coefs = np.concatenate(coefs)
        m = coefs.shape[0]//n    # Number of localized MOs
        momatrix = pd.DataFrame.from_dict({'coef': coefs,
                                           'orbital': [i for i in range(m) for _ in range(n)],
                                           'chi': [j for _ in range(m) for j in range(n)]})
        momatrix['frame'] = self.atom['frame'].unique()[-1]
        self.sphr_momatrix = momatrix


    def __init__(self, *args, **kwargs):
        super(Output, self).__init__(*args, **kwargs)
