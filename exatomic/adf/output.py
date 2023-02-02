# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
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
from exatomic.exa.util.units import Length
from exatomic.exa import TypedMeta, Editor as _Editor
from exatomic.base import sym2z
from exatomic.algorithms.basis import lmap, enum_cartesian
from exatomic.algorithms.numerical import dfac21
from exatomic.core.atom import Atom, Frequency
from exatomic.core.gradient import Gradient
from exatomic.core.basis import BasisSet, BasisSetOrder
from exatomic.core.orbital import Orbital, Excitation, MOMatrix
from exatomic.core.tensor import NMRShielding, JCoupling
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
    gradient = Gradient
    frequency = Frequency
    nmr_shielding = NMRShielding
    j_coupling = JCoupling
    electric_dipole = pd.DataFrame
    magnetic_dipole = pd.DataFrame

class NMR(six.with_metaclass(OutMeta, Editor)):
    def parse_atom(self):
        # use the regex instead of find because we have a similar search
        # string in an nmr and cpl calculation for the nuclear coordinates
        _reatom = "(?i)NUCLEAR COORDINATES"
        found = self.regex(_reatom, keys_only=True)
        atom = []
        for idx, val in enumerate(found):
            start = val + 3
            stop = start
            while self[stop].strip(): stop += 1
            # a bit of a hack to make sure that there is no formatting change
            # depending on the number of atoms in the molecule as the index is
            # right justified so if there are more than 100 atoms it will fill
            # the alloted space for the atom index and change the delimitter
            # and therefore the number of columns
            self[start:stop] = map(lambda x: x.replace('(', ''), self[start:stop])
            df = self.pandas_dataframe(start, stop, ncol=5)
            df.columns = ['symbol', 'set', 'x', 'y', 'z']
            df[['x', 'y', 'z']] *= [Length['Angstrom', 'au']]*3
            df['Z'] = df['symbol'].map(sym2z)
            df['frame'] = idx
            # remove the trailing chracters from the index
            df['set'] = list(map(lambda x: x.replace('):', ''), df['set']))
            df['set'] = df['set'].astype(int) - 1
            atom.append(df)
        atom = pd.concat(atom)
        self.atom = atom

    def parse_nmr_shielding(self):
        ''' Parse the NMR shielding in the cartesian axis represntation. '''
        # TODO: Figure out what units the pricipal axis is printed in
        #       and add a keyword argument to parse them.
        _reatom = "N U C L E U S :"
        _reshield = "==== total shielding tensor"
        _renatom = "NUCLEAR COORDINATES (ANGSTROMS)"
        found = self.find(_reatom, keys_only=True)
        if not found:
            return
        ncalc = self.find(_renatom, keys_only=True)
        ncalc.append(len(self))
        ndx = 0
        dfs = []
        for start in found:
            try:
                ndx = ndx if start > ncalc[ndx] and start < ncalc[ndx+1] else ndx+1
            except IndexError:
                raise IndexError("It seems that there was an issue with " \
                                 +"determining which NMR calculation we are in.")
            start_shield = self.find(_reshield, keys_only=True, start=start)[0] \
                           + start + 2
            end_shield = start_shield + 3
            symbol, index = self[start].split()[-1].split('(')
            index = int(index.replace(')', ''))
            isotropic = float(self[start_shield+4].split()[-1])
            df = self.pandas_dataframe(start_shield, end_shield, ncol=3)
            cols = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
            df = pd.DataFrame(df.unstack().values.reshape(1,9), columns=cols)
            df['isotropic'] = isotropic
            df['atom'] = index - 1
            df['symbol'] = symbol
            df['label'] = 'nmr shielding'
            df['frame'] = ndx
            dfs.append(df)
        shielding = pd.concat(dfs, ignore_index=True)
        self.nmr_shielding = shielding

    def parse_j_coupling(self, cartesian=False):
        '''
        Parse the J Coupling matrices.

        Args:
            cartesian (:obj:`bool`, optional): Parameter to parse the
                tensors in the Cartesian axis reprentation. Set to
                `False` for the pricipal axis representation.
                Defaults to `True`.
        '''
        _recoupl = "total calculated spin-spin coupling:"
        _reatom = "Internal CPL numbering of atoms:"
        found = self.find(_reatom, keys_only=True)
        if not found:
            return
        found = self.find(_reatom, _recoupl, keys_only=True)
        # we grab the tensors inside the principal axis representation
        # for the cartesian axis representation we start the list at
        # 0 and grab every other instance
        if cartesian:
            start_coupl = found[_recoupl][::2]
        else:
            start_coupl = found[_recoupl][1::2]
        start_pert = np.array(found[_reatom]) - 3
        dfs = []
        # grab atoms
        cols = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        for ln, start in zip(start_pert, start_coupl):
            line = self[ln].split()
            # we just replace all of the () in the strings
            pert_nucl = list(map(lambda x: x.replace('(', '').replace(')', ''), line[5:]))
            nucl = list(map(lambda x: x.replace('(', '').replace(')', ''), line[1:3]))
            # grab both tensors
            df = self.pandas_dataframe(start+2, start+5, ncol=6)
            # this will grab the iso value and tensor elements for the j coupling in hz
            df.drop(range(3), axis='columns', inplace=True)
            df = pd.DataFrame(df.unstack().values.reshape(1,9), columns=cols)
            iso = self[start+1].split()[-1]
            # place all of the dataframe columns
            df['isotropic'] = float(iso)
            df['atom'] = int(nucl[0])
            df['symbol'] = nucl[1]
            df['pt_atom'] = int(pert_nucl[0])
            df['pt_symbol'] = pert_nucl[1]
            df['label'] = 'j coupling'
            df['frame'] = 0
            dfs.append(df)
        # put everything together
        j_coupling = pd.concat(dfs, ignore_index=True)
        j_coupling['atom'] -= 1
        j_coupling['pt_atom'] -= 1
        self.j_coupling = j_coupling

    def __init__(self, *args, **kwargs):
        super(NMR, self).__init__(*args, **kwargs)

class ADF(six.with_metaclass(OutMeta, Editor)):
    """The ADF output parser."""
    def parse_atom(self):
        # TODO : only supports single frame, gets last atomic positions
        #        this will actually get the very first coordinates
        #_re_atom_00 = 'Atoms in this Fragment     Cart. coord.s (Angstrom)'
        _re_atom_00 = 'ATOMS'
        found1 = self.find(_re_atom_00, keys_only=True)
        # to find the optimized frames
        _reopt = "Coordinates (Cartesian)"
        found_opt = self.find(_reopt, keys_only=True)
        if found_opt:
            starts = np.array(found_opt) + 6
            stop = starts[0]
            while '------' not in self[stop]: stop += 1
            stops = starts + stop - starts[0]
            dfs = []
            for idx, (start, stop) in enumerate(zip(starts, stops)):
                # parse everything as they may be useful in the future
                df = self.pandas_dataframe(start, stop, ncol=11)
                # drop everything
                df.drop(list(range(5, 11)), axis='columns', inplace=True)
                # we read the coordinates in bohr so no need to convrt
                df.columns = ['set', 'symbol', 'x', 'y', 'z']
                df['set'] = df['set'].astype(int)
                df['Z'] = df['symbol'].map(sym2z)
                df['frame'] = idx
                df['set'] -= 1
                dfs.append(df)
            atom = pd.concat(dfs, ignore_index=True)
        elif found1:
            start = stop = found1[-1] + 4
            while self[stop].strip(): stop += 1
            atom = self.pandas_dataframe(start, stop, ncol=8)
            atom.drop(list(range(5,8)), axis='columns', inplace=True)
            atom.columns = ['set', 'symbol', 'x', 'y', 'z']
            for c in ['x', 'y', 'z']: atom[c] *= Length['Angstrom', 'au']
            atom['Z'] = atom['symbol'].map(sym2z)
            atom['set'] -= 1
            atom['frame'] = 0
        else:
            raise NotImplementedError("We could not find the atom table in this output. Please submit "+ \
                                      "an issue ticket so we can add it in.")
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
        data['frame'] = [0]*len(data['set'])
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
        ldx = found[key][-1] + 4
        starts = []
        stops = []
        irreps = []
        while self[ldx].strip() != '':
            # error catching for when we have a symmetry label
            try:
                _ = int(self[ldx].strip()[0])
                ldx += 1
            except ValueError:
                stops.append(ldx)
                irreps.append(self[ldx])
                # to ensure that we do not skip over the blank line
                # and exdecute an infinite while loop
                if not (self[ldx].strip() == ''):
                    ldx += 1
                    starts.append(ldx)
                else:
                    break
        else:
            # to get the bottom of the table
            stops.append(ldx)
        # the first entry is actually the very beginning of the table
        stops = stops[1:]
        # put everything together
        dfs = []
        for start, stop, irrep in zip(starts, stops, irreps):
            df = self.pandas_dataframe(start, stop, cols[key])
            df['irrep'] = irrep.strip()
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
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
        _reexc = "no.     E/a.u.        E/eV      f"
        found = self.find(_reexc, keys_only=True)
        if not found:
            return
        # there should only be one in the entire output
        start = found[0] + 2
        stop = start
        while self[stop].strip(): stop += 1
        df = self.pandas_dataframe(start, stop, ncol=6)
        df.columns = ['excitation', 'energy', 'energy_ev', 'osc', 'tau', 'symmetry']
        nan = np.where(df.isna().values)[0]
        df.loc[nan, 'symmetry'] = df.loc[nan, 'tau'].copy()
        df.loc[nan, 'tau'] = 0.0
        df['tau'] = df['tau'].astype(float)
        df['excitation'] = [int(x[:-1]) for x in df['excitation'].values]
        df.index = df['excitation'] - 1
        df.drop('excitation', axis=1, inplace=True)
        df['frame'] = 0
        df['group'] = 0
        self.excitation = df


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

    def parse_gradient(self):
        _regrad = "Energy gradients wrt nuclear displacements"
        found = self.find(_regrad, keys_only=True)
        if not found:
            return
        starts = np.array(found) + 6
        stop = starts[0]
        while '----' not in self[stop]: stop += 1
        stops = starts + (stop - starts[0])
        dfs = []
        for i, (start, stop) in enumerate(zip(starts, stops)):
            df = self.pandas_dataframe(start, stop, ncol=5)
            df.columns = ['atom', 'symbol', 'fx', 'fy', 'fz']
            df['frame'] = i
            df['atom'] -= 1
            dfs.append(df)
        grad = pd.concat(dfs, ignore_index=True)
        grad['Z'] = grad['symbol'].map(sym2z)
        grad = grad[['atom', 'Z', 'fx', 'fy', 'fz', 'symbol', 'frame']]
        for u in ['fx', 'fy', 'fz']: grad[u] *= 1./Length['Angstrom', 'au']
        self.gradient = grad

    def parse_frequency(self):
        _renorm = "Vibrations and Normal Modes"
        _refreq = "List of All Frequencies:"
        found = self.find(_refreq, keys_only=True)
        if not found:
            return
        elif len(found) > 1:
            raise NotImplementedError("We cannot parse more than one frequency calculation in a single output")
        found = self.find(_refreq, _renorm, keys_only=True)
        start = found[_refreq][0] + 9
        stop = start
        while self[stop]: stop += 1
        df = self.pandas_dataframe(start, stop, ncol=3)
        freqs = df[0].values
        n = int(np.ceil(freqs.shape[0]/3))
        start = found[_renorm][0] + 9
        stop = start
        while self[stop]: stop += 1
        natoms = stop - start
        dfs = []
        fdx = 0
        for i in range(n):
            if i == 0:
                start = found[_renorm][0] + 9
            else:
                start = stop + 4
            stop = start + natoms
            freqs = list(map(float, self[start-2].split()))
            ncol = len(freqs)
            df = self.pandas_dataframe(start, stop, ncol=1+3*ncol)
            tmp = list(map(lambda x: x.split('.'), df[0]))
            index, symbol = list(map(list, zip(*tmp)))
            slices = [list(range(1+i, 1+3*ncol, 3)) for i in range(ncol)]
            dx, dy, dz = [df[i].unstack().values for i in slices]
            freqdx = np.repeat(list(range(fdx, ncol+fdx)), natoms)
            zs = pd.Series(symbol).map(sym2z)
            freqs = np.repeat(freqs, natoms)
            stacked = pd.DataFrame.from_dict({'Z': np.tile(zs, ncol), 'label': np.tile(index, ncol), 'dx': dx,
                                              'dy': dy, 'dz': dz, 'frequency': freqs, 'freqdx': freqdx})
            stacked['ir_int'] = 0.0
            stacked['symbol'] = np.tile(symbol, ncol)
            dfs.append(stacked)
            fdx += ncol
        frequency = pd.concat(dfs, ignore_index=True)
        frequency['frame'] = 0
        # TODO: check units of the normal modes
        self.frequency = frequency

    def __init__(self, *args, **kwargs):
        super(ADF, self).__init__(*args, **kwargs)

class AMS(six.with_metaclass(OutMeta, Editor)):
    """The ADF output parser for versions newer than 2019"""
    def parse_atom(self):
        # TODO: current implementation is only for the AMS driver
        _re_atom_00 = 'Atoms'
        found = self.find(_re_atom_00, keys_only=True)
        if not found:
            raise ValueError("Could not find atomic positions.")
        if len(found) == 2:
            starts = np.array(found)[:-1] + 2
        else:
            starts = np.array(found)[:-2] + 2
        stop = starts[0]
        while self[stop].strip(): stop += 1
        stops = starts + (stop - starts[0])
        dfs = []
        for idx, (start, stop) in enumerate(zip(starts, stops)):
            df = self.pandas_dataframe(start, stop, ncol=5)
            df.columns = ['set', 'symbol', 'x', 'y', 'z']
            df[['x', 'y', 'z']] *= [Length['Angstrom', 'au']]*3
            df['Z'] = df['symbol'].map(sym2z)
            df['frame'] = idx
            df['set'] -= 1
            dfs.append(df)
        atom = pd.concat(dfs, ignore_index=True)
        self.atom = atom

    def parse_frequency(self):
        _refreq = "Frequency (cm-1)"
        found = self.find(_refreq, keys_only=True)
        found = found[1:]
        if not found:
            return
        starts = np.array(found) + 2
        stop = starts[0]
        while self[stop].strip(): stop += 1
        stops = starts + (stop - starts[0])
        natoms = stop - starts[0]
        if not hasattr(self, 'atom'):
            self.parse_atom()
        if natoms != self.atom.last_frame.shape[0]:
            raise ValueError("Issue with determining the number of atoms.")
        arr = zip(found, zip(starts, stops))
        dfs = []
        for idx, (ldx, (start, stop)) in enumerate(arr):
            line = self[ldx]
            d = line.split()
            df = self.pandas_dataframe(start, stop, ncol=5)
            df.columns = ['label', 'symbol', 'dx', 'dy', 'dz']
            df['label'] -= 1
            df['frequency'] = d[4]
            df['ir_int'] = d[7]
            df['freqdx'] = idx
            df['frame'] = 0
            dfs.append(df)
        freq = pd.concat(dfs, ignore_index=True)
        self.frequency = freq

    def parse_gradient(self):
        # TODO: check if the first search string is used in all
        #       ams driver engines
        _regrad = "Energy gradients wrt nuclear displacements"
        # other gradient matrix search string
        _regradfinal = "Gradients (hartree/bohr)"
        found = self.find(_regrad, _regradfinal, keys_only=True)
        if not found[_regrad]:
            return
        if not hasattr(self, 'atom'):
            self.parse_atom()
        dfs = []
        if self.atom.nframes > 1:
            starts = np.array(found[_regrad])[:-2] + 6
            stop = starts[0]
            while '---' not in self[stop]: stop += 1
            stops = starts + (stop - starts[0])
            for idx, (start, stop) in enumerate(zip(starts, stops)):
                df = self.pandas_dataframe(start, stop, ncol=5)
                df.columns = ['atom', 'symbol', 'fx', 'fy', 'fz']
                df['Z'] = df['symbol'].map(sym2z)
                df['frame'] = idx
                df[['fx', 'fy', 'fz']] /= [Length['Angstrom', 'au']]*3
                df['atom'] -= 1
                dfs.append(df)
            frame = dfs[-1]['frame'][0]
        else:
            frame = -1
        # get the last gradient matrix as it has more sig figs
        start = found[_regradfinal][0] + 2
        stop = start
        while self[stop].strip(): stop += 1
        df = self.pandas_dataframe(start, stop, ncol=5)
        df.columns = ['atom', 'symbol', 'fx', 'fy', 'fz']
        df['Z'] = df['symbol'].map(sym2z)
        df['frame'] = frame + 1
        df['atom'] -= 1
        dfs.append(df)
        grad = pd.concat(dfs, ignore_index=True)
        self.gradient = grad

    def parse_excitation(self):
        # the output has not changed from the last few iterations
        # of ADF
        ADF.parse_excitation(self)

    def parse_electric_dipole(self):
        _reexc = "Excitation energies E in a.u. and eV, dE wrt prev. cycle"
        _retrans = "(weak excitations are not printed)"
        _revel = "(using dipole velocity formula for mu)"
        found = self.find(_reexc, _retrans, _revel, keys_only=True)
        if not found[_reexc]:
            return
        velocity = True if found[_revel] else False
        # grab the information in the main table
        # important for determining which excitations are
        # printed later on
        start = found[_reexc][0] + 5
        stop = start
        while self[stop].strip(): stop += 1
        df = self.pandas_dataframe(start, stop, ncol=5)
        df.drop(4, axis=1, inplace=True)
        df.columns = ['excitation', 'energy_au', 'energy_ev', 'osc']
        df['excitation'] -= 1
        # grab the information with the transition dipole moments
        start = found[_retrans][0] + 4
        stop = start
        while self[stop].strip(): stop += 1
        nrows = stop - start
        tdm = self.pandas_dataframe(start, stop, ncol=10)
        tdm.dropna(axis=1, how='all', inplace=True)
        if tdm.shape[1] == 9:
            # TODO: should we get the magnitude here instead?
            tdm.columns = ['excitation', 'energy_ev', 'osc', 'remu_x' ,'remu_y', 'remu_z',
                           'immu_x', 'immu_y', 'immu_z']
        else:
            tdm.columns = ['excitation', 'energy_ev', 'osc', 'mu_x' ,'mu_y', 'mu_z']
        tdm['excitation'] -= 1
        diff = np.setdiff1d(df['excitation'].values.flatten(),
                            tdm['excitation'].values.flatten())
        if len(diff) > 0:
            for d in diff:
                df1 = tdm.loc[range(d)]
                df2 = tdm.loc[range(d, tdm.shape[0])]
                new_line = [d, df.loc[d, 'energy_ev'],
                            df.loc[d, 'osc']]+[0.0]*int(len(tdm.columns)-3)
                new_line = pd.DataFrame([new_line], columns=tdm.columns)
                new_df = pd.concat([df1, new_line, df2], ignore_index=True)
                tdm = new_df.copy()
        self.electric_dipole = tdm

    def parse_magnetic_dipole(self):
        _retrans = "magnetic transition dipole vectors m in a.u."
        _revel = "(using dipole velocity formula for mu)"
        found = self.find(_retrans, _revel, keys_only=True)
        if not found[_retrans]:
            return
        velocity = True if found[_revel] else False
        # grab the information with the transition dipole moments
        start = found[_retrans][0] + 4
        stop = start
        while self[stop].strip(): stop += 1
        tdm = self.pandas_dataframe(start, stop, ncol=10)
        tdm.dropna(axis=1, how='all', inplace=True)
        if tdm.shape[1] == 8:
            spinorbit=True
            tdm.columns = ['excitation', 'rot', 'rem_x' ,'rem_y', 'rem_z',
                           'imm_x', 'imm_y', 'imm_z']
        else:
            tdm.columns = ['excitation', 'energy_ev', 'osc', 'mu_x' ,'mu_y', 'mu_z']
            spinorbit=False
        tdm['excitation'] -= 1
        self.magnetic_dipole = tdm

    def __init__(self, *args, **kwargs):
        super(AMS, self).__init__(*args, **kwargs)

class Output(six.with_metaclass(OutMeta, Editor)):
    def _parse_data(self, attr, **kwargs):
        try:
            parser = getattr(self.meta['module'], attr)
        except AttributeError as e:
            if str(e).endswith("'"+attr+"'"):
                return
        parser(self, **kwargs)

    def parse_atom(self, **kwargs):
        self._parse_data('parse_atom', **kwargs)

    def parse_basis_set(self, **kwargs):
        self._parse_data('parse_basis_set', **kwargs)

    def parse_basis_set_order(self, **kwargs):
        self._parse_data('parse_basis_set_order', **kwargs)

    def parse_orbital(self, **kwargs):
        self._parse_data('parse_orbital', **kwargs)

    def parse_contribution(self, **kwargs):
        self._parse_data('parse_contribution', **kwargs)

    def parse_excitation(self, **kwargs):
        self._parse_data('parse_excitation', **kwargs)

    def parse_momatrix(self, **kwargs):
        self._parse_data('parse_momatrix', **kwargs)

    def parse_sphr_momatrix(self, **kwargs):
        self._parse_data('parse_sphr_momatrix', **kwargs)

    def parse_gradient(self, **kwargs):
        self._parse_data('parse_gradient', **kwargs)

    def parse_frequency(self, **kwargs):
        self._parse_data('parse_frequency', **kwargs)

    def parse_nmr_shielding(self, **kwargs):
        self._parse_data('parse_nmr_shielding', **kwargs)

    def parse_j_coupling(self, **kwargs):
        self._parse_data('parse_j_coupling', **kwargs)

    def parse_electric_dipole(self, **kwargs):
        self._parse_data('parse_electric_dipole', **kwargs)

    def parse_magnetic_dipole(self, **kwargs):
        self._parse_data('parse_magnetic_dipole', **kwargs)

    def __init__(self, file, *args, **kwargs):
        ed = _Editor(file)
        _readf = "A D F"
        _renmr = "N M R"
        _reams = "A M S"
        found = ed.find(_readf, _renmr, _reams, keys_only=True)
        super(Output, self).__init__(file, *args, **kwargs)
        if found[_renmr]:
            self.meta.update({'module': NMR})
        elif found[_readf]:
            self.meta.update({'module': ADF})
        elif found[_reams]:
            self.meta.update({'module': AMS})
        else:
            raise ValueError("Could not identify program from SCM")

