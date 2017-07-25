# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NWChem Output
#######################
Parse NWChem output files and convert them into an exatomic Universe container.
"""
import re
from os import sep, path
import numpy as np
import pandas as pd
from io import StringIO
from exatomic import Length
from exatomic.atom import Atom
from exatomic.frame import compute_frame_from_atom
from exatomic.basis import BasisSet, lmap
from exatomic.orbital import Orbital, MOMatrix
from .editor import Editor
from .basis import cartesian_ordering_function, spherical_ordering_function


class Output(Editor):
    """
    Editor for NWChem calculation output file (stdout).
    """
    def parse_atom(self):
        """
        Parse the atom dataframe.
        """
        found = self.find(_reatom01, _reatom02,
                          _reatom03, _reatom04, keys_only=True)
        unit = self[found[_reatom04][0]].split()[3]
        starts = np.array(found[_reatom01]) + 7
        stops = np.array(found[_reatom02]) - 1
        ecps = np.array(found[_reatom03]) + 2
        ecps = {self[ln].split()[0]: int(self[ln].split()[3]) for ln in ecps}
        columns = ['label', 'tag', 'Z', 'x', 'y', 'z']
        atom = pd.concat([self.pandas_dataframe(s, e, columns)
                          for s, e in zip(starts, stops)])
        atom['symbol'] = atom['tag'].str.extract('([A-z]{1,})([0-9]*)',
                                                 expand=False)[0].str.lower().str.title()
        atom['Z'] = atom['Z'].astype(np.int64)
        atom['Zeff'] = (atom['Z'] - atom['tag'].map(ecps).fillna(value=0)).astype(np.int64)
        n = len(atom)
        nf = atom.label.value_counts().max()
        nat = atom.label.max()
        basis_sets = np.empty((n, ), dtype=np.int64)
        atom['frame'] = [i for i in range(nf) for j in range(nat)]
        atom['label'] -= 1
        atom['x'] *= Length[unit, 'au']
        atom['y'] *= Length[unit, 'au']
        atom['z'] *= Length[unit, 'au']
        if atom['frame'].max() > 0:
            li = atom['frame'].max()
            atom = atom[~(atom['frame'] == li)]
            atom.reset_index(drop=True, inplace=True)
        del atom['label']
        self.atom = atom

    def parse_orbital(self):
        '''
        Parse the :class:`~exatomic.orbital.Orbital` dataframe.
        '''
        orbital = None
        check = self.find(_remo01)
        if any(['Alpha' in value for value in check.values()]):
            alpha_starts = np.array([no for no, line in check if 'Alpha' in line], dtype=np.int64) + 2
            alpha_stops = np.array([no for no, line in check if 'Beta' in line], dtype=np.int64) - 1
            beta_starts = alpha_stops + 3
            beta_stops = np.array(self.find(_remo02, keys_only=True), dtype=np.int64) - 1
            alpha_orbital = self._parse_orbital(alpha_starts, alpha_stops)
            beta_orbital = self._parse_orbital(beta_starts, beta_stops)
            alpha_orbital['spin'] = 0
            beta_orbital['spin'] = 1
            orbital = pd.concat((alpha_orbital, beta_orbital), ignore_index=True)
        else:
            starts = np.array(list(zip(*check))[0], dtype=np.int64) + 2
            stops = np.array(self.find(_remo03, keys_only=True), dtype=np.int64) - 1
            orbital = self._parse_orbital(starts, stops)
            orbital['spin'] = 0
        orbital['group'] = 0
        self.orbital = Orbital(orbital)

    def parse_momatrix(self):
        """
        Parse the :class:`~exatomic.orbital.MOMatrix` dataframe.

        Note:
            Must supply 'print "final vectors" "final vectors analysis" for momatrix
        """
        found = self.find(_reallmos, keys_only=True)
        if found:
            nrcol = 6
            start = found[0] + 8
            nfuncs = self.gaussian_basis_set.functions()
            nbas = self.atom['set'].map(nfuncs).sum()
            chunk = nbas + 3
            nchunks = np.ceil(nbas / nrcol).astype(np.int64)
            leftover = nbas % nrcol
            idxs = [(start + chunk * i, start + nbas + chunk * i) for i in range(nchunks)]
            print(idxs)
            dfs = []
            for i, (start, stop) in enumerate(idxs):
                if i == len(idxs) - 1 and leftover:
                    dfs.append(self.pandas_dataframe(start, stop, leftover))
                else:
                    dfs.append(self.pandas_dataframe(start, stop, nrcol))
            coefs = pd.concat(dfs, axis=1).unstack().values
            chis = np.tile(range(nbas), nbas)
            orbitals = np.repeat(range(nbas), nbas)
            self.momatrix = pd.DataFrame.from_dict({'coef': coefs,
                                                    'orbital': orbitals,
                                                    'chi': chis,
                                                    'frame': [0] * nbas ** 2})


    def _parse_orbital(self, starts, stops):
        '''
        This function actually performs parsing of :class:`~exatomic.orbital.Orbital`

        See Also:
            :func:`~exnwchem.output.Output.parse_orbital`
        '''
        joined = '\n'.join(['\n'.join(self[s:e]) for s, e in zip(starts, stops)])
        nvec = joined.count('Vector')
        nbas = self.atom['symbol'].value_counts().values * self.basis_set_summary['func_per_atom'].astype(np.int64)
        nbas = nbas.sum()
        nbas *= nvec
        # Orbital dataframe -- alternatively one could parse the strings
        # into the DataFrame and then use the pd.Series.str methods to
        # perform all the replacements at the same time, eg. 'D' --> 'E'
        # and 'Occ=' --> '', etc.
        orb_no = np.empty((nvec, ), dtype=np.int64)
        occ = np.empty((nvec, ), dtype=np.float64)
        nrg = np.empty((nvec, ), dtype=np.float64)
        x = np.empty((nvec, ), dtype=np.float64)
        y = np.empty((nvec, ), dtype=np.float64)
        z = np.empty((nvec, ), dtype=np.float64)
        frame = np.empty((nvec, ), dtype=np.int64)
        fc = -1   # Frame counter
        oc = 0   # Orbital counter
        for s, e in zip(starts, stops):
            fc += 1
            for line in self[s:e]:
                ls = line.split()
                if 'Vector' in line:
                    orb_no[oc] = ls[1]
                    occ[oc] = ls[2].replace('Occ=', '').replace('D', 'E')
                    nrg[oc] = ls[3].replace('E=', '').replace('D', 'E') if 'E=-' in line else ls[4].replace('D', 'E')
                    frame[oc] = fc
                elif 'MO Center' in line:
                    x[oc] = ls[2].replace(',', '').replace('D', 'E')
                    y[oc] = ls[3].replace(',', '').replace('D', 'E')
                    z[oc] = ls[4].replace(',', '').replace('D', 'E')
                    oc += 1
        orb_no -= 1
        return pd.DataFrame.from_dict({'x': x, 'y': z, 'z': z, 'frame': frame,
                                       'vector': orb_no, 'occupation': occ, 'energy': nrg})

    def parse_gaussian_basis_set(self):
        """
        Parse the :class:`~exatomic.basis.BasisSet` dataframe.
        """
        found = self.find(_rebas01, _rebas02)
        spherical = True if 'spherical' in found[_rebas01][0][1] else False
        start = found[_rebas01][0][0]
        stop = found[_rebas02][0][0] - 1 if start < found[_rebas02][0][0] else found[_rebas02][1][0] - 1
        df = self.pandas_dataframe(start, stop, 4, **{'index_col': False})
        dtype = [('shell', 'i8'), ('L', 'i8'), ('alpha', 'f8'),
                 ('d', 'f8'), ('set', 'i8')]
        keep = np.empty((df.shape[0],), dtype=dtype)
        seht, cnt = -1, 0
        tags = {}
        print(df)
        for i, (shell, l, alpha, d) in enumerate(zip(df[0], df[1], df[2], df[3])):
            print(d)
            if len(shell) == 8:
                seht += 1
                tags[df[0].values[i - 2]] = seht
            if not np.isnan(d):
                keep[cnt] = (shell, lmap[l.lower()], alpha, d, seht)
                cnt += 1
        gaussian_basis_set = pd.DataFrame(keep[:cnt])
        gaussian_basis_set['shell'] -= 1
        gaussian_basis_set['frame'] = 0
        self.gaussian_basis_set = gaussian_basis_set
        self.gaussian_basis_set.spherical = spherical
        self.atom['set'] = self.atom['tag'].map(tags)

    def parse_basis_set_order(self):
        dtype = [('center', 'i8'), ('shell', 'i8'), ('L', 'i8')]
        if self.gaussian_basis_set.spherical:
            dtype += [('ml', 'i8')]
        else:
            dtype += [('l', 'i8'), ('m', 'i8'), ('n', 'i8')]
        nbas = self.atom['set'].map(self.gaussian_basis_set.functions()).sum()
        bso = np.empty((nbas,), dtype=dtype)
        cnt = 0
        bases = self.gaussian_basis_set.groupby('set')
        for seht, center in zip(self.atom['set'], self.atom.index):
            bas = bases.get_group(seht).groupby('shell')
            if self.gaussian_basis_set.spherical:
                for shell, grp in bas:
                    l = grp['L'].values[0]
                    for ml in spherical_ordering_function(l):
                        bso[cnt] = (center, shell, l, ml)
                        cnt += 1
            else:
                for shell, grp in bas:
                    l = grp['L'].values[0]
                    for L, ll, m, n in cartesian_ordering_function(l):
                        bso[cnt] = (center, shell, l, ll, m, n)
                        cnt += 1
        self.basis_set_order = pd.DataFrame(bso)

    def parse_frame(self):
        """
        Create a minimal :class:`~exatomic.frame.Frame` from the (parsed)
        :class:`~exatomic.atom.Atom` object.
        """
        self.frame = compute_frame_from_atom(self.atom)
        found = self.find(_rescfen, _redften)
        scfs = found[_rescfen]
        dfts = found[_redften]
        if scfs and dfts:
            print('Warning: found total energies from scf and dft, using dft')
            dfts = [float(val.split()[-1]) for key, val in dfts.items()]
            self.frame['total_energy'] = dfts
        elif scfs:
            scfs = [float(val.split()[-1]) for key, val in scfs.items()]
            self.frame['total_energy'] = scfs
        elif dfts:
            dfts = [float(val.split()[-1]) for key, val in dfts.items()]
            self.frame['total_energy'] = dfts


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# Some parsing strings for NWChem outputs
_reatom01 = 'Geometry "'
_reatom02 = 'Atomic Mass'
_reatom03 = 'ECP       "ecp basis"'
_reatom04 = 'Output coordinates in'
_rebas01 = ' Basis "'
_rebas02 = ' Summary of "'
_rebas03 = [' s ', ' px ', ' py ', ' pz ', ' d ', ' f ', ' g ', ' h ', ' i ',
            ' j ', ' k ', ' l ', ' m ', ' p ']
_remo01 = 'Molecular Orbital Analysis'
_remo02 = 'alpha - beta orbital overlaps'
_remo03 = 'center of mass'
#_reovl01 = ' Begin overlap 1-e integrals'
#_reovl02 = ' End overlap 1-e integrals'
_rescfen = 'Total SCF energy'
_redften = 'Total DFT energy'
_reallmos = 'Final MO vectors'


class Ecce(Editor):
    def _parse_movecs(self, start, stop):
        ndim = int(self[start].split('%')[3].split()[0])
        vals = []
        small = []
        for line in self[start+1:stop]:
            ll = line.split()
            cnt = 0
            for l in ll:
                try:
                    small.append(float(l))
                    if len(small) == ndim:
                        vals.append(small)
                        small = []
                except:
                    try:
                        num, val = l.split('*')
                        num = int(num)
                        val = float(val)
                        for i in range(num):
                            small.append(val)
                            if len(small) == ndim:
                                vals.append(small)
                                small = []
                    except:
                        print('something went wrong parsing ecce movecs')
                        return
        vals = [chi for mo in vals for chi in mo]
        vals = pd.DataFrame(vals)
        vals.columns = ['coef']
        vals['chi'] = np.tile(list(range(ndim)), ndim)
        vals['orbital'] = np.repeat(list(range(ndim)), ndim)
        vals['frame'] = 0
        self.momatrix = vals

    def _parse_occupations(self):
        bb = list(self._regex[self._rebmooccs].keys())
        b = bb[0] + 1
        ee = list(self._regex[self._reemooccs].keys())
        if len(bb) > 1:
            print('ambiguous orbital selection: are these the orbitals you are looking for?')
        e = ee[0]
        occs = self.pandas_dataframe(b, e, 4).stack().values
        self.occupations = occs

    def parse_momatrix(self):
        #try:
        b = list(self._regex[self._rebmovecs].keys())[0]
        e = list(self._regex[self._reemovecs].keys())[0]
        self._parse_movecs(b, e)
        self._parse_occupations()
        #except:
        #    print(self._regex)
        #    print('did not parse momatrix with kind={} and spin={}'.format(self._kind, self._spin))

    def parse(self):
        if self._kind is not None:
            if self._spin is not None:
                self._rebmovecs = r'.*' + self._kind + '%begin%molecular orbital vectors.*' + self._spin
                self._reemovecs = r'.*' + self._kind + '%end%molecular orbital vectors.*' + self._spin
                self._rebmooccs = r'.*' + self._kind + '%begin%molecular orbital occupations.*' + self._spin
                self._reemooccs = r'.*' + self._kind + '%end%molecular orbital occupations.*' + self._spin
            else:
                self._rebmovecs = r'.*' + self._kind + '%begin%molecular orbital vectors'
                self._reemovecs = r'.*' + self._kind + '%end%molecular orbital vectors'
                self._rebmooccs = r'.*' + self._kind + '%begin%molecular orbital occupations'
                self._reemooccs = r'.*' + self._kind + '%end%molecular orbital occupations'
        else:
            #if self._spin is not None:
            #    try:
            #else:
                try:
                    self._rebmovecs = r'.*%begin%molecular orbital vectors'
                    self._regex = self.regex(self._rebmovecs)
                    b = list(self._regex[self._rebmovecs].keys())[0]
                    self._reemovecs = r'.*%end%molecular orbital vectors'
                    self._rebmooccs = r'.*%begin%molecular orbital occupations'
                    self._reemooccs = r'.*%end%molecular orbital occupations'
                except IndexError:
                    print('could not find which movecs to parse, try specifying kind and/or spin')
        #if self._kind == 'scf':
        #    if self._spin is None:
        #        self._rebmovecs = r'.*scf%begin%molecular orbital vectors'
        #        self._reemovecs = r'.*scf%end%molecular orbital vectors'
        #        self._rebmooccs = r'.*scf%begin%molecular orbital occupations'
        #        self._reemooccs = r'.*scf%end%molecular orbital occupations'
        #    else:
        #        self._rebmovecs = r'.*scf%begin%molecular orbital vectors UHF ' + self._spin
        #        self._reemovecs = r'.*scf%end%molecular orbital vectors UHF ' + self._spin
        #        self._rebmooccs = r'.*scf%begin%molecular orbital occupations UHF ' + self._spin
        #        self._reemooccs = r'.*scf%end%molecular orbital occupations UHF ' + self._spin
        #elif self._kind == 'dft':
        #    if self._spin is None:
        #        self._rebmovecs = r'.*dft%begin%molecular orbital vectors'
        #        self._reemovecs = r'.*dft%end%molecular orbital vectors'
        #        self._rebmooccs = r'.*dft%begin%molecular orbital occupations'
        #        self._reemooccs = r'.*dft%end%molecular orbital occupations'
        #    else:
        #        self._rebmovecs = r'.*dft%begin%molecular orbital vectors dft ' + self._spin
        #        self._reemovecs = r'.*dft%end%molecular orbital vectors dft ' + self._spin
        #        self._rebmooccs = r'.*dft%begin%molecular orbital occupations dft ' + self._spin
        #        self._reemooccs = r'.*dft%end%molecular orbital occupations dft ' + self._spin

        self._regex = self.regex(self._rebmovecs, self._reemovecs,
                                 self._rebmooccs, self._reemooccs)
        self.parse_momatrix()


    def __init__(self, *args, kind=None, spin=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._kind = kind
        self._spin = spin
        self.parse()


def parse_nwchem(file_path, ecce=None, kind='scf'):
    """
    Will parse an NWChem output file. Optionally it will attempt to parse
    an ECCE (extensible computational chemistry environment) output containing
    the C matrix to be used in visualization of molecular orbitals. The kind
    parameter chooses the 'scf' or 'dft' C matrix in the ecce output.

    Args
        file_path (str): file path to the output file
        ecce (str): name of the ecce output in the same directory

    Returns
        parsed (Editor): contains many attributes similar to the
                         exatomic Universe
    """
    uni = Output(file_path)
    dirtree = sep.join(file_path.split(sep)[:-1])
    if ecce is not None:
        fp = sep.join([dirtree, ecce])
        if path.isfile(fp):
            momat = Ecce(fp, kind=kind)
            uni.momatrix = momat.momatrix
        else:
            print('Is {} in the same directory as {}?'.format(ecce, file_path))
    return uni
