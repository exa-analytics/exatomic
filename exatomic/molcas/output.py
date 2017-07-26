# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Molcas Output Parser
#####################
Multiple frames are not currently supported
'''
import os
import pandas as pd
import numpy as np
from io import StringIO

from .editor import Editor

from exatomic.basis import Overlap, lmap, rlmap, spher_lml_count
from exatomic.orbital import DensityMatrix
from exa.relational.isotope import symbol_to_z

symbol_to_z = symbol_to_z()

class Orb(Editor):

    def to_universe(self):
        raise NotImplementedError("No atom information given. " \
                                  "Attach these attributes to a universe.")

    def _one_el(self, starts, step, ncol):
        func = pd.read_csv
        kwargs = {'header': None}
        if ncol == 1:
            func = pd.read_fwf
            kwargs['widths'] = [18] * 4
        else:
            kwargs['delim_whitespace'] = True
        return [func(StringIO('\n'.join(self[start:start + step])),
                     **kwargs).stack().values for start in starts]

    def parse_momatrix(self):
        dim = int(self[5])
        ndim = dim * dim
        found = self.find(_re_orb, _re_occ,
                          _re_ens, keys_only=True)
        skips = found[_re_orb]
        start = skips[0]
        occs = [i + 1 for i in found[_re_occ]]
        ens = [i + 1 for i in found[_re_ens]]
        if not found[_re_ens]: ens = False
        ncol = len(self[start + 1].split())
        cols = 4 if ncol == 1 else ncol
        chnk = np.ceil(dim / cols).astype(np.int64)
        orbdx = np.repeat(range(dim), chnk)
        if len(occs) == 2:
            skips.insert(dim, skips[dim] - 1)
            orbdx = np.concatenate([orbdx, orbdx])
        skips = [i - skips[0] for i in skips]
        if ncol == 1:
            coefs = pd.read_fwf(StringIO('\n'.join(self[start:occs[0]-2])),
                                skiprows=skips, header=None, widths=[18]*4)
            if ens: ens = self._one_el(ens, chnk, ncol)
        else:
            coefs = self.pandas_dataframe(start, occs[0]-2, ncol,
                                          **{'skiprows': skips})
            if ens:
                echnk = np.ceil(dim / len(self[ens[0] + 1].split())).astype(np.int64)
                ens = self._one_el(ens, echnk, ncol)
        occs = self._one_el(occs, chnk, ncol)
        coefs['idx'] = orbdx
        coefs = coefs.groupby('idx').apply(pd.DataFrame.stack).drop(
                                           'idx', level=2).values
        mo = {'orbital': np.repeat(range(dim), dim), 'frame': 0,
              'chi': np.tile(range(dim), dim)}
        if ens:
            orb = {'frame': 0, 'group': 0}
        if len(occs) == 2:
            mo['coef'] = coefs[:len(coefs)//2]
            mo['coef1'] = coefs[len(coefs)//2:]
            self.occupation_vector = {'coef': occs[0], 'coef1': occs[1]}
            if ens:
                orb['occupation'] = np.concatenate(occs)
                orb['energy'] = np.concatenate(ens)
                orb['vector'] = np.concatenate([range(dim), range(dim)])
                orb['spin'] = np.concatenate([np.zeros(dim), np.ones(dim)])
        else:
            mo['coef'] = coefs
            self.occupation_vector = occs[0]
            if ens:
                orb['occupation'] = occs[0]
                orb['energy'] = ens[0]
                orb['vector'] = range(dim)
                orb['spin'] = np.zeros(dim)
        self.momatrix = pd.DataFrame.from_dict(mo)
        if ens:
            self.orbital = pd.DataFrame.from_dict(orb)

    def __init__(self, *args, **kwargs):
        super(Orb, self).__init__(*args, **kwargs)


_re_orb = 'ORBITAL'
_re_occ = 'OCCUPATION NUMBERS'
_re_ens = 'ONE ELECTRON ENERGIES'


class Output(Editor):

    def parse_atom(self):
        '''Parses the atom list generated in SEWARD.'''
        start = stop = self.find(_re_atom, keys_only=True)[0] + 8
        while self[stop].split(): stop += 1
        columns = ['label', 'tag', 'x', 'y', 'z', 5, 6, 7]
        atom = self.pandas_dataframe(start, stop, columns).drop([5, 6, 7], axis=1)
        atom['symbol'] = atom['tag'].str.extract('([A-z]{1,})([0-9]*)',
                                                 expand=False)[0].str.lower().str.title()
        atom['Z'] = atom['symbol'].map(symbol_to_z).astype(np.int64)
        atom['label'] -= 1
        atom['frame'] = 0
        self.atom = atom

    def parse_basis_set_order(self):
        '''
        Parses the shell ordering scheme if BSSHOW specified in SEWARD.
        '''
        start = stop = self.find(_re_bas_order, keys_only=True)[0] + 1
        while self[stop].strip(): stop += 1
        df = self.pandas_dataframe(start, stop, ['idx', 'tag', 'type', 'center'])
        df.drop(['idx', 'tag'], inplace=True, axis=1)
        if 'set' not in self.atom.columns: self.parse_basis_set()
        mldict = {'': 0, 'x': 1, 'y': -1, 'z': 0}
        df['center'] -= 1
        df['n'] = df['type'].str[0]
        df['n'].update(df['n'].map({'*': 0}))
        df['n'] = df['n'].astype(np.int64)
        fill = df['n'] + 1
        fill.index += 1
        df.loc[df[df['n'] == 0].index, 'n'] = fill
        df['L'] = df['type'].str[1].map(lmap)
        df['ml'] = df['type'].str[2:]
        df['ml'].update(df['ml'].map(mldict))
        df['ml'].update(df['ml'].str[::-1])
        df['ml'] = df['ml'].astype(np.int64)
        funcs = self.basis_set.functions_by_shell()
        shells = []
        for seht in self.atom['set']:
            tot = 0
            lml_count = spher_lml_count
            for l, n in funcs[seht].items():
                for i in range(lml_count[l]):
                    shells += list(range(tot, n + tot))
                tot += n
        df['shell'] = shells
        df['frame'] = 0
        self.basis_set_order = df


    def parse_basis_set(self):
        '''
        Parses the primitive exponents, coefficients and shell if BSSHOW specified in SEWARD.
        '''
        found = self.find(_re_bas_0, _re_bas_1, _re_bas_2, keys_only=True)
        bmaps = [i + 1 for i in found[_re_bas_0]]
        atoms = [i + 2 for i in found[_re_bas_1]]
        alphs = [i + 1 for i in found[_re_bas_2]]
        widths = [11, 7, 8, 11, 10, 12]
        names = _re_bas_0.split()
        setmap, basmap = {}, []
        for seht, (start, atst) in enumerate(zip(bmaps, atoms)):
            stop = start
            while self[stop].strip(): stop += 1
            while self[atst].strip():
                setmap[self[atst].split()[0]] = seht
                atst += 1
            basmap.append(pd.read_fwf(StringIO('\n'.join(self[start:stop])),
                                      widths=widths, header=None, names=names))
            basmap[-1]['set'] = seht
        self.atom['set'] = self.atom['tag'].map(setmap)
        basmap = pd.concat(basmap).reset_index(drop=True)
        basmap['Shell'] = basmap['Shell'].map(lmap)
        prims, pset, shell = [], 0, 0
        for start, seht, L, nprim, nbas in zip(alphs, basmap['set'], basmap['Shell'],
                                               basmap['nPrim'], basmap['nBasis']):
            if pset != seht: shell = 0
            # In case contraction coefficients overflow to next line
            neat = len(self[start].split()) == len(self[start + 1].split())
            if neat: block = self.pandas_dataframe(start, start + nprim, nbas + 2)
            else:
                stop = start + 2 * nprim
                most = self[start:stop:2]
                extr = self[start + 1:stop:2]
                ncols = len(most[0].split()) + len(extr[0].split())
                block = pd.read_csv(StringIO('\n'.join([i + j for i, j in zip(most, extr)])),
                                    delim_whitespace=True, names=range(ncols))
            alps = (pd.concat([block[1]] * nbas).reset_index(drop=True)
                    .str.replace('D', 'E').astype(np.float64))
            ds = block[list(range(2, nbas + 2))].unstack().reset_index(drop=True)
            pdf = pd.concat([alps, ds], axis=1)
            pdf.columns = ['alpha', 'd']
            pdf['L'] = L
            pdf['shell'] = np.repeat(range(shell, shell + nbas), nprim)
            pdf['set'] = seht
            prims.append(pdf)
            shell += nbas
            pset = seht
        prims = pd.concat(prims).reset_index(drop=True)
        prims['frame'] = 0
        self.basis_set = prims

    def __init__(self, *args, **kwargs):
        super(Output, self).__init__(*args, **kwargs)


_re_atom = 'Molecular structure info'
_re_bas_order = 'Basis Label        Type   Center'
_re_bas_0 = 'Shell  nPrim  nBasis  Cartesian Spherical Contaminant'
_re_bas_1 = 'Label   Cartesian Coordinates / Bohr'
_re_bas_2 = 'No.      Exponent    Contraction Coefficients'


def parse_molcas(fp, momatrix=None, overlap=None, occvec=None, **kwargs):
    """
    Will parse a Molcas output file. Optionally it will attempt
    to parse additional information obtained from the same directory
    from specified Orb files or the AO overlap matrix and density matrix.
    If density keyword is specified, the momatrix keyword is ignored.

    Args
        fp (str): Path to output file
        momatrix (str): file name of the C matrix of interest
        overlap (str): file name of the overlap matrix
        occvec (str): an occupation vector

    Returns
        parsed (Editor): contains many attributes similar to the
            exatomic universe
    """
    uni = Output(fp, **kwargs)
    adir = os.sep.join(fp.split(os.sep)[:-1])
    if momatrix is not None:
        fp = os.sep.join([adir, momatrix])
        if os.path.isfile(fp):
            orb = Orb(fp)
            uni.momatrix = orb.momatrix
            uni.occupation_vector = orb.occupation_vector
            occvec = occvec if occvec is not None else orb.occupation_vector
            d = DensityMatrix.from_momatrix(orb.momatrix, occvec)
            uni.density = d
        else:
            print('Is {} in the same directory as {}?'.format(momatrix, fp))
    if overlap is not None:
        fp = os.sep.join([adir, overlap])
        if os.path.isfile(fp): uni.overlap = Overlap.from_file(fp)
        else: print('Is {} in the same directory as {}?'.format(overlap, fp))
    return uni
