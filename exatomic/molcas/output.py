# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Output Parser
#####################
Multiple frames are not currently supported
'''
import re
import os
from os import sep
import pandas as pd
import numpy as np
from io import StringIO

from .editor import MolcasEditor

from exatomic.atom import Atom
from exatomic.basis import GaussianBasisSet, BasisSetOrder, Overlap
from exatomic.orbital import MOMatrix, DensityMatrix
from exatomic.algorithms.basis import (cart_ml_count, spher_ml_count, lmap, lorder,
                                       spher_lml_count)
from exatomic import Isotope
from exa.relational.isotope import symbol_to_z, z_to_symbol

symbol_to_Z = symbol_to_z()
Z_to_symbol = z_to_symbol()
rlmap = {value: key for key, value in lmap.items()}

class Base(MolcasEditor):

    def _parse_momatrix(self):
        dim = int(self[5])
        found = self.find(_orb_orb, _orb_occ, keys_only=True)
        ndim = dim * dim
        orbstarts = np.array(found[_orb_orb]) + 1
        occstart = found[_orb_occ][0] + 1
        nrcol = len(self[orbstarts[0]].split())
        nrcolocc = len(self[occstart].split())
        coefs = np.empty(ndim, dtype=np.float64)
        occvec = np.empty(dim, dtype=np.float64)
        if nrcol == 1:
            orbstops = np.ceil(orbstarts + dim / 4).astype(int)
            occstop = np.ceil(occstart + dim / 4).astype(int)
            for i, (start, stop) in enumerate(zip(orbstarts, orbstops)):
                tmp = [[ln[chnk] for chnk in _orb_slice if ln[chnk]] for ln in self[start:stop]]
                coefs[i*dim:i*dim + dim] = pd.DataFrame(tmp).stack().values
            tmp = [[ln[chnk] for chnk in _orb_slice if ln[chnk]] for ln in self[occstart:occstop]]
            occvec[:] = pd.DataFrame(tmp).stack().values
        else:
            orbstops = np.ceil(orbstarts + dim / nrcol).astype(int)
            occstop = np.ceil(occstart + dim / nrcolocc).astype(int)
            for i, (start, stop) in enumerate(zip(orbstarts, orbstops)):
                coefs[i*dim:i*dim + dim] = self.pandas_dataframe(start, stop, nrcol).stack().values
            occvec[:] = self.pandas_dataframe(occstart, occstop, nrcolocc).stack().values
        momatrix = pd.DataFrame.from_dict({'coefficient': coefs,
                                           'orbital': np.repeat(range(dim), dim),
                                           'chi': np.tile(range(dim), dim),
                                           'frame': 0})
        self.momatrix = momatrix
        self.occupation_vector = occvec

class Grid(Base):

    def parse_atom(self):
        fidx, nat = self.find(_re_grid_nat, keys_only=True)[0]
        nat = int(nat.split()[-1])
        atom = self.pandas_dataframe(fidx, fidx + nat, 4)
        atom['frame'] = 0
        atom.columns = ['symbol', 'x', 'y', 'z', 'frame']
        self._atom = Atom(atom)

    def parse_orbital(self):
        self.orbital = None

    def parse_momatrix(self):
        self._parse_momatrix()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parse()

# Grid based regex
_re_grid_nat = 'Natom'
_re_grid_matdim = 'N_of_MO'
_re_grid_ens = 'ONE ELECTRON ENERGIES'
_re_grid_idx = r'#INDEX'

# Works for both
_orb_orb = 'ORBITAL'
_orb_occ = 'OCCUPATION NUMBERS'

class Orb(Base):

    def parse_momatrix(self):
        self._parse_momatrix()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Output(MolcasEditor):

    def parse_atom(self):
        '''
        Parses the atom list generated in SEWARD.
        '''
        start = self.find(_re_atom, keys_only=True)[0] + 8
        stop = self._find_break(start, finds=['****', '--'])
        atom = self.pandas_dataframe(start, stop, 8)
        atom.drop([5, 6, 7], axis=1, inplace=True)
        atom.columns = ['label', 'tag', 'x', 'y', 'z']
        atom['symbol'] = atom['tag'].str.extract('([A-z]{1,})([0-9]*)',
                                                 expand=False)[0].str.lower().str.title()
        atom['Z'] = atom['symbol'].map(symbol_to_Z).astype(np.int64)
        atom['label'] -= 1
        atom['frame'] = 0
        self.atom = atom

    def parse_basis_set_order(self):
        '''
        Parses the shell ordering scheme if BSSHOW specified in SEWARD.
        '''
        start = self.find(_re_bas_order, keys_only=True)[0] + 1
        stop = self._find_break(start)
        basis_set_order = self.pandas_dataframe(start, stop, 4)
        basis_set_order.drop(0, 1, inplace=True)
        basis_set_order.columns = ['tag', 'type', 'center']
        shls = self.gaussian_basis_set.nshells()
        sets = self.atom['set']
        funcs = self.gaussian_basis_set.functions_by_shell()
        self.basis_set_order = _fix_basis_set_order(basis_set_order, shls, sets, funcs)

    def _basis_set_map(self):
        '''
        Breaks if there is anything in Contaminant column regarding basis sets.
        May only work for ANO-RCC basis sets.
        '''
        regex = self.regex(_re_bas_names01, _re_bas_dims, _re_bas_names02)
        names = []
        for i, (key, val) in enumerate(regex[_re_bas_names01]):
            try:
                key2, val2 = regex[_re_bas_names02][i]
            except IndexError: # In case second regex is not found
                key2 = key
                val2 = val
            try:
                tmp = val.split(':')[1].split('.')
                names.append([tmp[0].strip(), tmp[1].strip(), tmp[-2].strip()])
            except IndexError: # In case first regex is not as expected
                tmp = val2.split(':')[1].split('.')
                names.append([tmp[0].strip(), tmp[1].strip(), tmp[-2].strip()])
        summary = pd.DataFrame(names, columns=('tag', 'name', 'scheme'))
        sets = []
        tags = list(summary['tag'].values)
        for sym, tag in zip(self.atom['symbol'], self.atom['tag']):
            if sym.upper() in tags:
                sets.append(tags.index(sym.upper()))
            elif tag in tags:
                sets.append(tags.index(tag))
        self.atom['set'] = sets
        dim_starts = [i[0] + 1 for i in regex[_re_bas_dims]]
        dim_stops = [self._find_break(start) for start in dim_starts]
        basis_map = pd.concat([self._basis_map(start, stop, seht)
                               for start, stop, seht
                               in zip(dim_starts, dim_stops, summary.index)])
        basis_map.columns = ['shell', 'nprim', 'nbasis', 'set', 'spherical']
        return basis_map


    def parse_gaussian_basis_set(self):
        '''
        Parses the primitive exponents, coefficients and shell if BSSHOW specified in SEWARD.
        '''
        basis_map = self._basis_set_map()
        linenos = [i[0] + 1 for i in self.regex(_re_prims)]
        lisdx = 0
        lfsdx = 0
        basis_set = pd.DataFrame()
        blocks = []
        sets = basis_map.groupby('set')
        for sdx, seht in sets:
            shfunc = 0
            lfsdx += len(seht)
            starts = linenos[lisdx:lfsdx]
            lisdx = lfsdx
            prims = []
            for i, start in enumerate(starts):
                prim = seht['nprim'].values[i]
                bas = seht['nbasis'].values[i]
                chk1 = len(self[start].split())
                chk2 = len(self[start + 1].split())
                if chk1 == chk2:
                    block = self.pandas_dataframe(start, start + prim, bas + 2)
                else:
                    block = self[start:start + 2 * prim]
                    most = block[::2]
                    extr = block[1::2]
                    ncols = len(most[0].split()) + len(extr[0].split())
                    block = pd.read_csv(StringIO('\n'.join([i + j for i, j in zip(most, extr)])),
                                        delim_whitespace=True, names=range(ncols))
                alphas = pd.concat([block[1]] * bas).reset_index(drop=True).str.replace('D', 'E').astype(np.float64)
                coeffs = block[list(range(2, bas + 2))].unstack().reset_index(drop=True)
                primdf = pd.concat([alphas, coeffs], axis=1)
                primdf.columns = ['alpha', 'd']
                primdf['L'] = lorder.index(seht['shell'].values[i])
                primdf['shell'] = np.repeat(range(shfunc, shfunc + bas), prim)
                shfunc += bas
                prims.append(primdf)
            block = pd.concat(prims)
            block['set'] = sdx
            blocks.append(block)
        gaussian_basis_set = pd.concat(blocks).reset_index(drop=True)
        gaussian_basis_set['frame'] = 0
        self.gaussian_basis_set = gaussian_basis_set

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parse_atom()
        self.parse_gaussian_basis_set()


_re_bas_order = 'Basis Label        Type   Center'
_re_bas_znums = 'Associated Actual Charge'
_re_bas_names01 = 'Basis set label:'
_re_bas_names02 = 'Basis Set  '
_re_bas_dims = 'Shell  nPrim  nBasis  Cartesian Spherical Contaminant'
#_re_bas_names = 'Basis set:([A-z]{1,})([0-9]*)'
_re_num_prims = 'Number of primitives'
#_re_nbas = 'Number of basis functions'
_re_atom = 'Molecular structure info'
_re_prims = 'No.      Exponent    Contraction Coefficients'
_orb_slice = [slice(18*i, 18*i + 18) for i in range(4)]

def _fix_basis_set_order(df, shls, sets, funcs):
    mldict = {'': 0, 'x': 1, 'y': -1, 'z': 0}
    df['center'] -= 1
    try:
        df['n'] = df['type'].str[0].astype(np.int64)
    except ValueError:
        ns = list(df['type'].str[0].values)
        newns = [int(ns[0])]
        for i, j in zip(ns, ns[1:]):
            if j == '*':
                try:
                    cache = int(i) + 1
                    newns.append(cache)
                except ValueError:
                    newns.append(cache)
            else:
                try:
                    newns.append(int(j))
                except ValueError:
                    newns.append(cache)
        df['n'] = newns
    df['L'] = df['type'].str[1].map(lmap)
    df['ml'] = df['type'].str[2:]
    df['ml'].update(df['ml'].map(mldict))
    df['ml'].update(df['ml'].str[::-1])
    df['ml'] = df['ml'].astype(np.int64)
    shfuncs = []
    for seht in sets:
        tot = 0
        #lml_count = cart_lml_count
        lml_count = spher_lml_count
        for l, n in funcs[seht].items():
            for i in range(lml_count[l]):
                shfuncs += list(range(tot, n + tot))
            tot += n
    df['shell'] = shfuncs
    return df


def _parse_ovl(fp):
    ovl = pd.read_csv(fp, header=None)
    ovl.columns = ['coefficient']
    nbas = np.round(np.roots((1, 1, -2 * ovl.shape[0]))[1]).astype(np.int64)
    chi1 = np.empty(ovl.shape[0], dtype=np.int64)
    chi2 = np.empty(ovl.shape[0], dtype=np.int64)
    cnt = 0
    for i in range(nbas):
        for j in range(i + 1):
            chi1[cnt] = i
            chi2[cnt] = j
            cnt += 1
    ovl['chi1'] = chi1
    ovl['chi2'] = chi2
    ovl['frame'] = 0
    return ovl


def parse_molcas(file_path, momatrix=None, overlap=None, occvec=None, density=None, **kwargs):
    """
    Will parse a Molcas output file. Optionally it will attempt
    to parse additional information obtained from the same directory
    from specified Orb files or the AO overlap matrix and density matrix.
    If density keyword is specified, the momatrix keyword is ignored.

    Args
        file_path (str): Path to output file
        momatrix (str): file name of the C matrix of interest
        overlap (str): file name of the overlap matrix
        density (str): file name of the density matrix

    Returns
        parsed (Editor): contains many attributes similar to the
            exatomic universe
    """
    uni1 = Output(file_path, **kwargs)
    dirtree = sep.join(file_path.split(sep)[:-1])
    if density is not None:
        fp = sep.join([dirtree, density])
        if os.path.isfile(fp):
            dens = DensityMatrix(_parse_ovl(fp))
            uni1.density = dens
        else:
            print('Is {} in the same directory as {}?'.format(density, file_path))
    if momatrix is not None and density is None:
        fp = sep.join([dirtree, momatrix])
        if os.path.isfile(fp):
            orbs = Orb(fp)
            orbs.parse_momatrix()
            if occvec is not None:
                dens = DensityMatrix.from_momatrix(orbs.momatrix, occvec)
            else:
                dens = DensityMatrix.from_momatrix(orbs.momatrix, orbs.occupation_vector)
            uni1.momatrix = orbs.momatrix
            uni1.occupation_vector = orbs.occupation_vector
            uni1.density = dens
        else:
            print('Is {} in the same directory as {}?'.format(momatrix, file_path))
    if overlap is not None:
        fp = sep.join([dirtree, overlap])
        if os.path.isfile(fp):
            ovl = _parse_ovl(fp)
            uni1.overlap = ovl
        else:
            print('Is {} in the same directory as {}?'.format(overlap, file_path))
    return uni1
