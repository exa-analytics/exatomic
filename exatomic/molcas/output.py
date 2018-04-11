# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Molcas Output Parser
#####################
Multiple frames are not currently supported
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import six
import pandas as pd
import numpy as np
from six import StringIO
from exa import TypedMeta
from .editor import Editor
from exatomic import Atom
from exatomic.algorithms.numerical import _flat_square_to_triangle
from exatomic.algorithms.basis import lmap, spher_lml_count
from exatomic.core.basis import Overlap, BasisSet, BasisSetOrder
from exatomic.core.orbital import DensityMatrix, MOMatrix, Orbital
from exatomic.base import sym2z, z2sym


class OrbMeta(TypedMeta):
    momatrix = MOMatrix
    orbital = Orbital


class Orb(six.with_metaclass(OrbMeta, Editor)):
    """
    Parser for molcas coefficient matrix dumps (e.g. RasOrb).
    
    Note:
        This parser assumes the file contains data from a single
        calculation (i.e. a single frame).
    """
    def to_universe(self):
        raise NotImplementedError("This editor has no parse_atom method.")

    def parse_momatrix(self):
        """Wrapper for :func:`~exatomic.molcas.output.Orb.parse`."""
        self.parse()
    
    def parse_orbital(self):
        """Wrapper for :func:`~exatomic.molcas.output.Orb.parse`."""
        self.parse()

    def parse(self):
        """Parse all information contained in the orbital/coefficient matrix dump."""
        nmo = int(self[5])
        mo_coef_segments = self.find(" ORBITAL ", " OCCUPATION NUMBERS", 
                                     " ONE ELECTRON ENERGIES", "#INDEX", keys_only=True)
        start = mo_coef_segments[' ORBITAL '][0]
        stop = mo_coef_segments[' OCCUPATION NUMBERS'][0]
        start1 = mo_coef_segments[" ONE ELECTRON ENERGIES"][0]
        stop1 = mo_coef_segments["#INDEX"][0]
        try:
            end = mo_coef_segments[' OCCUPATION NUMBERS'][1]
        except IndexError:
            end = start1
        df = pd.read_fwf(StringIO("\n".join(self[start:stop])), widths=[22]*5, names=range(5))
        starting_points = df[df[0].str.contains(" ORBITAL")].index.values + 1
        didx = starting_points[1] - 2
        coef = []
        for i in starting_points:
            coef.append(df.iloc[i:i+didx].values.ravel()[:nmo])
        coef = np.concatenate(coef).astype(float)
        orb = [i for i in range(nmo) for _ in range(nmo)]
        chi = [j for _ in range(nmo) for j in range(nmo)]
        momatrix = pd.DataFrame.from_dict({'coef': coef, 'chi': chi, 'orbital': orb, 'chi': chi})
        momatrix['frame'] = 0
        occ = pd.read_fwf(StringIO("\n".join(self[stop+1:end-1])), widths=[22]*5, names=range(5)).values.ravel().astype(float)
        occ = occ[~np.isnan(occ)]
        nrg = pd.read_fwf(StringIO("\n".join(self[start1+1:stop1])), widths=[12]*10, names=range(10)).values.ravel().astype(float)
        nrg = nrg[~np.isnan(nrg)]
        orbital = pd.DataFrame.from_dict({'occupation': occ, 'energy': nrg})
        orbital['frame'] = 0
        orbital['spin'] = 0
        orbital['group'] = 0
        orbital['vector'] = range(nmo)
        self.momatrix = momatrix
        self.orbital = orbital
        self.occupation_vector = occ

    def __init__(self, *args, **kwargs):
        super(Orb, self).__init__(*args, **kwargs)



class OutMeta(TypedMeta):
    atom = Atom
    basis_set = BasisSet
    basis_set_order = BasisSetOrder


class Output(six.with_metaclass(OutMeta, Editor)):

    def add_orb(self, path, mocoefs='coef', orbocc='occupation'):
        orb = Orb(path)
        if mocoefs != 'coef' and orbocc == 'occupation':
            orbocc = mocoefs
        for attr, col, de in [['momatrix', mocoefs, 'coef'],
                              ['orbital', orbocc, 'occupation']]:
            df = getattr(self, attr, None)
            if df is None:
                setattr(self, attr, getattr(orb, attr))
            elif col in df.columns:
                raise ValueError('This action would replace '
                                 '"{}" in uni.{}'.format(col, attr))
            else:
                df[col] = getattr(orb, attr)[de]

    def add_overlap(self, path):
        self.overlap = Overlap.from_column(path)

    def parse_atom(self):
        """Parses the atom list generated in SEWARD."""
        _re_atom = 'Label   Cartesian Coordinates'
        starts = [i + 2 for i in self.find(_re_atom, keys_only=True)]
        stops = starts[:]    # Copy the list
        for i in range(len(stops)):
            while len(self[stops[i]].strip().split()) > 3:
                stops[i] += 1
                if not self[stops[i]].strip(): break
            stops[i] -= 1
        lns = StringIO('\n'.join([self._lines[i] for j in (range(i, j + 1)
                                 for i, j in zip(starts, stops)) for i in j]))
        atom = pd.read_csv(lns, delim_whitespace=True,
                           names=['tag', 'x', 'y', 'z'])
        atom['symbol'] = atom['tag'].str.extract(
            '([A-z]{1,})([0-9]*)', expand=False)[0].str.lower().str.title()
        atom['Z'] = atom['symbol'].map(sym2z).astype(np.int64)
        atom['label'] = range(atom.shape[0])
        atom['frame'] = 0
        self.atom = atom

    def parse_basis_set_order(self):
        """
        Parses the shell ordering scheme if BSSHOW specified in SEWARD.
        """
        _re_bas_order = 'Basis Label        Type   Center'
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
        try:
            df['l'] = df['ml'].copy()
            df['l'].update(df['l'].map({'': 0, 'x': 1, 'y': 0, 'z': 0}))
            df['l'] = df['l'].astype(np.int64)
            df['m'] = df['ml'].copy()
            df['m'].update(df['m'].map({'': 0, 'y': 1, 'x': 0, 'z': 0}))
            df['m'] = df['m'].astype(np.int64)
            df['n'] = df['ml'].copy()
            df['n'].update(df['n'].map({'': 0, 'z': 1, 'x': 0, 'y': 0}))
            df['n'] = df['n'].astype(np.int64)
        except:
            pass
        df['ml'].update(df['ml'].map(mldict))
        df['ml'].update(df['ml'].str[::-1])
        df['ml'] = df['ml'].astype(np.int64)
        funcs = self.basis_set.functions_by_shell()
        shells = []
        for seht in self.atom['set']:
            tot = 0
            lml_count = spher_lml_count
            for l, n in funcs[seht].items():
                #for i in range(lml_count[l]):
                for _ in range(lml_count[l]):
                    shells += list(range(tot, n + tot))
                tot += n
        df['shell'] = shells
        df['frame'] = 0
        self.basis_set_order = df

    def parse_basis_set(self):
        """Parses the primitive exponents, coefficients and
        shell if BSSHOW specified in SEWARD."""
        _re_bas_0 = 'Shell  nPrim  nBasis  Cartesian Spherical Contaminant'
        _re_bas_1 = 'Label   Cartesian Coordinates / Bohr'
        _re_bas_2 = 'No.      Exponent    Contraction Coefficients'
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
            nmatch = len(self[start].split())
            neat = nmatch == len(self[start + 1].split())
            if neat:
                block = self.pandas_dataframe(start, start + nprim, nbas + 2)
            else:
                # Extra obfuscation to handle exotic cases
                ext = 1
                while nmatch != len(self[start + ext].split()):
                    ext += 1
                stop = start + ext * nprim
                collated = [''.join(self[start + i * ext : start + i * ext + ext])
                            for i in range(nprim)]
                ncols = len(collated[0].split())
                block = pd.read_csv(StringIO('\n'.join(collated)),
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
        self.meta['spherical'] = True
        if self.basis_set.lmax < 2:
            self.meta['spherical'] = False

    def __init__(self, *args, **kwargs):
        super(Output, self).__init__(*args, **kwargs)


class HDFMeta(TypedMeta):
    atom = Atom
    overlap = Overlap
    orbital = Orbital
    momatrix = MOMatrix
    basis_set_order = BasisSetOrder


class HDF(six.with_metaclass(HDFMeta, object)):

    _getter_prefix = 'parse'
    _to_universe = Editor._to_universe

    def to_universe(self):
        return self._to_universe()

    def parse_atom(self):
        Z = pd.Series(self._hdf['CENTER_CHARGES']).astype(np.int64)
        xyzs = np.array(self._hdf['CENTER_COORDINATES'])
        labs = pd.Series(self._hdf['CENTER_LABELS']).apply(
                         lambda s: s.decode('utf-8').strip())
        self.atom = pd.DataFrame.from_dict({
            'x': xyzs[:, 0], 'y': xyzs[:,1], 'z': xyzs[:,2],
            'symbol': Z.map(z2sym), 'label': labs, 'Z': Z, 'frame': 0})

    def parse_basis_set_order(self):
        bso = np.array(self._hdf['BASIS_FUNCTION_IDS'])
        df = {'center': bso[:, 0], 'shell': bso[:, 1],
              'L': bso[:, 2], 'frame': 0}
        if bso.shape[1] == 4:
            df['ml'] = bso[:, 3]
        else:
            df['l'] = bso[:, 3]
            df['m'] = bso[:, 4]
            df['n'] = bso[:, 5]
        self.basis_set_order = pd.DataFrame.from_dict(df)

    def parse_orbital(self):
        ens = np.array(self._hdf['MO_ENERGIES'])
        self.orbital = pd.DataFrame.from_dict({
            'energy': ens, 'vector': range(len(ens)),
            'occupation': np.array(self._hdf['MO_OCCUPATIONS']),
            'label': pd.Series(self._hdf['MO_TYPEINDICES']).apply(
                               lambda s: s.decode('utf-8')),
            'frame': 0, 'group': 0, 'spin': 0})

    def parse_overlap(self):
        self.overlap = Overlap.from_column(
            _flat_square_to_triangle(np.array(self._hdf['AO_OVERLAP_MATRIX'])))

    def parse_momatrix(self):
        coefs = np.array(self._hdf['MO_VECTORS'])
        dim = np.int64(np.sqrt(coefs.shape[0]))
        self.momatrix = pd.DataFrame.from_dict({
            'orbital': np.repeat(range(dim), dim), 'frame': 0,
            'chi': np.tile(range(dim), dim), 'coef': coefs})

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        raise KeyError()

    def __init__(self, *args, **kwargs):
        try:
            import h5py
        except ImportError:
            print("You must install h5py for access to HDF5 utilities.")
            return
        self._hdf = h5py.File(args[0], 'r')



def parse_molcas(fp, momatrix=None, overlap=None, occvec=None, **kwargs):
    """
    Will parse a Molcas output file. Optionally it will attempt
    to parse additional information obtained from the same directory
    from specified Orb files or the AO overlap matrix and density matrix.
    If density keyword is specified, the momatrix keyword is ignored.

    Args:
        fp (str): Path to output file
        momatrix (str): file name of the C matrix of interest
        overlap (str): file name of the overlap matrix
        occvec (str): an occupation vector

    Returns:
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
