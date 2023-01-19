# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
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
import re
import six
import pandas as pd
import numpy as np
from six import StringIO
from exatomic.exa import TypedMeta
import exatomic
from .editor import Editor
from exatomic.core import Atom, Gradient, Frequency
from exatomic.algorithms.numerical import _flat_square_to_triangle, _square_indices
from exatomic.algorithms.basis import lmap, spher_lml_count
from exatomic.core.basis import (Overlap, BasisSet, BasisSetOrder,
                                 deduplicate_basis_sets)
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


    def _read_one(self, found, keys, start, stop, osh, old, ret, key):
        if not old:
            # In order for this not to break if someone decides
            # to change the print width format, compute it here
            ln = self[start]
            width = ln[3:].index(' ') + 3
            keys['widths'] = [width] * len(ln.split())
            keys['names'] = range(len(keys['widths']))
        df = pd.read_fwf(StringIO('\n'.join(self[start:stop])), **keys)
        if not osh:
            ret.update({key: df.stack().dropna().values})
        else: # Clean for open shell calcs
            rm = found[1] - found[0] - 1
            df.drop([rm - 1, rm], inplace=True)
            df[0] = df[0].astype(np.float64)
            df = df.stack().dropna().astype(np.float64).values
            ret.update({key: df[:len(df)//2],
                        key+'1': df[len(df)//2:]})


    def parse_momatrix(self):
        _re_orb = 'ORBITAL'
        _re_occ = 'OCCUPATION NUMBERS'
        _re_ens = 'ONE ELECTRON ENERGIES'
        _re_osh = '#UORB'
        _re_idx = '#INDEX'
        _re_hmn = 'human'
        found = self.find(_re_orb, _re_occ, _re_ens,
                          _re_osh, _re_idx, keys_only=True)
        found[_re_hmn] = [i for i in found[_re_occ]
                          if _re_hmn in self[i].lower()]
        # Dimensions per irrep
        dims = list(map(int, self[5].split()))
        ndim = sum(dims)
        osh = len(found[_re_osh]) > 0
        start = found[_re_orb][0] + 1
        stops = found[_re_occ] + found[_re_ens]
        stop = min(stops) - 1
        # Old file format
        old = len(self[start].split()) == 1
        # MOMatrix table
        widths = [18] * 4 if old else [22] * 5
        kws = {'widths': widths, 'names': range(len(widths))}
        df = pd.read_fwf(StringIO('\n'.join(self[start:stop])),
                         widths=widths, names=range(len(widths)))
        df.drop(df[df[0].str.contains(_re_orb)].index, inplace=True)
        # Orbital occupations
        mo, orb = {}, {}
        start = found[_re_occ][0] + 1
        stops = found[_re_hmn] + found[_re_ens] + found[_re_idx] + [len(self) + 1]
        stop = min(stops) - 1
        self._read_one(found[_re_occ], kws, start, stop,
                       osh, old, orb, 'occupation')
        # Orbital energies
        if found[_re_ens]:
            start = found[_re_ens][0] + 1
            stop = found[_re_idx][0]
            self._read_one(found[_re_ens], kws, start, stop,
                           osh, old, orb, 'energy')
        else:
            orb.update({'energy': 0})

        # Get all the groupby indices
        if len(dims) > 1: # Symmetrized calc.
            mo.update({
                'irrep': [i for i, d in enumerate(dims) for _ in range(d * d)],
                'orbital': [i for d in dims for i in range(d) for _ in range(d)],
                'chi': [i for d in dims for _ in range(d) for i in range(d)]})
            orb.update({'vector': [j for d in dims for j in range(d)],
                        'irrep': [i for i, d in enumerate(dims) for _ in range(d)]})
        else:
            ordx, chidx = _square_indices(ndim)
            mo.update({'orbital': ordx, 'chi': chidx})
            orb.update({'vector': range(ndim)})

        # Unused groupby indices
        orb.update({'group': 0, 'spin': 0, 'frame': 0})
        mo.update({'frame': 0})

        if not osh:
            df[0] = df[0].astype(np.float64)
            mo.update({'coef': df.stack().dropna().values})
        else: # Open shell calc.
            off = found[_re_orb][0] + 1
            df.drop(found[_re_osh][0] - off, inplace=True)
            df[0] = df[0].astype(np.float64)
            coef = df.stack().dropna().values
            mo.update({'coef': coef[:len(coef)//2],
                       'coef1': coef[len(coef)//2:]})

        self.momatrix = pd.DataFrame.from_dict(mo)
        self.orbital = pd.DataFrame.from_dict(orb)


    def __init__(self, *args, **kwargs):
        super(Orb, self).__init__(*args, **kwargs)



class OutMeta(TypedMeta):
    atom = Atom
    gradient = Gradient
    basis_set = BasisSet
    basis_set_order = BasisSetOrder
    sf_dipole_moment = pd.DataFrame
    sf_quadrupole_moment = pd.DataFrame
    sf_angmom = pd.DataFrame
    sf_energy = pd.DataFrame
    so_energy = pd.DataFrame
    sf_oscillator = pd.DataFrame
    so_oscillator = pd.DataFrame
    frequency = Frequency
    natural_occ = pd.DataFrame
    caspt2_energy = pd.DataFrame


class Output(six.with_metaclass(OutMeta, Editor)):
    _resta = "STATE"
    def _property_parsing(self, props, data_length):
        ''' Helper method for parsing the spin-free properties sections. '''
        all_dfs = []
        for idx, prop in enumerate(props):
            # find where the data blocks are printed
            starts = np.array(self.find(self._resta, start=prop, keys_only=True)) + prop + 2
            # data_length should always be the same
            stops = starts + data_length
            # use np.ceil as if we have a 7x7 matrix we get one data set
            # with 4 columns and one with 3 columns
            n = int(np.ceil(data_length/4))
            dfs = []
            # grab all of the data
            # we use all hits up to n as there are many places with the
            # 'STATE' keyword so we want to just grab the data starting
            # at the property keyword
            for ndx, (start, stop) in enumerate(zip(starts[:n], stops[:n])):
                # should be four in all but the last data set
                ncols = len(self[start-2].split())
                df = self.pandas_dataframe(start, stop, ncol=ncols)
                # the only column we do not need will always be the
                # column with name 0
                df.drop(0, axis=1, inplace=True)
                # set the columns as they should be
                df.columns = list(range(ndx*4, ndx*4+ncols-1))
                dfs.append(df)
            # put the component together
            all_dfs.append(pd.concat(dfs, axis=1))
            all_dfs[-1]['component'] = idx
        df = pd.concat(all_dfs, ignore_index=True)
        return df

    def _oscillator_parsing(self, start_idx):
        ''' Helper method to parse the oscillators. '''
        ldx = start_idx + 6
        oscillators = []
        while '-----' not in self[ldx]:
            oscillators.append(self[ldx].split())
            ldx += 1
        df = pd.DataFrame(oscillators)
        df.columns = ['nrow', 'ncol', 'oscil', 'a_x', 'a_y', 'a_z', 'a_tot']
        df[['nrow', 'ncol']] = df[['nrow', 'ncol']].astype(int)
        df[['nrow', 'ncol']] -= [1, 1]
        df[['nrow', 'ncol']] = df[['nrow', 'ncol']].astype('category')
        cols = ['oscil', 'a_x', 'a_y', 'a_z', 'a_tot']
        df[cols] = df[cols].astype(np.float64)
        return df

    def add_orb(self, path, mocoefs='coef', orbocc='occupation'):
        """
        Add a MOMatrix and Orbital table to a molcas.Output. If path is
        an Editor containing momatrix and orbital tables then adds them
        directly, otherwise assumes it is a molcas.Orb file.

        Args:
            path (str, :class:`exatomic.core.editor.Editor`): path to file or Editor object
            mocoefs (str): rename coefficients
            orbocc (str): rename occupations
        """
        if isinstance(path, exatomic.Editor): orb = path
        else: orb = Orb(path)
        if mocoefs != 'coef' and orbocc == 'occupation':
            orbocc = mocoefs
        # MOMatrix
        curmo = getattr(self, 'momatrix', None)
        if curmo is None:
            self.momatrix = orb.momatrix
            if mocoefs != 'coef':
                self.momatrix.rename(columns={'coef': mocoefs}, inplace=True)
        else:
            if mocoefs in self.momatrix.columns:
                raise ValueError('This action would overwrite '
                                 'coefficients. Specify mocoefs parameter.')
            for i, default in enumerate(['coef', 'coef1']):
                final = mocoefs + '1' if i else mocoefs
                if default in orb.momatrix:
                    self.momatrix[final] = orb.momatrix[default]
        # Orbital
        curorb = getattr(self, 'orbital', None)
        if curorb is None:
            self.orbital = orb.orbital
            if orbocc != 'occupation':
                self.orbital.rename(columns={'occupation': orbocc}, inplace=True)
        else:
            if orbocc in self.orbital.columns:
                raise ValueError('This action would overwrite '
                                 'occupations. Specify orbocc parameter.')
            for i, default in enumerate(['occupation', 'occupation1']):
                final = orbocc + '1' if i else orbocc
                if default in orb.orbital.columns:
                    self.orbital[final] = orb.orbital[default]


    def add_overlap(self, path):
        try: # If it's an ASCII text file
            self.overlap = Overlap.from_column(path)
        except Exception: # If it's an HDF5 file
            hdf = HDF(path)
            if 'DESYM_CENTER_CHARGES' not in hdf._hdf:
                self.overlap = hdf.overlap
                return
            if 'irrep' not in self.momatrix:
                raise Exception("Trying to set symmetrized overlap with "
                                "desymmetrized MOMatrix data.")
            ovl = pd.DataFrame(np.array(hdf._hdf['AO_OVERLAP_MATRIX']),
                               columns=('coef',))
            ovl['irrep'] = self.momatrix['irrep']
            ovl['chi0'] = self.momatrix['chi']
            ovl['chi1'] = self.momatrix['orbital']
            ovl['frame'] = 0
            self.overlap = ovl


    def _check_atom_sym(self):
        """Parses a less accurate atom list to check for symmetry."""
        _re_sym = 'Cartesian Coordinates / Bohr, Angstrom'
        start = self.find(_re_sym, keys_only=True)[0] + 4
        cols = ['center', 'tag', 'x', 'y', 'z', 'd1', 'd2', 'd3']
        stop = start
        while len(self[stop].split()) == len(cols): stop += 1
        atom = self.pandas_dataframe(start, stop, cols)
        atom.drop(['d1', 'd2', 'd3'], axis=1, inplace=True)
        atom['symbol'] = atom['tag'].str.extract(
            '([A-z]{1,})([0-9]*)', expand=False)[0].str.lower().str.title()
        atom['frame'] = 0
        atom['center'] -= 1
        return atom


    def parse_atom(self, seward=True):
        """Parses the atom list generated in SEWARD."""
        if seward:
            _re_atom0 = 'Label   Cartesian Coordinates'
            _re_atom1 = 'Center  Label'
            found = self.find(_re_atom0, _re_atom1, keys_only=True)
            if found[_re_atom0]:
                accurate = True
                starts = [i + 2 for i in found[_re_atom0]]
            else:
                accurate = False
                starts = [i + 1 for i in found[_re_atom1]]
            stops = starts[:]    # Copy the list
            for i in range(len(stops)):
                while len(self[stops[i]].strip().split()) > 3:
                    stops[i] += 1
                    if not self[stops[i]].strip(): break
                stops[i] -= 1
            if accurate:
                lns = StringIO('\n'.join([self._lines[i] for j in (range(i, j + 1)
                                         for i, j in zip(starts, stops)) for i in j]))
                cols = ['tag', 'x', 'y', 'z']
            else:
                lns = StringIO('\n'.join(self[starts[0]:stops[0] + 1]))
                cols = ['center', 'tag', 'x', 'y', 'z', 'xa', 'ya', 'za']
            atom = pd.read_csv(lns, delim_whitespace=True,
                               names=cols)
            if len(cols) == 8:
                atom.drop(['xa', 'ya', 'za'], axis=1, inplace=True)
            atom['symbol'] = atom['tag'].str.extract(
                '([A-z]{1,})([0-9]*)', expand=False)[0].str.lower().str.title()
            atom['Z'] = atom['symbol'].map(sym2z).astype(np.int64)
            atom['center'] = range(atom.shape[0])
            atom['frame'] = 0
            self.atom = atom
            # Work-around for symmetrized calcs?
            allatom = self._check_atom_sym()
            self.meta['symmetrized'] = False
            if allatom.shape[0] > self.atom.shape[0]:
                self.atom = allatom
                self.meta['symmetrized'] = True
                try:
                    self.atom['utag'] = _add_unique_tags(self.atom)
                except ValueError:
                    pass
        else:
            _reatom = "Nuclear coordinates for the next iteration / Bohr"
            _reatom02 = "Nuclear coordinates of the final structure / Bohr"
            found = self.find(_reatom, _reatom02, keys_only=True)
            if len(found[_reatom02]) > 1:
                raise ValueError("Found more than one final atom table.")
            elif len(found[_reatom02]) == 0:
                raise ValueError("Could not find a final atom table in the file.")
            starts = np.array(found[_reatom])
            starts = np.array(list(starts)+found[_reatom02])+3
            stop = starts[0]
            while self[stop].strip(): stop += 1
            stops = starts + (stop - starts[0])
            cols = ['symbol', 'x', 'y', 'z']
            dfs = []
            for idx, (start, stop) in enumerate(zip(starts, stops)):
                df = self.pandas_dataframe(start, stop, ncol=4)
                df.columns = cols
                label = np.concatenate(list(map(lambda x: re.findall(r'\D+', x),
                                                df['symbol'])))
                if label.shape[0] > df.shape[0]:
                    text = "There was an issue with interpreting the labels " \
                           +"of the atom table. Expected {} labels but " \
                           +"interpreted {}"
                    raise ValueError(text.format(df.shape[0], label.shape[0]))
                df['symbol'] = label
                df['Z'] = df['symbol'].map(sym2z)
                df['set'] = range(df.shape[0])
                df['frame'] = idx
                dfs.append(df)
            atom = pd.concat(dfs, ignore_index=True)
            self.atom = atom

    def parse_gradient(self):
        _regrad = "Molecular gradients"
        _reirr = "Irreducible representation:"
        found = self.find(_regrad, _reirr, keys_only=True)
        if not found[_regrad]:
            return
        if len(found[_regrad]) != len(found[_reirr]):
            text = "Do not have support for multiple irreducible " \
                   +"representations yet."
            raise NotImplementedError(text)
        starts = np.array(found[_regrad]) + 8
        stop = starts[0]
        while self[stop].strip()[:2] != '--': stop += 1
        stops = starts + (stop - starts[0])
        dfs = []
        for idx, (start, stop) in enumerate(zip(starts, stops)):
            df = self.pandas_dataframe(start, stop, ncol=4)
            cols = ['symbol', 'fx', 'fy', 'fz']
            df.columns = cols
            label = np.concatenate(list(map(lambda x: re.findall(r'\D+', x),
                                            df['symbol'])))
            df['symbol'] = label
            df['atom'] = range(df.shape[0])
            df['frame'] = idx
            df['Z'] = df['symbol'].map(sym2z)
            dfs.append(df)
        grad = pd.concat(dfs, ignore_index=True)
        self.gradient = grad

    def parse_frequency(self, linear=False, normalize=True):
        _refreq = "Frequency:"
        _reint = "Intensity:"
        _remass = "Red. mass:"
        _rerot = "Note that rotational and translational degrees " \
                 +"have been automatically removed,"
        found = self.find(_refreq, _reint, _rerot, _remass)
        if not found[_refreq]:
            return
        start = found[_remass][-1][0] + 2
        stop = found[_remass][-1][0] + 2
        while self[stop].strip(): stop += 1
        lines = stop - start
        dfs = []
        count = 0
        arr = zip(found[_refreq], found[_reint], found[_remass])
        for idx, ((_, freq), (_, inten), (ldx, mass)) in enumerate(arr):
            # parse the normal modes
            start = ldx + 2
            stop = start + lines
            arr = self.pandas_dataframe(start, stop, ncol=8)
            arr.dropna(how='all', axis=1, inplace=True)
            # extract the atomic symbols
            symbols = arr[0].drop_duplicates()
            symbols = list(map(lambda x: re.findall(r'\D+', x)[0], symbols))
            nat = len(symbols)
            if linear and idx == 0 and found[_rerot]:
                arr.drop([2, 3, 4, 5, 6], axis=1, inplace=True)
            elif not found[_rerot] and idx == 0 and not linear:
                print("Found rotational modes in the calculation")
                continue
            # set the columns where the normal modes are on the table
            cols = range(2, arr.columns.max()+1)
            arr.drop([0], axis=1, inplace=True)
            # re-organize the normal mode data
            tmp = arr.groupby(1).apply(lambda x: x[cols].values.T.flatten())
            df = pd.DataFrame(tmp.to_dict())
            df.columns = ['dx', 'dy', 'dz']
            errortext = "Something went wrong when trying to turn the {} " \
                        +"into floating point numbers."
            # get the frequencies
            # skip the first element as it is the table label
            freqs = freq.split()[1:]
            # replace all imaginary signs with '-'
            freqs = list(map(lambda x: x.replace('i', '-'), freqs))
            # turn them into floats
            try:
                freqs = list(map(float, freqs))
            except ValueError:
                print(freqs)
                raise ValueError(errortext.format('frequencies'))
            # get the reduced masses
            # skip the first element as it is the table label
            rmass = mass.split()[2:]
            # turn them into floats
            try:
                rmass = list(map(float, rmass))
            except ValueError:
                print(rmass)
                raise ValueError(errortext.format('reduced masses'))
            # get the IR intensities
            # skip the first element as it is the table label
            irint = inten.split()[1:]
            # turn them into floats
            try:
                irint = list(map(float, irint))
            except ValueError:
                print(irint)
                raise ValueError(errortext.format('IR intensities'))
            # put everything together
            df['symbol'] = np.tile(symbols, len(cols))
            df['label'] = np.tile(range(nat), len(cols))
            df['Z'] = df['symbol'].map(sym2z)
            df['frequency'] = np.repeat(freqs, nat)
            df['r_mass'] = np.repeat(rmass, nat)
            df['ir_int'] = np.repeat(irint, nat)
            df.loc[df['ir_int'].abs() < 1e-9, 'ir_int'] = 0
            df['freqdx'] = np.repeat(range(count*6, (count+1)*len(cols)), nat)
            count += 1
            df['frame'] = 0
            if normalize and found[_rerot]:
                df[['dx', 'dy', 'dz']] *= np.sqrt(df['r_mass'].values.reshape(-1,1))
            elif normalize and not found[_rerot]:
                pass
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        cols = ['Z', 'label', 'dx', 'dy', 'dz', 'frequency', 'freqdx',
                'ir_int', 'r_mass', 'symbol', 'frame']
        self.frequency = df[cols]

    def parse_basis_set_order(self):
        """
        Parses the shell ordering scheme if BSSHOW specified in SEWARD.
        """
        _re_bas_order = 'Basis Label        Type   Center'
        starts = [i + 1 for i in self.find(_re_bas_order, keys_only=True)]
        lines, irreps, vecs, vec, nsym = [], [], [], 0, 0
        for i, start in enumerate(starts):
            stop = start
            while self[stop].strip():
                lines.append(stop)
                irreps.append(i)
                vecs.append(vec)
                nsym = max(nsym, len(self[stop].split()))
                stop += 1
                vec += 1
            lines.append(stop)
            vec = 0
        symcols = [('ocen{}'.format(i), 'sign{}'.format(i))
                    for i in range((nsym - 5) // 2)]
        if symcols: self.meta['symmetrized'] = True
        cols = ['idx', 'tag', 'type', 'center', 'phase'] + \
               [c for t in symcols for c in t]
        df = pd.read_csv(StringIO('\n'.join(self[i] for i in lines)),
                         delim_whitespace=True, names=cols)
        # Symmetrized basis functions
        for col in df.columns:
            if col.startswith('ocen'):
                df[col] = df[col].fillna(0.).astype(np.int64) - 1
        # Extra info for symmetrized calcs
        df['irrep'] = irreps
        df['vector'] = vecs
        df.drop(['idx'], inplace=True, axis=1)
        df['frame'] = 0
        # 0-based indexing
        df['center'] -= 1
        # Clean up (L, ml) values to ints
        df['L'] = df['type'].str[1].map(lmap)
        df['ml'] = df['type'].str[2:]
        df['ml'].update(df['ml'].map({'': 0, 'x': 1, 'y': -1, 'z': 0}))
        try:
            df['ml'].update(df['ml'].str[::-1])
        except Exception:
            pass
        df['ml'] = df['ml'].astype(np.int64)
        # Seward and gateway may each print basis info.
        # See if this happened and if so, keep only the first half.
        try: # Apparently BSSHOW doesn't always print the basis set.
            if 'set' not in self.atom.columns:
                self.parse_basis_set()
        except ValueError:
            self.basis_set_order = df
            return
        bs = self.basis_set
        sp = self.meta['spherical']
        nbf = self.atom.set.map(bs.functions(sp).groupby('set').sum()).sum()
        if df.shape[0] > nbf:
            df = df.loc[:nbf - 1]
            irreps = irreps[:nbf]
            vecs = vecs[:nbf]
        self.basis_set_order = df
        shls = []
        grps = df.groupby(['irrep', 'center', 'L', 'ml'])
        for (_, _, L, ml), grp in grps:    # (? cen L, ml)
            shl = 0
            for _ in grp.index:
                shls.append(shl)
                shl += 1
        self.basis_set_order['shell'] = shls


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

    def parse_sf_dipole_moment(self):
        ''' Get the Spin-Free electric dipole moment from RASSI '''
        # define the search string
        _retdm = "PROPERTY: MLTPL  1"
        component_map = {0: 'x', 1: 'y', 2: 'z'}
        found = self.find(_retdm, keys_only=True)
        if not found:
            return
        if len(found) > 6:
            # in new version of molcas they introduced
            # extra matrices that are printed so this
            # was added due to that change
            props = np.array(found)[:6:2]
        else:
            props = np.array(found)[:3]
        # find the number of lines that the data spans
        stop = props[0] + 5
        while self[stop].strip(): stop += 1
        data_length = stop - props[0] - 5
        # get the data
        stdm = self._property_parsing(props, data_length)
        stdm['component'] = stdm['component'].map(component_map)
        self.sf_dipole_moment = stdm

    def parse_sf_quadrupole_moment(self):
        ''' Get the Spin-Free electric quadrupole moment from RASSI '''
        _requad = "PROPERTY: MLTPL  2"
        component_map = {0: 'xx', 1: 'xy', 2: 'xz', 3: 'yy', 4: 'yz', 5: 'zz'}
        found = self.find(_requad, keys_only=True)
        if not found:
            return
        # molcas prints the upper triangular tensor
        # elements of the full quadrupole tensor
        props = np.array(found)[:6]
        # find the number of lines that the data spans
        stop = props[0] + 5
        while self[stop].strip(): stop += 1
        data_length = stop - props[0] - 5
        # get the data
        sqdm = self._property_parsing(props, data_length)
        sqdm['component'] = sqdm['component'].map(component_map)
        self.sf_quadrupole_moment = sqdm

    def parse_sf_angmom(self):
        ''' Get the Spin-Free angular momentum from RASSI '''
        _reangm = "PROPERTY: ANGMOM"
        component_map = {0: 'x', 1: 'y', 2: 'z'}
        found = self.find(_reangm, keys_only=True)
        if not found:
            return
        props = np.array(found)[:3]
        # find the number of lines that the data spans
        stop = props[0] + 5
        while self[stop].strip(): stop += 1
        data_length = stop - props[0] - 5
        # get the data
        sangm = self._property_parsing(props, data_length)
        sangm['component'] = sangm['component'].map(component_map)
        self.sf_angmom = sangm

    def parse_sf_energy(self):
        ''' Get the Spin-Free energies from RASSI or RASSCF '''
        _reenerg = " RASSI State "
        _reenerg_rasscf = " RASSCF root number"
        found = self.find(_reenerg, _reenerg_rasscf)
        key = ''
        if found[_reenerg]:
            if found[_reenerg_rasscf]:
                msg = "Found RASSCF and RASSI Spin-Free energies.\n" \
                      +"Will only parse the RASSI Spin-Free energies."
                print(msg)
            key = _reenerg
        elif found[_reenerg_rasscf]:
            key = _reenerg_rasscf
        else:
            return
        # should not be necessary but you never know
        if key == '':
            msg = "There was an issue in determining the key to " \
                  +"be used in the parsing of the SF energies. " \
                  +"Please open an issue ticket as this should " \
                  +"never show."
            raise ValueError(msg)
        energies = []
        for _, line in found[key]:
            energy = float(line.split()[-1])
            energies.append(energy)
        # should always be the case that the first energy is the
        # ground state minimum energy
        rel_energy = list(map(lambda x: x - energies[0], energies))
        df = pd.DataFrame.from_dict({'energy': energies, 'rel_energy': rel_energy})
        self.sf_energy = df

    def parse_so_energy(self):
        ''' Get the Spin-Orbit energies from RASSI '''
        _reenerg = " SO-RASSI State "
        found = self.find(_reenerg)
        if not found:
            return
        energies = []
        for _, line in found:
            energy = float(line.split()[-1])
            energies.append(energy)
        rel_energy = list(map(lambda x: x - energies[0], energies))
        df = pd.DataFrame.from_dict({'energy': energies, 'rel_energy': rel_energy})
        self.so_energy = df

    def parse_sf_oscillator(self):
        ''' Get the printed Spin-Free oscillators from RASSI '''
        # TODO: check how this has to be adjusted for different
        #       molcas print levels
        _reosc = "++ Dipole transition strengths (spin-free states):"
        found = self.find(_reosc, keys_only=True)
        if not found:
            return
        if len(found) > 1:
            raise NotImplementedError("We have found more than one key for the spin-free " \
                                      +"oscillators.")
        df = self._oscillator_parsing(found[0])
        self.sf_oscillator = df

    def parse_so_oscillator(self):
        ''' Get the printed Spin-Orbit oscillators from RASSI '''
        # TODO: check how this has to be adjusted for different
        #       molcas print levels
        _reosc = "++ Dipole transition strengths (SO states):"
        found = self.find(_reosc, keys_only=True)
        if not found:
            return
        if len(found) > 1:
            raise NotImplementedError("We have found more than one key for the spin-orbit " \
                                      +"oscillators.")
        df = self._oscillator_parsing(found[0])
        self.so_oscillator = df

    def parse_natural_occ(self):
        # define search string for natural occupations
        _renoo = "Natural orbitals and occupation numbers for root"
        # define search string for energies per root
        _reeng = "energy="
        found = self.find(_renoo, _reeng)
        if not found[_renoo]:
            return
        # get the occupations, roots, and energies of each root
        occs = {}
        roots = []
        energies = []
        for (nooldx, nooline), (_, engline) in zip(found[_renoo], found[_reeng]):
            energy = float(engline.split('=')[-1].strip())
            energies.append(energy)
            root = list(map(int, re.findall(r'\d+', nooline)))
            # just an error check should never actually happen
            if len(root) != 1:
                raise ValueError("Found more than one group of numbers where one " \
                                 +"number should have been listed for the root. " \
                                 +"Found line {}".format(nooline))
            roots.append(root[0])
            # get the natural occupations
            dldx = nooldx+1
            vals = {}
            while self[dldx].strip() != '':
                d = self[dldx].split()
                if d[0] == 'sym':
                    arr = list(map(float, d[2:]))
                    symm = int(d[1].replace(':', ''))
                    if root[0] == 1:
                        occs[symm] = []
                    vals[symm] = []
                else:
                    arr = list(map(float, d))
                vals[symm].append(arr)
                dldx += 1
            for key, item in vals.items():
                occs[key].append(list(np.concatenate(item)))
        dfs = []
        for symm, occ in occs.items():
            df = pd.DataFrame(occ)
            df['symmetry'] = symm
            df['root'] = roots
            df['energy'] = energies
            df['rel_energy'] = df['energy'] - df['energy'].values.min()
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        self.natural_occ = df.sort_values(by=['symmetry', 'root']).reset_index(drop=True)

    def parse_caspt2_energy(self):
        # parsing strings
        _resspt2 = " CASPT2 Root"
        _remspt2 = " MS-CASPT2 Root"
        _reref = "Reference energy:"
        _reweight = "Reference weight:"
        _reroot = "Compute H0 matrices for state"
        _respin = "Spin quantum number"
        found = self.find(_resspt2, _reref, _reweight, _reroot, _respin,
                          _remspt2)
        if not found[_resspt2]:
            return
        # TODO: this needs to be verified as it assumes that there is
        #       only one spin/multiplicity in the system
        #       do not know if this is always the case
        # get the spin quantum number of the system
        spin = float(found[_respin][0][1].split()[-1])
        # get the multiplicity
        mult = np.repeat(int(2*spin + 1), len(found[_remspt2]))
        sspt2 = [] # SSPT2 energies
        mspt2 = [] # MSPT2 energies
        reference = [] # reference energies from the wavefunction file
        weights = [] # reference weight (not sure what it actually is)
        roots = [] # root index
        arr = zip(found[_resspt2], found[_reref], found[_reweight],
                  found[_reroot], found[_remspt2])
        # grab all of the values we are interested in
        for (_, ss), (_, ref), (_, weight), (_, root), (_, ms) in arr:
            sspt2.append(float(ss.split()[-1]))
            reference.append(float(ref.split()[-1]))
            weights.append(float(weight.split()[-1]))
            roots.append(int(re.findall(r'\d+', root)[-1]))
            mspt2.append(float(ms.split()[-1]))
        df = pd.DataFrame.from_dict({'sspt2_au': sspt2, 'mspt2_au': mspt2,
                                     'ras_au': reference, 'weight': weights,
                                     'root': roots, 'mult': mult})
        self.caspt2_energy = df

    def __init__(self, *args, **kwargs):
        super(Output, self).__init__(*args, **kwargs)


class HDFMeta(TypedMeta):
    atom = Atom
    overlap = Overlap
    orbital = Orbital
    momatrix = MOMatrix
    basis_set = BasisSet
    basis_set_order = BasisSetOrder


class HDF(six.with_metaclass(HDFMeta, object)):

    _getter_prefix = 'parse'
    _to_universe = Editor._to_universe

    def to_universe(self):
        return self._to_universe()

    def parse_atom(self):
        ztag = 'CENTER_CHARGES'
        xtag = 'CENTER_COORDINATES'
        ltag = 'CENTER_LABELS'
        self.meta['symmetrized'] = False
        if 'DESYM_CENTER_CHARGES' in self._hdf:
            self.meta['symmetrized'] = True
            ztag = 'DESYM_' + ztag
            xtag = 'DESYM_' + xtag
            ltag = 'DESYM_' + ltag
        Z = pd.Series(self._hdf[ztag]).astype(np.int64)
        xyzs = np.array(self._hdf[xtag])
        labs = pd.Series(self._hdf[ltag]).apply(
                         lambda s: s.decode('utf-8').strip())
        self.atom = pd.DataFrame.from_dict({'Z': Z,
                                            'x': xyzs[:, 0],
                                            'y': xyzs[:, 1],
                                            'z': xyzs[:, 2],
                                            'center': range(len(Z)),
                                            'symbol': Z.map(z2sym),
                                            'label': labs,
                                            'frame': 0})
        if self.meta['symmetrized']:
            symops = {'E': np.array([ 1.,  1.,  1.]),
                      'x': np.array([-1.,  0.,  0.]),
                      'y': np.array([ 0., -1.,  0.]),
                      'z': np.array([ 0.,  0., -1.]),
                     'xy': np.array([-1., -1.,  0.]),
                     'xz': np.array([-1.,  0., -1.]),
                     'yz': np.array([ 0., -1., -1.]),
                    'xyz': np.array([-1., -1., -1.])}
            self.meta['symops'] = symops
            self.atom[['tag', 'symop']] = self.atom['label'].str.extract('(.*):(.*)',
                                                                         expand=True)
            self.atom['tag'] = self.atom['tag'].str.strip()
            self.atom['symop'] = self.atom['symop'].str.strip()
            try:
                self.atom['utag'] = _add_unique_tags(self.atom)
            except ValueError:
                pass


    def parse_basis_set_order(self):
        bso = np.array(self._hdf['BASIS_FUNCTION_IDS'])
        df = {'center': bso[:, 0] - 1,
               'shell': bso[:, 1] - 1,
                   'L': bso[:, 2],
               'frame': 0}
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
        if 'symmetrized' not in self.meta: self.parse_atom()
        key = 'AO_OVERLAP_MATRIX'
        if not self.meta['symmetrized']:
            self.overlap = Overlap.from_column(
                _flat_square_to_triangle(np.array(self._hdf[key])))
        else:
            print('Symmetrized overlap indices not set correctly.')
            self.overlap = Overlap.from_dict({
                'coef': np.array(self._hdf[key]),
                'chi0': 0, 'chi1': 0, 'frame': 0
            })

    def parse_basis_set(self):
        if 'symmetrized' not in self.meta: self.parse_atom()
        bset = np.array(self._hdf['PRIMITIVES'])
        idxs = np.array(self._hdf['PRIMITIVE_IDS'])
        bset = pd.DataFrame.from_dict({
            'alpha': bset[:, 0], 'd': bset[:, 1],
            'center': idxs[:, 0] - 1, 'L': idxs[:, 1],
            'shell': idxs[:, 2] - 1
        })
        self.no_dup = bset
        if self.meta['symmetrized']:
            self.basis_set, atommap = deduplicate_basis_sets(bset)
            self.atom['set'] = self.atom['center'].map(atommap)
        else:
            self.basis_set = bset.rename(columns={'center': 'set'})



    def parse_momatrix(self):
        if 'MO_VECTORS' not in self._hdf: return
        coefs = np.array(self._hdf['MO_VECTORS'])
        try:
            symm = np.array(self._hdf['DESYM_MATRIX'])
            print('Symmetry not supported on HDF yet.')
            return
        except KeyError:
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
        if not os.path.isfile(args[0]):
            print('Argument is likely incorrect file path.')
        self._hdf = h5py.File(args[0], 'r')
        self.meta = {'gaussian': True, 'program': 'molcas'}


def _add_unique_tags(atom):
    """De-duplicates atom identifier in symmetrized calcs."""
    utags = []
    for tag in atom['tag']:
        utag = tag
        while utag in utags:
            utag = ''.join(filter(str.isalpha, utag)) + \
            str(int(''.join(filter(str.isdigit, utag))) + 1)
        utags.append(utag)
    return utags


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
