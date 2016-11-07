# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Gaussian Output Editor
#########################
Editor classes for various types of Gaussian output files
"""
import re
import numpy as np
import pandas as pd
from io import StringIO

from exatomic import Length
from .editor import Editor
from exa.relational.isotope import z_to_symbol
from exatomic.frame import compute_frame_from_atom
from exatomic.algorithms.basis import lmap

z_to_symbol = z_to_symbol()

class Output(Editor):

    def parse_atom(self):
        # Find our data
        found = self.find(_regeom01, _regeom02, keys_only=True)
        # Check if nosymm was specified
        key = _regeom02 if found[_regeom02] else _regeom01
        starts = np.array(found[key]) + 5
        stop = starts[0]
        # Find where the data stops
        while '-------' not in self[stop]: stop += 1
        stops = starts + (stop - starts[0])
        dfs = []
        # Iterate over frames
        for i, (start, stop) in enumerate(zip(starts, stops)):
            atom = self.pandas_dataframe(start, stop, 6)
            atom['frame'] = i
            dfs.append(atom)
        atom = pd.concat(dfs).reset_index(drop=True)
        # Drop the column of atomic type (whatever that is)
        atom.drop([2], axis=1, inplace=True)
        # Name the data
        atom.columns = ['set', 'Z', 'x', 'y', 'z', 'frame']
        # Python is zero-based, sorry FORTRAN
        atom['set'] -= 1
        # Convert to atomic units
        atom['x'] *= Length['A', 'au']
        atom['y'] *= Length['A', 'au']
        atom['z'] *= Length['A', 'au']
        # Map atomic symbols onto Z numbers
        atom['symbol'] = atom['Z'].map(z_to_symbol)
        self.atom = atom

    def parse_gaussian_basis_set(self):
        # First check if gfinput was specified
        check = self.regex(_rebas01, stop=250, flags=re.IGNORECASE)
        if not check: return
        # Find where the basis set is printed
        found = self.find(_rebas02[:-1], _rebas03)
        stop = found[_rebas02[:-1]][0][0] - 1
        start = stop - 1
        # Find where the data actually starts
        while not len(self[start].split()) > 4: start -= 1
        # Call out to the mess that actually parses it
        self.gaussian_basis_set, setmap = _basis_set(
            self.pandas_dataframe(start + 1, stop, 4)[[0, 1]])
        self.atom['set'] = self.atom['set'].map(setmap)

    def parse_orbital(self):
        # Find where our data is
        found = self.regex(_reorb01, _reorb02, _rebas02)
        # Basis dimension
        nbas = int(found[_rebas02][0][1].split()[0])
        # If no orbital energies, quit
        if not found[_reorb01]: return
        # Check if open shell
        os = True if any(('Beta' in ln for lno, ln in found[_reorb01])) else False
        # Find out how big our data is
        # 5 eigenvalues are printed per line
        nrows = len(found[_reorb01]) * 5 // nbas
        nsets = nrows // 2 if os else nrows
        # Allocate a numpy array to store it
        # index is arbitrary for the momentum
        dtypes = [('energy', 'f8'), ('occupation', 'f8'),
                  ('spin', 'i8'), ('index', 'i8')]
        data = np.empty((nbas * nrows,), dtype=dtypes)
        cnt, idx = 0, 0
        idxchk = 2 * nbas if os else nbas
        # Populate and increment accordingly
        for lno, ln in found[_reorb01]:
            for i in _orbslice:
                en = ln[28:][i]
                if en:
                    occ = 1 if 'occ' in ln else 0
                    spn = 0 if 'Alpha' in ln else 1
                    data[cnt] = (en, occ, spn, idx)
                    cnt += 1
                    if cnt == idxchk: idx += 1
        orbital = pd.DataFrame(data)
        orbital['frame'] = 0
        # Symmetry labels
        if found[_reorb02]:
            # Gaussian seems to print out a lot of these blocks
            # try to get a handle on them
            if len(found[_reorb02]) != nsets:
                if nsets == 1:
                    found[_reorb02] = found[_reorb02][-1:]
                elif nsets == 2:
                    found[_reorb02] = found[_reorb02][:1] + found[_reorb02][-1:]
                else:
                    print('Mismatch in eigenvalue and symmetry blocks. '
                          'Continuing without symmetry.')
                    found[_reorb02] = []
            allsyms = []
            match = ['(', 'Orbitals']
            for i, (start, ln) in enumerate(found[_reorb02]):
                # Find the start, stop indices for each block
                while match[0] not in self[start]: start += 1
                stop = start + 1
                while any((i in self[stop] for i in match)): stop += 1
                # Clean up the text block so it is just symmetries
                syms = _resympat.sub(lambda m: _symrep[m.group(0)],
                                     ' '.join([i.strip() for i in
                                     self[start:stop]])).split()
                # cat the syms for each block together
                allsyms += syms
            # Add it to our dataframe
            orbital['symmetry'] = allsyms
        self.orbital = orbital


    def parse_momatrix(self):
        """
        Parses the MO matrix if asked for in the input.

        Note:
            Requires specification of pop(full) or pop(no) or the like.
        """
        if hasattr(self, '_momatrix'): return
        # Check if a full MO matrix was specified in the input
        check = self.regex(_remomat01, stop=250, flags=re.IGNORECASE)
        if not check: return
        # Find approximately where our data is
        found = self.find(_remomat02, _rebas02)
        # Get some dimensions
        ndim = len(found[_remomat02])
        nbas = int(found[_rebas02][0][1].split()[0])
        nblocks = np.int64(np.ceil(nbas / 5))
        # Allocate a big ol' array
        coefs = np.empty((nbas ** 2, ndim), dtype=np.float64)
        # Dynamic column generation hasn't been worked out yet
        colnames = ['coef'] + ['coef' + str(i) for i in range(1, ndim - 1)]
        # Iterate over where the data was found
        # c counts the column in the resulting momatrix table
        for c, (lno, ln) in enumerate(found[_remomat02]):
            start = self.find_next('Eigenvalues', start=lno, keys_only=True) + 1
            stop = start + nbas
            # The basis set order is printed with every chunk of eigenvectors
            if c == 0: self.basis_set_order = _basis_set_order(self[start:stop])
            # Some fudge factors due to extra lines being printed
            space = start - lno - 1
            fnbas = nbas + space
            span = start + fnbas * nblocks
            # Finally get where our chunks are
            starts = np.arange(start, span, fnbas)
            stops = np.arange(stop, span, fnbas)
            stride = 0
            # b counts the blocks of eigenvectors per column in momatrix
            for b, (start, stop) in enumerate(zip(starts, stops)):
                # Number of eigenvectors in this block
                ncol = len(self[start][20:].split())
                _csv_args['names'] = range(ncol)
                # Massage the text so that we can read csv
                block = '\n'.join([ln[20:] for ln in self[start:stop]])
                block = _rebaspat.sub(lambda m: _basrep[m.group(0)], block)
                # Enplacen the resultant unstacked values
                coefs[stride:stride + nbas * ncol, c] = pd.read_csv(
                    StringIO(block), **_csv_args).unstack().values
                stride += nbas * ncol
        # Index chi, phi
        chis = np.tile(range(nbas), nbas)
        orbs = np.repeat(range(nbas), nbas)
        momatrix = pd.DataFrame(coefs, columns=colnames)
        momatrix['chi'] = chis
        momatrix['orbital'] = orbs
        # Frame not really implemented for momatrix
        momatrix['frame'] = 0
        self.momatrix = momatrix

    def parse_basis_set_order(self):
        if hasattr(self, '_basis_set_order'): return
        self.parse_momatrix()


    def parse_frame(self):
        # Get the default frame from the atom table
        self.frame = compute_frame_from_atom(self.atom)
        # Find our data
        found = self.find(_retoten, _realphaelec, _reelecstate)
        self.frame['total_energy'] = [float(i[1].split()[4]) for i in found[_retoten]]
        # We will assume number of electrons doesn't change per frame
        ae, x, x, be, x, x = found[_realphaelec][0][1].split()
        self.frame['N_e'] = int(ae) + int(be)
        self.frame['N_a'] = int(ae)
        self.frame['N_b'] = int(be)
        # Try to get the electronic state but don't try too hard
        try:
            states = []
            for lno, ln in found[_reelecstate]:
                if 'initial' in ln: continue
                states.append(ln.split()[4].replace('.', ''))
            self.frame['state'] = states
        except (IndexError, ValueError):
            pass


    def parse_excitation(self):
        chk = self.find(_retddft, stop=250, keys_only=True)
        if not chk: return
        # Find the data
        found = self.find(_reexcst)
        # Allocate the array
        dtype = [('eV', 'f8'), ('oscstr', 'f8'), ('occ', 'i8'),
                 ('virt', 'i8'), ('kind', 'O'), ('symmetry', 'O')]
        data = np.empty((len(found),), dtype=dtype)
        # Iterate over what we found
        for i, (lno, ln) in enumerate(found):
            # Split this line up into what we want and x
            x, x, x, kind, en, x, x, x, osc, x = line.split()
            # Same for the line right after it
            occ, x, virt, x = self[ln + 1].split()
            # Assign the values
            data[i] = (en, osc.replace('f=', ''), occ, virt) + tuple(kind.split('-'))
        excitation = pd.DataFrame(data)
        # Internal units dictate we should have Hartrees as 'energy'
        excitation['energy'] = excitation['eV'] * Energy['eV', 'Ha']
        # Frame not really implemented here
        excitation['frame'] = 0
        self.excitation = excitation


    def parse_frequency(self):
        found = self.regex(_refreq, stop=250, flags=re.IGNORECASE)
        if not found: return
        # Don't need the input deck or 2 from the summary at the end
        found = self.find(_refreq)[1:-2]
        # Total lines per block minus the unnecessary ones
        span = found[1][0] - found[0][0] - 7
        dfs, fdx = [], 0
        # Iterate over what we found
        for lno, ln in found:
            # Get the frequencies first
            freqs = ln[15:].split()
            nfreqs = len(freqs)
            # Get just the atom displacement vectors
            start = lno + 5
            stop = start + span
            cols = range(2 + 3 * nfreqs)
            df = self.pandas_dataframe(start, stop, ncol=cols)
            # Split up the df and unstack it
            slices = [list(range(2 + i, 2 + 3 * nfreqs, 3)) for i in range(nfreqs)]
            dx, dy, dz = [df[i].unstack().values for i in slices]
            # Generate the appropriate dimensions of other columns
            labels = np.tile(df[0].values, nfreqs)
            zs = np.tile(df[1].values, nfreqs)
            idxs = np.repeat(range(fdx, fdx + nfreqs), df.shape[0])
            fdx += nfreqs
            # Put it all together
            stacked = pd.DataFrame.from_dict({'Z': zs, 'label': labels,
                                             'dx': dx, 'dy': dy, 'dz': dz,
                                             'frequency': freqs, 'freqdx': idxs})
            stacked['symbol'] = stacked['Z'].map(z_to_symbol)
            dfs.append(stacked)
        # Now put all our frequencies together
        frequency = pd.concat(dfs).reset_index(drop=True)
        # Pretty sure displacements are in cartesian angstroms
        # TODO: verify with an external program that vibrational
        #       modes look the same as the ones generated with
        #       this methodology.
        frequency['dx'] *= Length['A', 'au']
        frequency['dy'] *= Length['A', 'au']
        frequency['dz'] *= Length['A', 'au']
        # Frame not really implemented here either
        frequency['frame'] = 0
        self.frequency = frequency


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



def _basis_set_order(chunk):
    centag = []
    for ln in chunk:
        ll = ln.split()
        if len(ll) == 9:
            cen = ll[1]
            tag = ll[2]
        centag.append((cen, tag))
    basord = pd.DataFrame(centag, columns=('center', 'tag'))
    basord['center'] = basord['center'].astype(np.int64)
    basord['center'] -= 1
    block = '\n'.join([ln[10:] for ln in chunk])
    block = _rebaspat.sub(lambda m: _basrep[m.group(0)], block)
    basord['type'] = [ln.split()[0] if len(ln.split()) == 6 else ln.split()[1] for ln in block.splitlines()]
    split = basord['type'].str.extract(r"([0-9]{1,})([A-z])(.*)", expand=True)
    shfuncs = []
    shfunc = 0
    prevcen = basord['center'].values[0]
    prevtyp = basord['type'].values[0]
    for cen, typ in zip(basord['center'].values, basord['type'].values):
        if not cen == prevcen:
            shfunc = -1
        if prevtyp[0] == typ[0]:
            if prevtyp[1] == typ[1]:
                pass
            else:
                shfunc += 1
        else:
            shfunc += 1
        shfuncs.append(shfunc)
        prevcen = cen
        prevtyp = typ
    basord['shell'] = shfuncs
    basord['L'] = split[1].str.lower().map(lmap)
    basord['L'] = basord['L'].astype(np.int64)
    basord['ml'] = split[2]
    basord['ml'].update(split[2].map({'': 0, 'X': 1, 'Y': -1, 'Z': 0}))
    basord['ml'] = basord['ml'].astype(np.int64)
    return basord

def _basis_set(raw):
    lmap['e'] = 2
    raw[0] = raw[0].str.replace('D', 'E')
    raw[1] = raw[1].str.replace('D', 'E')
    center, shell, l, cnt = -1, -1, 0, 0
    dtype = [('alpha', 'f8'), ('d', 'f8'), ('center', 'i8'),
             ('shell', 'i8'), ('L', 'i8')]
    keep = np.empty((raw.shape[0],), dtype=dtype)
    for alpha, d in zip(raw[0], raw[1]):
        try:
            int(alpha)
            center += 1
            shell = -1
        except ValueError:
            if alpha.isalpha():
                shell += 1
                l = lmap[alpha.lower()]
            try:
                keep[cnt] = (alpha, d, center, shell, l)
                cnt += 1
            except ValueError:
                pass
    keep = pd.DataFrame(keep[:cnt])
    centers = keep['center'].unique()
    chk = ['alpha', 'd']
    sets = keep.groupby('center')
    unq = [sets.get_group(0)]
    setmap = {0: 0}
    for center in centers[1:]:
        try:
            if np.allclose(unq[-1][chk], sets.get_group(center)[chk]):
                setmap[center] = len(unq) - 1
                continue
        except ValueError:
            pass
        unq.append(keep[keep['center'] == center])
        setmap[center] = len(unq) - 1
    df = pd.concat(unq).reset_index(drop=True)
    df.rename(columns={'center': 'set'}, inplace=True)
    # TODO : extend to multiple frames or assume a single basis set?
    df['frame'] = 0
    return df, setmap


_csv_args = {'delim_whitespace': True, 'header': None}
# Atom flags
_regeom01 = 'Input orientation'
_regeom02 = 'Standard orientation'
# Orbital flags
_reorb01 = '(?=Alpha|Beta).*(?=occ|virt)'
_reorb02 = 'Orbital symmetries'
_orbslice = [slice(10 * i, 10 * i + 9) for i in range(5)]
_symrep = {'Occupied': '', 'Virtual': '', 'Alpha Orbitals': '',
           'Beta  Orbitals': '', '\(': '', '\)': ''}
_resympat = re.compile('|'.join(_symrep.keys()))
#_resympat['('] = ''
#_resympat[')'] = ''
# MOMatrix flags
_remomat01 = r'pop.*(?=full|no)'
_remomat02 = 'Orbital Coefficients'
# Basis flags
_rebas01 = r'gfinput'
_rebas02 = 'basis functions,'
_rebas03 = ' ****'
_rebas04 = 'General basis'
_basrep = {'D 0': 'D0', 'F 0': 'F0',
           'G 0': 'G0', 'H 0': 'H0', 'I 0': 'I0'}
_rebaspat = re.compile('|'.join(_basrep.keys()))
# Frame flags
_retoten = 'SCF Done:'
_realphaelec = 'alpha electrons'
_reelecstate = 'The electronic state'
# Frequency flags
_refreq = 'Freq'
# TDDFT flags
_retddft = 'TD'
_reexcst = 'Excited State'

#class Fchk(Editor):
#
#    def parse_atom(self):
#        nat = int(self[2].split()[-1])
#        found = self.find(_reznum, _reposition, keys_only=True)
#        start = found[_reznum][0] + 1
#        col = min(len(self[start].split()), nat)
#        stop = np.ceil(start + nat / col).astype(np.int64)
#        znums = self.pandas_dataframe(start, stop, col).stack()
#        symbols = znums.map(z_to_symbol).values
#        start = found[_reposition][0] + 1
#        col = min(len(self[start].split()), nat * 3)
#        stop = np.ceil(start + nat * 3 / col).astype(np.int64)
#        pos = self.pandas_dataframe(start, stop, col).stack().values.reshape(nat, 3)
#        self.atom = pd.DataFrame.from_dict({'symbol': symbols,
#                                            'x': pos[:,0], 'y': pos[:,1], 'z': pos[:,2],
#                                            'frame': [0] * len(znums)})
#
#    def parse_gaussian_basis_set(self):
#        found = self.find(_rebasdim, _reshelltype, _reprimpershell,
#                          _reshelltoatom, _reprimexp, _recontcoef,
#                          _recrdshell)
#        nbas = int(found[_rebasdim][0][1].split()[-1])
#        dim1 = int(found[_reshelltype][0][1].split()[-1])
#        dim2 = int(found[_reprimexp][0][1].split()[-1])
#        dim3 = int(found[_recrdshell][0][1].split()[-1])
#        # Shell types
#        start = found[_reshelltype][0][0] + 1
#        col = len(self[start].split())
#        stop = np.ceil(start + dim1 / col).astype(np.int64)
#        shelltypes = self.pandas_dataframe(start, stop, col).stack().values
#        # Primitives per shell
#        start = found[_reprimpershell][0][0] + 1
#        col = len(self[start].split())
#        stop = np.ceil(start + dim1 / col).astype(np.int64)
#        primpershell = self.pandas_dataframe(start, stop, col).stack().values
#        # Shell to atom map
#        start = found[_reshelltoatom][0][0] + 1
#        col = len(self[start].split())
#        stop = np.ceil(start + dim1 / col).astype(np.int64)
#        shelltoatom = self.pandas_dataframe(start, stop, col).stack().values
#        # Primitive exponents
#        start = found[_reprimexp][0][0] + 1
#        col = len(self[start].split())
#        stop = np.ceil(start + dim2 / col).astype(np.int64)
#        primexps = self.pandas_dataframe(start, stop, col).stack().values
#        # Contraction coefficients
#        start = found[_recontcoef][0][0] + 1
#        col = len(self[start].split())
#        stop = np.ceil(start + dim2 / col).astype(np.int64)
#        contcoefs = self.pandas_dataframe(start, stop, col).stack().values
#        # Coordinates of each shell
#        start = found[_recrdshell][0][0] + 1
#        col = len(self[start].split())
#        stop = np.ceil(start + dim3 / col).astype(np.int64)
#        crdshells = self.pandas_dataframe(start, stop, col).stack().values
#        print('shell types    :', shelltypes.shape, shelltypes.sum())
#        print('primpershell   :', primpershell.shape, primpershell.sum())
#        print('shelltoatom    :', shelltoatom.shape, shelltoatom.sum())
#        print('primexps       :', primexps.shape, primexps.sum())
#        print('contcoefs      :', contcoefs.shape, contcoefs.sum())
#        print('crdshells      :', crdshells.shape, crdshells.sum())
#        self.shelltypes = shelltypes
#        self.primpershell = primpershell
#        self.shelltoatom = shelltoatom
#        self.primexps = primexps
#        self.contcoefs = contcoefs
#        self.crdshells = crdshells
#
#
#    def parse_orbital(self):
#        found = self.find(_realphaen)
#
#    def parse_momatrix(self):
#        found = self.find(_rebasdim, _reindepdim, _reamomatrix)
#        nbas = int(found[_rebasdim][0][1].split()[-1])
#        try:
#            ninp = int(found[_reindepdim][0][1].split()[-1])
#        except IndexError:
#            ninp = nbas
#        ncoef = int(found[_reamomatrix][0][1].split()[-1])
#        if nbas * ninp != ncoef:
#            raise Exception('Dimensions are inconsistent.')
#            return
#        start = found[_reamomatrix][0][0] + 1
#        col = len(self[start].split())
#        stop = np.ceil(start + ncoef / col).astype(np.int64)
#        coefs = self.pandas_dataframe(start, stop, col).stack().values
#        chis = np.tile(range(nbas), ninp)
#        orbitals = np.repeat(range(ninp), nbas)
#        frame = np.zeros(ncoef, dtype=np.int64)
#        self.momatrix = pd.DataFrame.from_dict({'chi': chis, 'orbital': orbitals,
#                                                'coefficient': coefs, 'frame': frame})
#
#
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#
## Atom regex
#_reznum = 'Atomic numbers'
#_reposition = 'Current cartesian coordinates'
#
## Basis set regex
#_rebasdim = 'Number of basis functions'
#_recontdim = 'Number of contracted shells'
#_reprimdim = 'Number of primitive shells'
#_reshelltype = 'Shell types'
#_reprimpershell = 'Number of primitives per shell'
#_reshelltoatom = 'Shell to atom map'
#_reprimexp = 'Primitive exponents'
#_recontcoef = 'Contraction coefficients'
#_repcontcoef = 'P\(S=P\) Contraction coefficients'
#_recrdshell = 'Coordinates of each shell'
#
## MOMatrix regex
## also uses _rebasdim
#_reindepdim = 'Number of independant functions'
#_realphaen = 'Alpha Orbital Energies'
#_reamomatrix = 'Alpha MO coefficients'
#
#def _construct_basis_set_order(shelltypes, primpershell, shelltoatom):
#    #for sh, p, s
#    pass
#
#
