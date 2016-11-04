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
from exatomic.algorithms.basis import lmap, spher_lml_count, cart_lml_count

z_to_symbol = z_to_symbol()

class Fchk(Editor):

    def parse_atom(self):
        nat = int(self[2].split()[-1])
        found = self.find(_reznum, _reposition, keys_only=True)
        start = found[_reznum][0] + 1
        col = min(len(self[start].split()), nat)
        stop = np.ceil(start + nat / col).astype(np.int64)
        znums = self.pandas_dataframe(start, stop, col).stack()
        symbols = znums.map(z_to_symbol).values
        start = found[_reposition][0] + 1
        col = min(len(self[start].split()), nat * 3)
        stop = np.ceil(start + nat * 3 / col).astype(np.int64)
        pos = self.pandas_dataframe(start, stop, col).stack().values.reshape(nat, 3)
        self.atom = pd.DataFrame.from_dict({'symbol': symbols,
                                            'x': pos[:,0], 'y': pos[:,1], 'z': pos[:,2],
                                            'frame': [0] * len(znums)})

    def parse_gaussian_basis_set(self):
        found = self.find(_rebasdim, _reshelltype, _reprimpershell,
                          _reshelltoatom, _reprimexp, _recontcoef,
                          _recrdshell)
        nbas = int(found[_rebasdim][0][1].split()[-1])
        dim1 = int(found[_reshelltype][0][1].split()[-1])
        dim2 = int(found[_reprimexp][0][1].split()[-1])
        dim3 = int(found[_recrdshell][0][1].split()[-1])
        # Shell types
        start = found[_reshelltype][0][0] + 1
        col = len(self[start].split())
        stop = np.ceil(start + dim1 / col).astype(np.int64)
        shelltypes = self.pandas_dataframe(start, stop, col).stack().values
        # Primitives per shell
        start = found[_reprimpershell][0][0] + 1
        col = len(self[start].split())
        stop = np.ceil(start + dim1 / col).astype(np.int64)
        primpershell = self.pandas_dataframe(start, stop, col).stack().values
        # Shell to atom map
        start = found[_reshelltoatom][0][0] + 1
        col = len(self[start].split())
        stop = np.ceil(start + dim1 / col).astype(np.int64)
        shelltoatom = self.pandas_dataframe(start, stop, col).stack().values
        # Primitive exponents
        start = found[_reprimexp][0][0] + 1
        col = len(self[start].split())
        stop = np.ceil(start + dim2 / col).astype(np.int64)
        primexps = self.pandas_dataframe(start, stop, col).stack().values
        # Contraction coefficients
        start = found[_recontcoef][0][0] + 1
        col = len(self[start].split())
        stop = np.ceil(start + dim2 / col).astype(np.int64)
        contcoefs = self.pandas_dataframe(start, stop, col).stack().values
        # Coordinates of each shell
        start = found[_recrdshell][0][0] + 1
        col = len(self[start].split())
        stop = np.ceil(start + dim3 / col).astype(np.int64)
        crdshells = self.pandas_dataframe(start, stop, col).stack().values
        print('shell types    :', shelltypes.shape, shelltypes.sum())
        print('primpershell   :', primpershell.shape, primpershell.sum())
        print('shelltoatom    :', shelltoatom.shape, shelltoatom.sum())
        print('primexps       :', primexps.shape, primexps.sum())
        print('contcoefs      :', contcoefs.shape, contcoefs.sum())
        print('crdshells      :', crdshells.shape, crdshells.sum())
        self.shelltypes = shelltypes
        self.primpershell = primpershell
        self.shelltoatom = shelltoatom
        self.primexps = primexps
        self.contcoefs = contcoefs
        self.crdshells = crdshells


    def parse_orbital(self):
        found = self.find(_realphaen)

    def parse_momatrix(self):
        found = self.find(_rebasdim, _reindepdim, _reamomatrix)
        nbas = int(found[_rebasdim][0][1].split()[-1])
        try:
            ninp = int(found[_reindepdim][0][1].split()[-1])
        except IndexError:
            ninp = nbas
        ncoef = int(found[_reamomatrix][0][1].split()[-1])
        if nbas * ninp != ncoef:
            raise Exception('Dimensions are inconsistent.')
            return
        start = found[_reamomatrix][0][0] + 1
        col = len(self[start].split())
        stop = np.ceil(start + ncoef / col).astype(np.int64)
        coefs = self.pandas_dataframe(start, stop, col).stack().values
        chis = np.tile(range(nbas), ninp)
        orbitals = np.repeat(range(ninp), nbas)
        frame = np.zeros(ncoef, dtype=np.int64)
        self.momatrix = pd.DataFrame.from_dict({'chi': chis, 'orbital': orbitals,
                                                'coefficient': coefs, 'frame': frame})


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# Atom regex
_reznum = 'Atomic numbers'
_reposition = 'Current cartesian coordinates'

# Basis set regex
_rebasdim = 'Number of basis functions'
_recontdim = 'Number of contracted shells'
_reprimdim = 'Number of primitive shells'
_reshelltype = 'Shell types'
_reprimpershell = 'Number of primitives per shell'
_reshelltoatom = 'Shell to atom map'
_reprimexp = 'Primitive exponents'
_recontcoef = 'Contraction coefficients'
_repcontcoef = 'P\(S=P\) Contraction coefficients'
_recrdshell = 'Coordinates of each shell'

# MOMatrix regex
# also uses _rebasdim
_reindepdim = 'Number of independant functions'
_realphaen = 'Alpha Orbital Energies'
_reamomatrix = 'Alpha MO coefficients'

def _construct_basis_set_order(shelltypes, primpershell, shelltoatom):
    #for sh, p, s
    pass



class Output(Editor):

    def parse_atom(self):
        found = self.find(_regeom01, _regeom02, keys_only=True)
        key = _regeom02 if found[_regeom02] else _regeom01
        starts = np.array(found[key]) + 5
        stop = starts[0]
        while '-------' not in self[stop]: stop += 1
        stops = starts + (stop - starts[0])
        if len(starts) == 1:
            atom = self.pandas_dataframe(starts[0], stops[0], 6)
            atom['frame'] = 0
        else:
            atoms = []
            for i, (start, stop) in enumerate(zip(starts[:-1], stops[:-1])):
                atoms.append(self.pandas_dataframe(start, stop, 6))
                atoms[-1]['frame'] = i
            atom = pd.concat(atoms).reset_index(drop=True)
        atom.drop([0, 2], axis=1, inplace=True)
        atom.columns = ['Z', 'x', 'y', 'z', 'frame']
        atom['x'] *= Length['A', 'au']
        atom['y'] *= Length['A', 'au']
        atom['z'] *= Length['A', 'au']
        atom['set'] = atom.index
        atom['symbol'] = atom['Z'].map(z_to_symbol)
        self.atom = atom

    def parse_gaussian_basis_set(self):
        check = self.regex(_rebas01, flags=re.IGNORECASE)
        if not check:
            print("Must specify 'gfinput' to get full basis set info.")
            return
        found = self.find(_rebas02[:-1], _rebas03)
        stop = found[_rebas02[:-1]][0][0] - 1
        start = stop - 1
        while True:
            if len(self[start].split()) > 4: break
            start -= 1
        self.gaussian_basis_set, setmap = _basis_set(self.pandas_dataframe(start + 1, stop, 4)[[0, 1]])
        self.atom['set'] = self.atom['set'].map(setmap)

    def parse_orbital(self):
        check = self.regex(_reorb055, stop=250, flags=re.IGNORECASE)
        found = self.find(_reorb01, _reorb02, _reorb03, _reorb04)
        if not check:
            symstop = found[_reorb01][0][0] - 1
            symstart = symstop
            while _reorb05 not in self[symstart]:
                symstart -= 1
            symstart += 1
            syms = _resymcom.sub(lambda m: _resympat[m.group(0)],
                                 ' '.join([i.strip() for i in self[symstart:symstop]]))
        aeig = [i[1].replace(_reorb01, '') for i in found[_reorb01]]
        avir = [i[1].replace(_reorb02, '') for i in found[_reorb02]]
        aeig = [float(ln[i]) for i in _orbslice for ln in aeig if ln[i]]
        avir = [float(ln[i]) for i in _orbslice for ln in avir if ln[i]]
        adat = pd.DataFrame(aeig + avir, columns=('energy',))
        adat['spin'] = 0
        bdat = None
        if found[_reorb03]:
            beig = [i[1].replace(_reorb03, '') for i in found[_reorb03]]
            bvir = [i[1].replace(_reorb04, '') for i in found[_reorb04]]
            beig = [float(ln[i]) for i in _orbslice for ln in beig if ln[i]]
            bvir = [float(ln[i]) for i in _orbslice for ln in bvir if ln[i]]
            bdat = pd.DataFrame(beig + bvir, columns=('energy',))
            adat['occupation'] = [1] * len(aeig) + [0] * len(avir)
            bdat['occupation'] = [1] * len(beig) + [0] * len(bvir)
            bdat['spin'] = 1
            if not check:
                bdat['symmetry'] = syms[len(syms)//2:]
                adat['symmetry'] = syms[:len(syms)//2]
        else:
            adat['occupation'] = [2] * len(aeig) + [0] * len(avir)
            if not check:
                adat['symmetry'] = syms
        if bdat is not None:
            dat = pd.concat([adat, bdat]).reset_index(drop=True)
        else:
            dat = adat
        dat['frame'] = 0
        self.orbital = dat

    def parse_momatrix(self):
        check = self.regex(_remomat01, _remomat011, stop=250, flags=re.IGNORECASE)
        if not any((i for i in check)):
            print("Must specify 'pop(full)' or 'pop=full' to get full MOMatrix.")
            return
        found = self.find(_remomat02, _remomat03, _rebas02)
        idx = 0
        dim = 1 if len(found[_remomat02]) == 1 else 2
        starts = np.array([lno for lno, ln in found[_remomat03]]) + 1
        nbas = int(found[_rebas02][0][1].split()[0])
        stops = starts + (starts[1] - starts[0]) - 3
        coefs = np.empty((nbas ** 2, dim), dtype=np.float64)
        cnt = 0
        self.basis_set_order = _basis_set_order(self[starts[0]:stops[0]])
        for start, stop in zip(starts, stops):
            ncol = len(self[start + 2].split()) - 2
            _csv_args['names'] = range(ncol)
            block = '\n'.join([ln[20:] for ln in self[start:stop]])
            block = _rebascom.sub(lambda m: _rebaspat[m.group(0)], block)
            try:
                coefs[cnt*nbas:cnt*nbas + nbas*ncol, idx] = pd.read_csv(
                    StringIO(block), **_csv_args).unstack().values
            except ValueError:
                cnt, idx = 0, 1
                coefs[cnt*nbas:cnt*nbas + nbas*ncol, idx] = pd.read_csv(
                    StringIO(block), **_csv_args).unstack().values
            cnt += ncol
        chis = np.tile(range(nbas), nbas)
        orbitals = np.repeat(range(nbas), nbas)
        frame = np.zeros(nbas * nbas, dtype=np.int64)
        todf = {'coefficient': coefs[:,0], 'orbital': orbitals,
                'chi': chis, 'frame': frame}
        if idx:
            todf['coefficient_beta'] = coefs[:,1]
        self.momatrix = pd.DataFrame.from_dict(todf)

    def parse_basis_set_order(self):
        if hasattr(self, '_basis_set_order'): pass
        else: self.parse_momatrix()


    def parse_basis_set_summary(self):
        nsets = self.gaussian_basis_set['set'].cat.as_ordered().max() + 1
        sets = self.gaussian_basis_set.groupby('set')
        sphchk = self.find(_rebas04)[0][1].split('(')[1].split(')')[0]
        if '5D' and '7F' in sphchk:
            spherical = True
            ml_count = spher_lml_count
        else:
            spherical = False
            ml_count = cart_lml_count
        bss = []
        for (seht, bas), symbol in zip(sets, self.atom['symbol']):
            shfuncs = bas.groupby('shell_function')
            cnt = 0
            for sh, shell in shfuncs:
                L = shell['L'].values[0]
                cnt += ml_count[L]
            bss.append([symbol, '', cnt, cnt, spherical, 0])
        columns = ('tag', 'name', 'function_count', 'func_per_atom', 'spherical', 'frame')
        self.basis_set_summary = pd.DataFrame(bss, columns=columns)

    def parse_frame(self):
        self.frame = compute_frame_from_atom(self.atom)
        found = self.find(_retoten, _realphaelec, _reelecstate)
        self.frame['total_energy'] = [float(i[1].split()[4]) for i in found[_retoten]]
        ae, x, x, be, x, x = found[_realphaelec][0][1].split()
        self.frame['total_electrons'] = int(ae) + int(be)
        try:
            self.frame['state'] = found[_reelecstate][0][1].split()[4].replace('.', '')
        except:
            pass

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
    block = _rebascom.sub(lambda m: _rebaspat[m.group(0)], block)
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
        #for sdx, seht in enumerate(unq):
        #    try:
        #        if np.allclose(seht[chk], sets.get_group(center)[chk]):
        #            setmap[center] = sdx
        #            break
        #    except ValueError:
        #        continue
        #unq.append(sets.get_group(center)[chk])
        #setmap[center] = len(unq) - 1
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
    # TODO : extend to multiple frames
    df['frame'] = 0
    return df, setmap
#def _gaussian_basis_set(chunk):
#        basdat = []
#        start = 0
#        stop = len(chunk)
#        seht, shell, nlns, shlcnt = 0, 0, 0, -1
#        while start < stop:
#            ll = chunk[start].split()
#            if '****' in chunk[start]:
#                seht += 1
#                shlcnt = -1
#                nlns = 0
#                start += 1
#            elif len(ll) == 4:
#                shlcnt += 1
#                nlns = int(ll[1])
#                try:
#                    shell = lmap[ll[0].lower()]
#                except KeyError:
#                    shell = [0, 1]
#                start += 1
#            else:
#                if nlns:
#                    if isinstance(shell, int):
#                        for i in range(nlns):
#                            ll = chunk[start + i].split()
#                            basdat.append([shell, ll[0], ll[1], shlcnt, seht, 0])
#                    elif isinstance(shell, list):
#                        for shl in shell:
#                            for i in range(nlns):
#                                ll = chunk[start + i].split()
#                                basdat.append([shl, ll[0], ll[1 + shl], shlcnt, seht, 0])
#                            shlcnt += 1
#                        shlcnt -= 1
#                    start += nlns
#                else:
#                    start += 1
#        columns = ['L', 'alpha', 'd', 'shell_function', 'set', 'frame']
#        gaussian_basis_set = pd.DataFrame(basdat, columns=columns)
#        gaussian_basis_set['alpha'] = gaussian_basis_set['alpha'].str.replace('D', 'E').astype(np.float64)
#        gaussian_basis_set['d'] = gaussian_basis_set['d'].str.replace('D', 'E').astype(np.float64)
#        return gaussian_basis_set


_csv_args = {'delim_whitespace': True, 'header': None}
_rebaspat = {'D 0': 'D0', 'F 0': 'F0',
             'G 0': 'G0', 'H 0': 'H0', 'I 0': 'I0'}
_rebascom = re.compile('|'.join(_rebaspat.keys()))
_resympat = {'Occupied': '', 'Virtual': '', 'Alpha Orbitals': '',
             'Beta  Orbitals': '', '\(': '', '\)': ''}
_resymcom = re.compile('|'.join(_resympat.keys()))
_resympat['('] = ''
_resympat[')'] = ''
_regeom01 = 'Input orientation'
_regeom02 = 'Standard orientation'
_reorb01 = ' Alpha  occ. eigenvalues -- '
_reorb02 = ' Alpha virt. eigenvalues -- '
_reorb03 = '  Beta  occ. eigenvalues -- '
_reorb04 = '  Beta virt. eigenvalues -- '
_reorb05 = 'Orbital symmetries'
_reorb055 = 'nosym'
_orbslice = [slice(10 * i, 10 * i + 9) for i in range(5)]
_remomat01 = r'POP\(FULL\)'
_remomat011 = r'POP=FULL'
_remomat02 = 'Molecular Orbital Coefficients'
_remomat03 = 'Eigenvalues'
_rebas01 = r'GFINPUT'
_rebas02 = 'basis functions,'
_rebas03 = ' ****'
_rebas04 = 'General basis'
_retoten = 'SCF Done:'
_realphaelec = 'alpha electrons'
_reelecstate = 'The electronic state'
