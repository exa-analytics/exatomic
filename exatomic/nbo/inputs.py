# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
exnbo Input Generator and Parser
###################################
"""
import six
import numpy as np
import pandas as pd
from exa import TypedMeta
from exatomic import __version__
from exatomic.base import z2sym
from .editor import Editor
from exatomic.core.atom import Atom
from exatomic.core.frame import Frame
from exatomic.core.basis import BasisSetOrder, Overlap
from exatomic.core.orbital import DensityMatrix, MOMatrix
from exatomic.algorithms.basis import lorder, cart_lml_count, spher_lml_count
from exatomic.algorithms.orbital_util import _check_column
from exatomic.base import sym2z
from itertools import combinations_with_replacement as cwr


from exatomic.algorithms.numerical import _flat_square_to_triangle, _tri_indices



_exaver = 'exatomic.v' + __version__


_header = """\
$GENNBO NATOMS={nat}    NBAS={nbas}  UPPER  BODM BOHR $END
$NBO BNDIDX NLMO AONBO=W AONLMO=W $END
$COORD
{exaver} -- {name} -- tr[P*S] = {check}
{atom}
$END
$BASIS
 CENTER = {center}
  LABEL = {label}
$END
$CONTRACT
 NSHELL = {nshell:>7}
   NEXP = {nexpnt:>7}
  NCOMP = {ncomps}
  NPRIM = {nprims}
   NPTR = {npntrs}
    EXP = {expnts}
{coeffs}
$END"""


_matrices = """
$OVERLAP
{overlap}
$END
$DENSITY
{density}
$END"""


def _nbo_labels():
    """Generate dataframes of NBO label, L, and ml or l, m, n."""
    sph = pd.DataFrame([(L, m) for L in range(7) for m in range(-L, L + 1)],
                       columns=('L', 'ml'))

    # See the NBO 6.0 manual for more details
    # This is the basis function labeling scheme
    # In order of increasing ml from most negative
    # to most positive in the same order as the
    # results from the solid_harmonics code.
    sph['label'] = [1,   101, 102, 103, 251, 253, 255,
                    252, 254, 357, 355, 353, 351, 352,
                    354, 356, 459, 457, 455, 453, 451,
                    452, 454, 456, 458, 561, 559, 557,
                    555, 553, 551, 552, 554, 556, 558,
                    560, 663, 661, 659, 657, 655, 653,
                    651, 652, 654, 656, 658, 660, 662]
    Ls, ls, ms, ns, label = [], [], [], [], []
    # Even NBO 6.0 doesn't support cartesian basis
    # functions with an l value greater than g functions
    for i in range(5):
        start = i * 100 + 1
        label += list(range(start, start + cart_lml_count[i]))
        car = [''.join(i) for i in list(cwr('xyz', i))]
        Ls += [i for k in car]
        ls += [i.count('x') for i in car]
        ms += [i.count('y') for i in car]
        ns += [i.count('z') for i in car]
    car = pd.DataFrame({'L': Ls, 'l': ls, 'm': ms,
                        'n': ns, 'label': label})
    return sph, car

spher, cart = _nbo_labels()

def _get_labels(Ls, mls=None, ls=None, ms=None, ns=None):
    """Get the NBO labels corresponding to L, (ml | l, m, n)."""
    if mls is not None:
        return [spher[(spher['L'] == l) &
                      (spher['ml'] == ml)]['label'].iloc[0]
                      for l, ml in zip(Ls, mls)]
    if ls is not None:
        return [cart[(cart['L'] == L) &
                     (cart['l'] == l) &
                     (cart['m'] == m) &
                     (cart['n'] == n)]['label'].iloc[0]
                     for L, l, m, n in zip(Ls, ls, ms, ns)]

def _clean_coeffs(arr, width=16, decimals=6):
    """Call _clean_to_string for each shell."""
    # Format C(shell) for coeffs
    ls = ['     {} = '.format('C' + l.upper()) for l in lorder]
    # Clean to string by shell
    dat = [''.join([l, _clean_to_string(ar, decimals=decimals, width=width), '\n'])
           for l, ar in zip(ls, arr)]
    # Return the whole minus the last line break
    return ''.join(dat)[:-1]

def _clean_to_string(arr, ncol=4, width=16, decimals='', just=True):
    """Convert a numerical array into nicely formatted text block."""
    # Justify the data arrays with the tags in the template
    pad = ' ' * 10 if just else ''
    # Some flexibility in how this function handles int/floats
    dec = '.' + str(decimals) + 'E' if decimals else decimals
    # A format string for the numbers in the array
    fmt = ''.join(['{:>', str(width), dec, '}'])
    # The formmatted array with tabs and new line breaks
    dat = [''.join(['\n', pad, fmt.format(a)]) if not i % ncol and i > 0
           else fmt.format(a) for i, a in enumerate(arr)]
    return ''.join(dat)

def _obtain_arrays(uni):
    """Get numerical arrays of information from a universe."""
    # Get number of functions by shell
    shells = uni.basis_set.functions_by_shell()
    # This is how many times each L value shows up
    shlcnt = shells.index.get_level_values(0)
    # Add subshells for each time L shows up
    shells = shells.groupby(shlcnt).apply(lambda x: x.sum())

    # Group our basis sets, will be used later
    bases = uni.basis_set[np.abs(uni.basis_set['d']) > 0].groupby('set')
    # Exponents per basis set
    expnts = bases.apply(lambda x: x.shape[0])
    # mapped onto the atoms with each basis set
    if uni.basis_set_order.irrep.max():
        raise Exception("Need to figure out basis desymmetrization.")
    else:
        center = uni.basis_set_order['center'].values.copy()
    kwargs = {'center': center,
              'nshell': uni.atom['set'].map(shells).sum(),
              'nexpnt': uni.atom['set'].map(expnts).sum(),
                   'L': uni.basis_set_order['L'].values}
    kwargs['center'] += 1
    nshell = kwargs['nshell']
    nexpnt = kwargs['nexpnt']

    if uni.meta['spherical']:
        # Spherical basis set
        kwargs.update({'ml': uni.basis_set_order['ml'].values})
        lml_count = spher_lml_count
    else:
        # Cartesian basis set
        kwargs.update({'l': uni.basis_set_order['l'].values,
                       'm': uni.basis_set_order['m'].values,
                       'n': uni.basis_set_order['n'].values})
        lml_count = cart_lml_count

    # For the NBO specicific arrays
    lmax = uni.basis_set['L'].cat.as_ordered().max()
    # ---- There are 3 that are length nshell
    # The number of components per basis function (l degeneracy)
    ncomps = np.empty(nshell, dtype=np.int64)
    # The number of primitive functions per basis function
    nprims = np.empty(nshell, dtype=np.int64)
    # The pointers in the arrays above for each basis funciton
    npntrs = np.empty(nshell, dtype=np.int64)
    # ---- And 2 that are length nexpnt
    # The total number of exponents in the basis set
    expnts = np.empty(nexpnt, dtype=np.float64)
    # The contraction coefficients within the basis set
    ds = np.empty((lmax + 1, nexpnt), dtype=np.float64)
    # The following algorithm must be generalized
    # and simplified by either some bound methods
    # on basis_set attributes
    cnt, ptr, xpc = 0, 1, 0
    for seht in uni.atom['set']:
        b = bases.get_group(seht)
        #for sh, grp in b.groupby('shell'):
        for _, grp in b.groupby('shell'):
            if len(grp) == 0: continue
            ncomps[cnt] = lml_count[grp['L'].values[0]]
            nprims[cnt] = grp.shape[0]
            npntrs[cnt] = ptr
            ptr += nprims[cnt]
            cnt += 1
        for l, d, exp in zip(b['L'], b['d'], b['alpha']):
            expnts[xpc] = exp
            #   i, ang
            for i, _ in enumerate(ds):
                ds[i][xpc] = d if i == l else 0
            xpc += 1
    kwargs.update({'nprims': nprims, 'ncomps': ncomps,
                   'npntrs': npntrs, 'expnts': expnts,
                   'coeffs': ds})
    return kwargs




class InpMeta(TypedMeta):
    atom = Atom
    basis_set_order = BasisSetOrder
    momatrix = MOMatrix
    frame = Frame
    overlap = Overlap

class Input(six.with_metaclass(InpMeta, Editor)):

    def parse_atom(self):
        start = 0
        while True:
            try:
                ln = self[start].split()
                _, _ = int(ln[0]), float(ln[2])
                break
            except:
                start += 1
        stop = self.find_next("$END", start=start, keys_only=True)
        cols = ('Z', 'Zeff', 'x', 'y', 'z')
        atom = self.pandas_dataframe(start, stop, cols)
        atom['frame'] = 0
        atom['symbol'] = atom['Z'].map(z2sym)
        self.atom = atom

    def parse_basis_set_order(self):
        start = self.find_next("$BASIS", keys_only=True) + 1
        stop = self.find_next("$END", start=start, keys_only=True)
        arr = ' '.join([' '.join(ln.strip().split()) for ln in self[start:stop]])
        arr = np.fromstring(arr.replace('CENTER=', '').replace('LABEL=', ''),
                            sep=' ', dtype=np.int64)
        nbas = arr.shape[0] // 2
        bso = pd.DataFrame.from_dict({'center': arr[:nbas] - 1,
                                      'label': arr[nbas:],
                                      'frame': 0, 'shell': 0})
        tmp = spher.set_index('label')
        bso['L'] = bso['label'].map(tmp['L'].to_dict())
        bso['ml'] = bso['label'].map(tmp['ml'].to_dict())
        self.basis_set_order = bso

    def parse_overlap(self):
        self._init()
        start = self.find_next("$OVERLAP", keys_only=True) + 1
        stop = self.find_next("$END", start=start, keys_only=True)
        ovl = self.pandas_dataframe(start, stop,
                                    range(len(self[start].split()))).stack().values
        if ovl.shape[0] != self._nbas * (self._nbas + 1) // 2:
            ovl = _flat_square_to_triangle(ovl.stack().values)
        chi0, chi1 = _tri_indices(ovl)
        self.overlap = Overlap.from_dict({'coef': ovl, 'chi0': chi0,
                                          'chi1': chi1, 'frame': 0})

    def parse_momatrix(self):
        arrs = {}
        kws = ['$FOCK', '$KINETIC', '$DIPOLE', '$DENSITY', '$LCAOMO']
        found = self.find(*kws, keys_only=True)
        nbas = None
        for find, start in found.items():
            start = start[0] + 1
            stop = self.find_next('$END', start=start, keys_only=True)
            arr = self.pandas_dataframe(start, stop, range(len(self[start].split())))
            arrs[find] = arr.stack().values
            if nbas is None: nbas = np.int64(np.sqrt(arrs[find].shape[0]))
        nbas = np.int64(np.sqrt(min((arr.shape[0] for arr in arrs.values()))))
        if nbas != self._nbas:
            print('Warning: matrices may be triangular or square.')
        nbas2 = nbas ** 2
        for key in kws:
            arr = arrs[key]
            nkey = key.strip('$').lower()
            if arr.shape[0] == nbas2:
                arrs[nkey] = arr
            else:
                rpt = arr.shape[0] // (nbas2)
                for i in range(rpt):
                    arrs[nkey + str(i)] = arr[i * nbas2 : i * nbas2 + nbas2]
            del arrs[key]
        arrs['chi'] = np.tile(range(nbas), nbas)
        arrs['orbital'] = np.repeat(range(nbas), nbas)
        arrs['frame'] = 0
        self.momatrix = MOMatrix.from_dict(arrs)

    def _init(self):
        find = self.find_next('NBAS', keys_only=True)
        self._nbas = int(self[find].split('NBAS=')[1].split()[0])


    @classmethod
    def from_universe(cls, uni, mocoefs=None, orbocc=None, name=''):
        """
        Generate an NBO input from a properly populated universe.
        uni must have atom, basis_set, basis_set_order, overlap,
        momatrix and orbital information.

        Args
            uni (:class:`~exatomic.container.Universe`): containing the above attributes
            mocoefs (str): column name of MO coefficients to use in uni.momatrix
            orbocc (str): column name of occupations in uni.orbital
            name (str): prefix of file name to write

        Returns
            editor (:class:`~exatomic.nbo.Input`)
        """
        for attr in ['overlap', 'momatrix', 'orbital']:
            if not hasattr(uni, attr):
                raise Exception('uni must have "{}" attribute.'.format(attr))
        mocoefs = _check_column(uni, 'momatrix', mocoefs)
        orbocc = mocoefs if orbocc is None and mocoefs != 'coef' else orbocc
        orbocc = _check_column(uni, 'orbital', orbocc)
        p1 = 'Assuming "{}" vector corresponds to "{}" matrix'
        print(p1.format(orbocc, mocoefs))

        kwargs = _obtain_arrays(uni)
        kwargs.update({'exaver': _exaver, 'name': name, 'check': '',
                       'nat': kwargs['center'].max(),
                       'nbas': len(kwargs['center'])})
        columns = ('Z', 'Z', 'x', 'y', 'z')
        if 'Zeff' in uni.atom.columns:
            columns = ('Z', 'Zeff', 'x', 'y', 'z')
        if 'Z' not in uni.atom.columns:
            uni.atom['Z'] = uni.atom.symbol.map(sym2z)
        kwargs['atom'] = uni.atom.to_xyz(columns=columns)
        # Assign appropriate NBO basis function labels
        if 'ml' in kwargs:
            labargs = {'mls': kwargs['ml']}
        else:
            labargs = {'ls': kwargs['l'],
                       'ms': kwargs['m'],
                       'ns': kwargs['n']}
        kwargs['label'] = _get_labels(kwargs['L'], **labargs)
        # Clean the arrays to strings for a text input file
        kwargs['label'] = _clean_to_string(kwargs['label'], ncol=10, width=5)
        kwargs['center'] = _clean_to_string(kwargs['center'], ncol=10, width=5)
        kwargs['ncomps'] = _clean_to_string(kwargs['ncomps'], ncol=10, width=5)
        kwargs['nprims'] = _clean_to_string(kwargs['nprims'], ncol=10, width=5)
        kwargs['npntrs'] = _clean_to_string(kwargs['npntrs'], ncol=10, width=5)
        kwargs['expnts'] = _clean_to_string(kwargs['expnts'], decimals=10, width=18)
        kwargs['coeffs'] = _clean_coeffs(kwargs['coeffs'])

        matargs = {'overlap': '', 'density': ''}
        margs = {'decimals': 15, 'width': 23, 'just': False}
        if 'irrep' in uni.overlap.columns:
            matargs['overlap'] = _clean_to_string(uni.overlap.square().values.flatten(),
                                                  **margs)
        else:
            matargs['overlap'] = _clean_to_string(uni.overlap['coef'].values,
                                                  **margs)
        d = DensityMatrix.from_momatrix(uni.momatrix,
                                        uni.orbital[orbocc].values,
                                        mocoefs=mocoefs)
        #if 'irrep' in uni.overlap.columns:
        matargs['density'] = _clean_to_string(d['coef'].values, **margs)
        kwargs['check'] = np.trace(np.dot(d.square(), uni.overlap.square()))
        print('If {:.8f} is not the correct number of electrons,\n'
              '"{}" vector in uni.orbital may not correspond to "{}" matrix\n'
              'in uni.momatrix'.format(kwargs['check'], orbocc, mocoefs))
        return cls(_header.format(**kwargs) + _matrices.format(**matargs))

    def __init__(self, *args, **kwargs):
        super(Input, self).__init__(*args, **kwargs)
