# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
exnbo Input Generator and Parser
===================================
'''

import numpy as np
import pandas as pd

from exa.relational.isotope import symbol_to_z

symbol_to_Z = symbol_to_z()

import exatomic
from exatomic.editor import Editor as ExatomicEditor
from exatomic.algorithms.basis import (spher_ml_count, cart_ml_count,
                                       spher_lml_count, cart_lml_count,
                                       lmap, lorder, _vec_normalize)
from exatomic.orbital import DensityMatrix
from exatomic.atom import Atom
from exatomic.container import Universe
#from exatomic.frame import minimal_frame

exaver = 'exatomic.v' + exatomic.__version__


_template = """\
$GENNBO NATOMS={nat}    NBAS={nbas}  UPPER  BODM BOHR $END
$NBO BNDIDX NLMO AONBO=W AONLMO=W $END
$COORD
{exaver} -- {name} -- tr[D*S] = {check}
{atom}
$END
$BASIS
 CENTER = {center}
  LABEL = {label}
$END
$CONTRACT
 NSHELL = {nshell:>7}
   NEXP = {nexp:>7}
  NCOMP = {ncomp}
  NPRIM = {nprim}
   NPTR = {nptr}
    EXP = {exponents}
{coefs}
$END
$OVERLAP
{overlap}
$END
$DENSITY
{density}
$END"""

class NBOInput(ExatomicEditor):
    '''
    Base NBO input editor class
    '''
    pass

class NBOInputGenerator(NBOInput):
    '''
    A class that can be constructed by calling the
    function write_nbo_input with an :class:`~exatomic.universe.Universe`
    or similar subclass of :class:`~exatomic.editor.Editor`.  The universe
    must contain the following properties:
    * atom
    * basis_set_summary
    * basis_set_order
    * gaussian_basis_set
    * overlap
    * momatrix
    * density

    See Also:
        :func:`~exnbo.inputs.write_nbo_input`
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



def _format_helper(data, field, ncol, just=True):
    ret = ''
    cnt = 0
    for datum in data:
        ret = ''.join([ret, field.format(datum)])
        cnt += 1
        if cnt == ncol:
            if not just:
                ret += '\n'
            else:
                ret += '\n' + ' ' * 10
            cnt = 0
    return ret


lmltolabels = {
    'spherical': {
        'canonical': {
            0: [1],
            1: list(range(101, 104)),
            2: list(range(251, 256)),
            3: list(range(351, 358)),
            4: list(range(451, 460)),
            5: list(range(551, 562)),
            6: list(range(651, 664))
        },
        'increasingml': {
            0: [1],
            1: [101, 102, 103],
            2: [251, 253, 255, 252, 254],
            3: [357, 355, 353, 351, 352, 354, 356],
            4: [459, 457, 455, 453, 451, 452, 454, 456, 458],
            5: [561, 559, 557, 555, 553, 551, 552, 554, 556, 558, 560],
            6: [663, 661, 659, 657, 655, 653, 651, 652, 654, 656, 658, 660, 662],
        },
        'exgaussian': {
            0: [1],
            1: [101, 102, 103],
            2: [255, 252, 253, 254, 251],
            3: list(range(351, 358)),
            4: list(range(451, 460)),
            5: list(range(551, 562)),
            6: list(range(651, 664)),
        }
    },
    'cartesian': {
        'canonical': {
            0: [1],
            1: list(range(101, 104)),
            2: list(range(201, 207)),
            3: list(range(301, 311)),
            4: list(range(401, 416))
        },
    }
}
## In the general case, the basis function ordering scheme coming from
## different computational codes does not need to be the canonical ordering
## scheme according to the NBO code. See the exgaussian entry as an example
lmltolabels['spherical']['exnwchem'] = lmltolabels['spherical']['increasingml']
lmltolabels['spherical']['exmolcas'] = lmltolabels['spherical']['increasingml']
lmltolabels['cartesian']['exnwchem'] = lmltolabels['cartesian']['canonical']
lmltolabels['cartesian']['exmolcas'] = lmltolabels['cartesian']['canonical']
lmltolabels['cartesian']['exgaussian'] = lmltolabels['cartesian']['canonical']

def _nbas_arrays(universe, nbas, fmt, kwargs):
    center = np.empty(nbas, dtype=np.int64)
    label = np.empty(nbas, dtype=np.int64)
    bases = universe.gaussian_basis_set[abs(universe.gaussian_basis_set['d']) > 0].groupby('set')
    cnt, lbl = 0, 0
    for seht, cent in zip(universe.atom['set'],
                          universe.atom['label'].astype(np.int64) + 1):
        summ = universe.basis_set_summary.ix[seht]
        b = bases.get_group(seht).groupby('shell_function')
        degen = 'cartesian'
        ml_count = cart_ml_count
        ibas = summ.cart_per_atom
        if summ.spherical:
            degen = 'spherical'
            ml_count = spher_ml_count
            ibas = summ.func_per_atom
        for i in range(ibas):
            center[cnt] = cent
            cnt += 1
        if universe.meta['program'] == 'exnwchem':
            for lval, grp in b:
                if len(grp) == 0: continue
                lval = grp['L'].values[0]
                for lab in lmltolabels[degen][universe.meta['program']][lval]:
                    label[lbl] = lab
                    lbl += 1
        elif universe.meta['program'] == 'exmolcas':
            for l, shell in enumerate(lorder):
                chk = 'bas_' + shell
                if chk in summ:
                    repeat = getattr(summ, chk)
                    for i in range(ml_count[shell]):
                        for j in range(repeat):
                            label[lbl] = lmltolabels[degen][universe.meta['program']][l][i]
                            lbl += 1

    kwargs.update({'center': _format_helper(center, fmt, 10),
                    'label': _format_helper(label, fmt, 10)})


def _nshell_arrays(universe, nshell, fmt, kwargs):
    ncomp = np.empty(nshell, dtype=np.int64)
    nprim = np.empty(nshell, dtype=np.int64)
    nptr = np.empty(nshell, dtype=np.int64)
    bases = universe.gaussian_basis_set[abs(universe.gaussian_basis_set['d']) > 0].groupby('set')
    cnt, ptr = 0, 1
    for seht, center in zip(universe.atom['set'], universe.atom['label'].astype(np.int64) + 1):
        summ = universe.basis_set_summary.ix[seht]
        if summ.spherical:
            comp_lookup = spher_lml_count
        else:
            comp_lookup = cart_lml_count
        b = bases.get_group(seht)
        for sh, grp in b.groupby('shell_function'):
            if len(grp) == 0: continue
            ncomp[cnt] = comp_lookup[grp['L'].values[0]]
            nprim[cnt] = grp.shape[0]
            nptr[cnt] = ptr
            ptr += nprim[cnt]
            cnt += 1
    kwargs.update({'ncomp': _format_helper(ncomp, fmt, 10),
                   'nprim': _format_helper(nprim, fmt, 10),
                    'nptr': _format_helper(nptr, fmt, 10)})


def _nexp_arrays(universe, nexp, fmt, kwargs):
    exponents = np.empty(nexp, dtype=np.float64)
    lmax = universe.gaussian_basis_set['L'].cat.as_ordered().max()
    ds = [np.empty(nexp, dtype=np.float64) for i in range(lmax + 1)]
    bases = universe.gaussian_basis_set[abs(universe.gaussian_basis_set['d']) > 0].groupby('set')
    exp = 0
    for seht, center in zip(universe.atom['set'], universe.atom['label'].astype(np.int64) + 1):
        summ = universe.basis_set_summary.ix[seht]
        b = bases.get_group(seht)
        for l, d, exponent in zip(b['L'], b['d'], b['alpha']):
            exponents[exp] = exponent
            for i, shell in enumerate(ds):
                if i == l:
                    ds[i][exp] = d
                else:
                    ds[i][exp] = 0.
            exp += 1
    ds = '\n'.join(['{:>8} ='.format('C' + lorder[i].upper()) + _format_helper(shell, fmt, 4)
                    for i, shell in enumerate(ds)])
    kwargs.update({'exponents': _format_helper(exponents, fmt, 4),
                       'coefs': ds})

def write_nbo_input(universe, fp=None):
    '''
    Args
        universe: universe must have the following attributes
                  atom, basis_set_summary, basis_set,
                  overlap, momatrix, occupations, code (string specifying which comp. code the universe came from)
    Returns
        if fp is not None, write the file to that path,
        else return the editor object.
    '''
    # setup
    fmt = '{:>7}'
    ffmt = '{:> 16.8E}'
    fffmt = '{:> 20.12E}'
    fl = NBOInputGenerator(_template)
    keys = [key.split('}')[0].split(':')[0] for key in _template.split('{')[1:]]
    kwargs = {key: '' for key in keys}
    nat = universe.atom.shape[0]
    nbas = universe.basis_set_summary['function_count'].sum()
    kwargs['nat'] = nat
    kwargs['nbas'] = nbas
    if universe.name is None:
        kwargs['name'] = universe.meta['program']
    else:
        kwargs['name'] = universe.name
    kwargs['exaver'] = exaver

    # coordinates
    universe.atom['Z'] = universe.atom['symbol'].map(symbol_to_Z).astype(np.int64)
    acols = ['Z', 'Zeff', 'x', 'y', 'z'] if 'Zeff' in universe.atom.columns else ['Z', 'Z', 'x', 'y', 'z']
    kwargs['atom'] = universe.atom[acols].to_string(index=None, header=None)

    # nshell and nexp
    select = [i for i in universe.basis_set_summary.columns if 'bas_' in i]
    nshell = (universe.basis_set_summary[select].sum(axis=1) * universe.atom['set'].value_counts()).sum()
    nexp = (universe.gaussian_basis_set[abs(universe.gaussian_basis_set['d']) > 0].groupby('set').apply(
                               lambda x: x.shape[0]) * universe.atom['set'].value_counts()).sum()
    kwargs['nshell'] = nshell
    kwargs['nexp'] = nexp

    _nbas_arrays(universe, nbas, fmt, kwargs)
    _nshell_arrays(universe, nshell, fmt, kwargs)
    _nexp_arrays(universe, nexp, ffmt, kwargs)

    try:
        kwargs['overlap'] = _format_helper(universe.overlap['coefficient'].values, fffmt, 4, just=False)
    except AttributeError:
        kwargs['overlap'] = '{overlap}'
    try:
        kwargs['density'] = _format_helper(universe.density['coefficient'].values, fffmt, 4, just=False)
        kwargs['check'] = np.trace(np.dot(universe.density.square(), universe.overlap.square()))
    except AttributeError:
        kwargs['density'] = '{density}'
        kwargs['check'] = 'density not provided'

    if fp is not None:
        fl.write(fp, **kwargs)
    else:
        return fl.format(inplace=True, **kwargs)
