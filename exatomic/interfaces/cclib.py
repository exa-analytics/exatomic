# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Interface to `cclib`_
#######################

.. _cclib: https://cclib.github.io/
"""
from ast import literal_eval
from exa.util.units import Length, Energy
from exatomic.algorithms.basis import lmap
from exatomic.base import z2sym


def universe_from_cclib(ccobj):
    data = ccobj.__dict__
    atom = parse_ccobj_atom(data)
    if atom is None:
        print('Cannot convert ccobj to exatomic.Universe')
        return
    dfs = {'atom': atom}
    orbital = parse_ccobj_orbital(data)
    if orbital is not None: dfs['orbital'] = orbital
    momatrix = parse_ccobj_momatrix(data)
    if momatrix is not None: dfs['momatrix'] = momatrix
    bso = parse_ccobj_basis_set_order(data)
    if bso is not None: dfs['basis_set_order'] = bso
    basis_set = parse_ccobj_gaussian_basis_set(data)
    if basis_set is not None: dfs['basis_set'] = basis_set
    return dfs

def parse_ccobj_atom(data):
    """Gets atom table information from ccobj."""
    if 'atomcoords' not in data or 'atomnos' not in data: return
    x = data['atomcoords'][:,:,0].flatten() * Length['Angstrom', 'au']
    y = data['atomcoords'][:,:,1].flatten() * Length['Angstrom', 'au']
    z = data['atomcoords'][:,:,2].flatten() * Length['Angstrom', 'au']
    Z = data['atomnos']
    nframe = len(z) // len(Z)
    frame = np.repeat(range(nframe), len(z) // nframe)
    Z = np.tile(Z, nframe)
    atom = pd.DataFrame.from_dict({'x': x, 'y': y, 'z': z,
                                   'Z': Z, 'frame': frame})
    atom['symbol'] = atom['Z'].map(z2sym)
    return atom

def parse_ccobj_orbital(data):
    if not all(i in data for i in ('nmo', 'homos', 'moenergies', 'mosyms')):
        return
    nbas = data['nmo']
    os = len(data['homos']) == 2
    dim = nbas * len(data['moenergies'])
    occs = np.empty(dim, dtype=np.float64)
    ens = np.empty(dim, dtype=np.float64)
    syms = np.empty(dim, dtype=object)
    for i, (homo, moens, sym) in enumerate(zip(data['homos'],
                                               data['moenergies'],
                                               data['mosyms'])):
        occs[:homo] = 1 if os else 2
        occs[homo:] = 0
        ens[i * nbas:i * nbas + nbas] = moens * Energy['eV', 'Ha']
        syms[i * nbas: i * nbas + nbas] = sym
    orbital = pd.DataFrame.from_dict({'occupation': occs, 'energy': ens,
                                      'symmetry': syms})
    orbital['frame'] = 0
    orbital['group'] = 0
    return orbital

def parse_ccobj_momatrix(data):
    if 'nmo' not in data or 'mocoeffs' not in data: return
    chis = np.tile(range(data['nmo']), data['nmo'])
    orbs = np.repeat(range(data['nmo']), data['nmo'])
    frame = np.zeros(data['nmo'] ** 2)
    momatrix = pd.DataFrame.from_dict({'chi': chis, 'orbital': orbs,
                                       'frame': frame})
    for i, coefs in enumerate([coef.flatten() for coef in data['mocoeffs']]):
        col = 'coef' if not i else 'coef{}'.format(i)
        momatrix[col] = coefs
    return momatrix

def parse_ccobj_basis_set_order(data, code='gaussian'):
    """Gets basis set information from ccobj. currently
    only works for Gaussian basis set formats."""
    if 'aonames' not in data: return
    if code == 'gaussian':
        split0 = r"([A-z])([0-9]*)"
        split1 = r"([0-9]*)([A-z]{1})(.*)"
        bso = pd.DataFrame([bf.split('_') for bf in data['aonames']])
        bso[['tag', 'center']] = bso[0].str.extract(split0, expand=True)
        bso['center'] = bso['center'].astype(np.int64) - 1
        bso[['n', 'L', 'ml']] = bso[1].str.extract(split1, expand=True)
        bso['L'] = bso['L'].str.lower().map(lmap).astype(np.int64)
        bso['ml'].update(bso['ml'].map({'': 0, 'X': 1, 'Y': -1, 'Z': 0}))
        shfns = []
        shl, pcen, pl, pn = -1, -1, -1, -1
        for cen, n, l in zip(bso['center'], bso['n'], bso['L']):
            if not pcen == cen: shl = -1
            if (not pl == l) or (not pn == n): shl += 1
            shfns.append(shl)
            pcen, pl, pn = cen, l, n
        bso['shell'] = shfns
        bso.drop([0, 1, 'n'], axis=1, inplace=True)
        return bso

def parse_ccobj_gaussian_basis_set(data, code='gaussian'):
    if 'gbasis' not in data: return
    seht, shell = 0, 0
    dat = {'alpha': [], 'd': [], 'shell': [], 'L': [], 'seht': []}
    for center in data['gbasis']:
        for angmom, contracted in center:
            for alpha, d in contracted:
                #L = lmap[angmom.lower()]
                for key in data.keys():
                    dat[key].append(literal_eval(key))
            shell += 1
        seht += 1
        shell = 0
    dat['set'] = data.pop('seht')
    return pd.DataFrame.from_dict(dat)
