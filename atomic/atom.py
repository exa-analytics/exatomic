# -*- coding: utf-8 -*-
'''
Atom DataFrame
==========================
'''
from exa import DataFrame
from atomic.twobody import TwoBody


class Atom(DataFrame):
    '''
    Required indexes: frame, atom

    Required columns: symbol, x, y, z
    '''
    __dimensions__ = ['frame', 'atom']
    __columns__ = ['symbol', 'x', 'y', 'z']


class SuperAtom(DataFrame):
    '''
    '''
    __dimensions__ = ['frame', 'superatom']
    __columns__ = ['x', 'y', 'z', 'atom']


def compute_twobody(atoms, periodic=None, k=None, bond_extra=0.45, dmax=13.0, dmin=0.3):
    '''
    Compute two body information given a :class:`~atomic.atom.Atom` dataframe.

    For non-periodic systems the only required argument is the table of atom
    positions and symbols. For periodic systems, at a minimum, the atom and
    periodic cell information dataframes must be provided.

    Bonds are computed semi-empirically and exist if:

    .. math::

        distance(A, B) < covalent\_radius(A) + covalent\_radius(B) + bond\_extra

    Args:
        atoms (:class:`~atomic.atom.Atom`): Table of nuclear positions and symbols
        periodic (:class:`~pandas.DataFrame`): DataFrame of periodic cell dimensions (or False if )
        k (int): Number of distances (per atom) to compute
        bond_extra (float): Extra distance to include when determining bonds (see above)
        dmax (float): Max distance of interest (larger distances are ignored)
        dmin (float): Min distance of interest (smaller distances are ignored)

    Return:
        df (:class:`~atomic.twobody.TwoBody`): Two body property table
    '''
    if periodic:
        return compute_periodic_twobody(atoms, periodic, k, bond_extra, dmax, dmin)
    else:
        raise NotImplementedError()


def compute_periodic_twobody(atoms, periodic, unitize=False, k=None,
                             bond_extra=0.45, dmax=13.0, dmin=0.3):
    '''
    '''
