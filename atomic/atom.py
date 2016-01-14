# -*- coding: utf-8 -*-
'''
Atom DataFrame
==========================
'''
from exa import DataFrame
from exa.errors import ColumnError
from atomic.twobody import TwoBody


class Atom(DataFrame):
    '''
    Absolute positions of atoms and their symbol.

    Required indexes: frame, atom

    Required columns: symbol, x, y, z
    '''
    __indices__ = ['frame', 'atom']
    __columns__ = ['symbol', 'x', 'y', 'z']


class PBCAtom(DataFrame):
    '''
    Periodic (boundary conditions) atoms positions with a foriegn key point
    back to the absolute positions table.
    '''
    __indices__ = ['frame', 'pbc_atom']
    __columns__ = ['x', 'y', 'z', 'atom']


class VisualAtom(DataFrame):
    '''
    Special positions for atoms (not the actual absolute positions) that
    are used in combination with the Atom table to generate representative
    animations.
    '''
    __indices__ = ['frame', 'vis_atom']
    __columns__ = ['x', 'y', 'z', 'atom']


def compute_twobody(universe, k=None, bond_extra=0.45, dmax=13.0, dmin=0.3):
    '''
    Compute two body information given a :class:`~atomic.atom.Atom` dataframe.

    For non-periodic systems the only required argument is the table of atom
    positions and symbols. For periodic systems, at a minimum, the atoms and
    frames dataframes must be provided.

    Bonds are computed semi-empirically and exist if:

    .. math::

        distance(A, B) < covalent\_radius(A) + covalent\_radius(B) + bond\_extra

    Args:
        atoms (:class:`~atomic.atom.Atom`): Table of nuclear positions and symbols
        frames (:class:`~pandas.DataFrame`): DataFrame of periodic cell dimensions (or False if )
        k (int): Number of distances (per atom) to compute
        bond_extra (float): Extra distance to include when determining bonds (see above)
        dmax (float): Max distance of interest (larger distances are ignored)
        dmin (float): Min distance of interest (smaller distances are ignored)

    Return:
        df (:class:`~atomic.twobody.TwoBody`): Two body property table
    '''
    if 'periodic' in universe.frames.columns:    # Figure out what type of two body properties to compute
        if any(universe.frames['periodic'] == True):
            req = ['xr', 'yr', 'zr']
            if any((mag not in universe.frames.columns for mag in req)): # Check the requirements are met
                raise ColumnError(req, universe.frames)
        atoms = universe.atoms[['symbol', 'x', 'y', 'z']]
        frames = universe.frames[['xr', 'yr', 'zr', 'atom_count']]
        return _compute_periodic_twobody(atoms, cells, k, bond_extra, dmax, dmin)
    else:
        raise NotImplementedError()


def _compute_periodic_twobody(atoms, cells, k, bond_extra, dmax, dmin):
    '''
    '''
    print('computing...')
    groups = atoms.groupby(level='frame')
    atompbcdfs = []
    twobodypbcdfs = []
    for i, (fdx, atomdf) =
    return TwoBody()
