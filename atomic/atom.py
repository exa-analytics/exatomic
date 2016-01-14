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


class AtomPBC(DataFrame):
    '''
    Periodic (boundary conditions) atoms positions with a foriegn key point
    back to the absolute positions table.
    '''
    __indices__ = ['frame', 'atompbc']
    __columns__ = ['x', 'y', 'z', 'atom']


class VisualAtom(DataFrame):
    '''
    Special positions for atoms (not the actual absolute positions) that
    are used in combination with the Atom table to generate representative
    animations.
    '''
    __indices__ = ['frame', 'atomvis']
    __columns__ = ['x', 'y', 'z', 'atom']


def compute_twobody(atoms, frames=None, k=None, bond_extra=0.45, dmax=13.0, dmin=0.3):
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
    # TODO: Check that necessary data have be computed (e.g. unit cell mags).
    if frames:
        req_cols = ['xi', 'xj', 'xk', 'yi', 'yj', 'yk', 'zi', 'zj', 'zk']
        mia = set(req_cols).difference(periodic.columns)
        if mia:                          # Check that we have cell dimensions
            raise ColumnError(mia, self)
        if any((mag not in periodic.columns for mag in ['xr', 'yr', 'zr'])):
            frames.cell_mags()
        return compute_periodic_twobody(atoms, periodic, k, bond_extra, dmax, dmin)
    else:
        raise NotImplementedError()


def _compute_periodic_twobody(atoms, periodic, unitize=False, k=None,
                              bond_extra=0.45, dmax=13.0, dmin=0.3):
    '''
    '''
    print('got here')
