# -*- coding: utf-8 -*-
'''
Atom DataFrame
==========================
'''
import gc
from itertools import combinations
from scipy.spatial import cKDTree
from exa import _np as np
from exa import _pd as pd
from exa import DataFrame
from exa.config import Config
from atomic.errors import PeriodicError
from atomic.tools import check
if Config.numba:
    from exa.jitted.iteration import periodic_supercell, repeat_i8, repeat_i8_array, tile_i8
    from exa.jitted.iteration import pdist2d as pdist
else:
    from exa.algorithms.iteration import periodic_supercell
    import numpy.repeat as repeat_i8
    import numpy.tile as tile_i8
    from scipy.spatial.distance import pdist


class Atom(DataFrame):
    '''
    Absolute positions of atoms and their symbol.

    Required indexes: frame, atom

    Required columns: symbol, x, y, z
    '''
    __pk__ = ['atom']
    __fk__ = ['frame']


class VisualAtom(DataFrame):
    '''
    Special positions for atoms used to generate coherent animations.
    '''
    pass


class PrimitiveAtom(DataFrame):
    '''
    Primitive (or in unit cell) coordinates.
    '''
    pass


class ProjectedAtom(DataFrame):
    '''
    A 3 x 3 x 3 super cell generate using the primitive cell positions.

    See Also:
        :class:`~atomic.atom.PrimitiveAtom`
    '''
    pass


def compute_primitive(universe):
    '''
    Compute the primitive cell positions for each frame in the universe.

    Args:
        universe (:class:`~atomic.universe.Universe`): Universe containing the atoms table

    Returns:
        prim_atoms (:class:`~atomic.atom.PrimitiveAtom`): Primitive positions table
    '''
    if check(universe):
        groups = universe.atoms[['x', 'y', 'z']].groupby(level='frame')
        xyz = np.empty((groups.ngroups, ), dtype='O')
        for i, (fdx, group) in enumerate(groups):
            if np.mod(i, Config.gc) == 0:
                gc.collect()
            r = universe.frames.ix[fdx, ['rx', 'ry', 'rz']].values
            o = universe.frames.ix[fdx, ['ox', 'oy', 'oz']].values
            xyz[i] = np.mod(group, r) + o    # Compute unit cell positions
        xyz = pd.concat(xyz)
        xyz[['x', 'y', 'z']] = xyz[['x', 'y', 'z']].astype(np.float64)
        return PrimitiveAtom(xyz)
    raise PeriodicError()


def compute_supercell(universe):
    '''
    '''
    if check(universe):
        if hasattr(universe, 'primitive_atoms'):
            groups = universe.primitive_atoms[['x', 'y', 'z']].groupby(level='frame')
            n = groups.ngroups
            pxyz_list = np.empty((n, ), dtype='O')
            atom_list = np.empty((n, ), dtype='O')
            index_list = np.empty((n, ), dtype='O')
            frame_list = np.empty((n, ), dtype='O')
            symbol_list = np.empty((n, ), dtype='O')
            for i, (fdx, xyz) in enumerate(groups):
                rx = universe.frames.ix[fdx, 'rx']
                ry = universe.frames.ix[fdx, 'ry']
                rz = universe.frames.ix[fdx, 'rz']
                ac = universe.frames.ix[fdx, 'atom_count']
                nn = ac * 27
                pxyz_list[i] = periodic_supercell(xyz.values, rx, ry, rz)
                atom_list[i] = tile_i8(xyz.index.get_level_values('atom').values, 27)
                index_list[i] = range(nn)
                frame_list[i] = repeat_i8(fdx, nn)
                symbol_list[i] = universe.atoms.ix[fdx, 'symbol'].tolist() * 27
            df = pd.DataFrame(np.concatenate(pxyz_list), columns=['x', 'y', 'z'])
            df['atom'] = np.concatenate(atom_list)
            obj = np.concatenate(index_list)
            df['super_atom'] = np.concatenate(index_list)
            df['frame'] = np.concatenate(frame_list)
            df['symbol'] = np.concatenate(symbol_list)
            df.set_index(['frame', 'super_atom'], inplace=True)
            return SuperAtom(df)
    raise PeriodicError()
