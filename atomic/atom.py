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
from exa.config import Config
from exa.frames import DataFrame, Updater
from atomic import Isotope
if Config.numba:
    from exa.jitted.iteration import project_coordinates, tile_i8
else:
    from exa.algorithms.iteration import project_coordinates
    from numpy import tile as tile_i8


class AtomBase:
    '''
    '''
    __pk__ = ['atom']
    __fk__ = ['frame']
    __traits__ = ['x', 'y', 'z', 'radius', 'color']
    __groupby__ = 'frame'

    def _prep_trait_values(self):
        '''
        '''
        if 'radius' not in self.columns:
            self['radius'] = self['symbol'].map(Isotope.lookup_radius_by_symbol)
        if 'color' not in self.columns:
            self['color'] = self['symbol'].map(Isotope.lookup_color_by_symbol)

    def _post_trait_values(self):
        '''
        '''
        del self['radius']
        del self['color']


class Atom(AtomBase, DataFrame):
    '''
    Absolute positions of atoms and their symbol.

    Required indexes: frame, atom

    Required columns: symbol, x, y, z
    '''
    def _compute_unit_atom_static_cell(self, rxyz, oxyz):
        '''
        '''
        xyz = self[['x', 'y', 'z']]
        unit = np.mod(xyz, rxyz) + oxyz
        unit = unit[unit != xyz].astype(np.float64).to_sparse()
        return unit


class UnitAtom(Updater):
    '''
    '''
    __key__ = ['atom']


class VisualAtom(DataFrame):
    '''
    Special positions for atoms used to generate coherent animations.
    '''
    pass


class ProjectedAtom(AtomBase, DataFrame):
    '''
    A 3 x 3 x 3 super cell generate using the primitive cell positions.

    See Also:
        :class:`~atomic.atom.PrimitiveAtom`
    '''
    __pk__ = ['prjd_atom']


def get_unit_atom(universe):
    '''
    Compute the :class:`~atomic.atom.UnitAtom` for a given
    :class:`~atomic.universe.Universe`.
    '''
    if universe.is_variable_cell():
        raise NotImplementedError()
    else:
        return _compute_unit_atom_static_cell(universe)


def _compute_unit_atom_static_cell(universe):
    '''
    '''
    rxyz = universe.frame._get_min_values(('rx', 'ry', 'rz'))
    oxyz = universe.frame._get_min_values(('ox', 'oy', 'oz'))
    df = universe.atom._compute_unit_atom_static_cell(rxyz, oxyz)
    return UnitAtom(df)


def get_projected_atom(universe):
    '''
    '''
    if universe.is_variable_cell():
        raise NotImplementedError()
    else:
        return _compute_projected_atom_static_cell(universe)


def _compute_projected_atom_static_cell(universe):
    '''
    '''
    rxyz = universe.frame._get_min_values(('rx', 'ry', 'rz'))
    xyz = universe.unit_atom._get_column_values(('x', 'y', 'z'))
    df = project_coordinates(xyz, rxyz)
    df = pd.DataFrame(df, columns=('x', 'y', 'z'))
    df.index.names = ['prjd_atom']
    df['frame'] = tile_i8(universe.atom['frame'].values, 27)
    df['symbol'] = universe.atom['symbol'].tolist() * 27
    return ProjectedAtom(df)
