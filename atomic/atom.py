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
    from exa.jitted.iteration import projected_unitcell, tile_i8
else:
    from exa.algorithms.iteration import projected_unitcell
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
    def _compute_unit_non_var_cell(self, rxyz, oxyz):
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
    '''
    rxyz = universe.frame.ix[0, ['rx', 'ry', 'rz']].values
    oxyz = universe.frame.ix[0, ['ox', 'oy', 'oz']].values
    obj = universe.atom._compute_unit_non_var_cell(rxyz, oxyz)
    return UnitAtom(obj)


def gen_projected_atom(universe):
    '''
    '''
    return _compute_projected_non_var_cell(universe)


def _compute_projected_non_var_cell(universe):
    '''
    '''
    rx = universe.frame.ix[0, 'rx']
    ry = universe.frame.ix[0, 'ry']
    rz = universe.frame.ix[0, 'rz']
    u = universe.unit_atom
    px = u['x'].values
    py = u['y'].values
    pz = u['z'].values
    df = projected_unitcell(px, py, pz, rx, ry, rz)
    df = pd.DataFrame(df, columns=['x', 'y', 'z'])
    df.index.names = ['prjd_atom']
    df['frame'] = tile_i8(universe.atom['frame'].values, 27)
    df['symbol'] = universe.atom['symbol'].tolist() * 27
    return ProjectedAtom(df)
