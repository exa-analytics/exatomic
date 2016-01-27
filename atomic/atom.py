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

    def _compute_unit_non_var_cell(self, rxyz, oxyz):
        '''
        '''
        xyz = self[['x', 'y', 'z']]
        unit = np.mod(xyz, rxyz) + oxyz
        return unit[unit != xyz].astype(np.float64).to_sparse()


class UnitAtom(Updater):
    '''
    '''
    __key__ = ['atom']


class VisualAtom(DataFrame):
    '''
    Special positions for atoms used to generate coherent animations.
    '''
    pass


class ProjectedAtom(DataFrame):
    '''
    A 3 x 3 x 3 super cell generate using the primitive cell positions.

    See Also:
        :class:`~atomic.atom.PrimitiveAtom`
    '''
    pass


def get_unit_atom(universe):
    '''
    '''
    rxyz = universe.frame.ix[0, ['rx', 'ry', 'rz']].values
    oxyz = universe.frame.ix[0, ['ox', 'oy', 'oz']].values
    obj = universe.atom._compute_unit_non_var_cell(rxyz, oxyz)
    return UnitAtom(obj)
