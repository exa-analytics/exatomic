# -*- coding: utf-8 -*-
'''
Atom DataFrame
==========================
A specialized dataframe for storing and manipulating information about atomic
nuclei.

The following are columns that can be expected in the dataframes provided by
this module:
+-------------------+----------+-------------------------------------------+
| Column            | Type     | Description                               |
+===================+==========+===========================================+
| x                 | float    | position in x                             |
+-------------------+----------+-------------------------------------------+
| y                 | float    | position in y                             |
+-------------------+----------+-------------------------------------------+
| z                 | float    | position in z                             |
+-------------------+----------+-------------------------------------------+
| fx                | float    | force in x                                |
+-------------------+----------+-------------------------------------------+
| fy                | float    | force in y                                |
+-------------------+----------+-------------------------------------------+
| fz                | float    | force in z                                |
+-------------------+----------+-------------------------------------------+
| vx                | float    | velocity in x                             |
+-------------------+----------+-------------------------------------------+
| vy                | float    | velocity in y                             |
+-------------------+----------+-------------------------------------------+
| vz                | float    | velocity in z                             |
+-------------------+----------+-------------------------------------------+
| symbol            | object   | element symbol                            |
+-------------------+----------+-------------------------------------------+
| label             | int      | non-unique integer label                  |
+-------------------+----------+-------------------------------------------+
'''
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
    Base class for :class:`~atomic.atom.Atom` and :class:`~atomic.atom.ProjectedAtom`.
    '''
    __pk__ = ['atom']
    __fk__ = ['frame']
    __traits__ = ['x', 'y', 'z', 'radius', 'color']
    __groupby__ = 'frame'

    def _prep_trait_values(self):
        '''
        Maps the covalent radii and colors (for rendering the atoms).
        '''
        if 'radius' not in self.columns:
            self['radius'] = self['symbol'].astype('O').map(Isotope.symbol_to_radius_map)
        if 'color' not in self.columns:
            self['color'] = self['symbol'].astype('O').map(Isotope.symbol_to_color_map)

    def _post_trait_values(self):
        '''
        Cleans up the mapped covalent radii and colors after generating the
        (JavaScript) traits.
        '''
        del self['radius']
        del self['color']


class Atom(AtomBase, DataFrame):
    '''
    Absolute positions of atoms and their symbol.
    '''
    def get_element_mass(self, inplace=False):
        '''
        Retrieve the mass of each element in the atom dataframe.
        '''
        masses = self['symbol'].astype('O').map(Isotope.symbol_to_mass())
        if inplace:
            self['mass'] = masses
        else:
            return masses

    def compute_simple_formula(self):
        '''
        Compute the simple formula for each frame.
        '''

    def _compute_unit_atom_static_cell(self, rxyz, oxyz):
        '''
        Given a static unit cell, compute the unit cell coordinates for each
        atom.
        '''
        xyz = self[['x', 'y', 'z']]
        unit = np.mod(xyz, rxyz) + oxyz
        return UnitAtom(unit[unit != xyz].astype(np.float64).to_sparse())


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
    __fk__ = ['atom']


def get_unit_atom(universe, inplace=False):
    '''
    Compute the :class:`~atomic.atom.UnitAtom` for a given
    :class:`~atomic.universe.Universe`.
    '''
    df = None
    if universe.is_periodic():
        if universe.is_variable_cell():
            raise NotImplementedError()
        else:
            df = _compute_unit_atom_static_cell(universe)
    if inplace:
        universe._unit_atom = df
    else:
        return df


def _compute_unit_atom_static_cell(universe):
    '''
    '''
    rxyz = universe.frame._get_min_values('rx', 'ry', 'rz')
    oxyz = universe.frame._get_min_values('ox', 'oy', 'oz')
    return universe.atom._compute_unit_atom_static_cell(rxyz, oxyz)


def get_projected_atom(universe, inplace=False):
    '''
    '''
    prjd_atom = None
    if universe.is_variable_cell():
        raise NotImplementedError()
    else:
        prjd_atom = _compute_projected_atom_static_cell(universe)
    if inplace:
        universe._prjd_atom = prjd_atom
    else:
        return prjd_atom


def _compute_projected_atom_static_cell(universe):
    '''
    '''
    rxyz = universe.frame._get_min_values('rx', 'ry', 'rz')
    xyz = universe.unit_atom._get_column_values('x', 'y', 'z')
    df = project_coordinates(xyz, rxyz)
    df = pd.DataFrame(df, columns=('x', 'y', 'z'))
    df.index.names = ['prjd_atom']
    df['frame'] = tile_i8(universe.atom['frame'].values, 27)
    df['symbol'] = universe.atom['symbol'].tolist() * 27
    df['symbol'] = df['symbol'].astype('category')
    df['atom'] = tile_i8(universe.atom.index.values, 27)
    return ProjectedAtom(df)
