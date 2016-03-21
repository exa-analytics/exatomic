# -*- coding: utf-8 -*-
'''
Atom Dataframes
==========================
A dataframe containing the nuclear positions, forces, velocities, symbols, etc.
Examples of data that may exist in this dataframe are given below (note that
the dataframe is not limited to only these record types - rather this provides
a guide for what type of data is required and can be expected).

+-------------------+----------+-------------------------------------------+
| Column            | Type     | Description                               |
+===================+==========+===========================================+
| x                 | float    | position in x (req.)                      |
+-------------------+----------+-------------------------------------------+
| y                 | float    | position in y (req.)                      |
+-------------------+----------+-------------------------------------------+
| z                 | float    | position in z (req.)                      |
+-------------------+----------+-------------------------------------------+
| frame             | category | non-unique integer (req.)                 |
+-------------------+----------+-------------------------------------------+
| symbol            | category | element symbol (req.)                     |
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
| label             | category | non-unique integer                        |
+-------------------+----------+-------------------------------------------+

See Also:
    :class:`~atomic.universe.Universe`
'''
import numpy as np
import pandas as pd
from traitlets import Dict
from exa.numerical import DataFrame


class AtomBase:
    '''
    Base class for :class:`~atomic.atom.Atom` and :class:`~atomic.atom.ProjectedAtom`.
    '''
    radius = Dict()
    color = Dict()
    _indices = ['atom']
    _columns = ['x', 'y', 'z', 'symbol', 'frame']
    _traits = ['x', 'y', 'z', 'radius', 'color']
    _groupbys = ['frame']
    _categories = {'frame': np.int64}


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


#class UnitAtom(Updater):
#    '''
#    '''
#    __key__ = ['atom']
#
#
#class VisualAtom(DataFrame):
#    '''
#    Special positions for atoms used to generate coherent animations.
#    '''
#    pass
#
#
#class ProjectedAtom(AtomBase, DataFrame):
#    '''
#    A 3 x 3 x 3 super cell generate using the primitive cell positions.
#
#    See Also:
#        :class:`~atomic.atom.PrimitiveAtom`
#    '''
#    _pkeys = ['prjd_atom']
#    _fkeys = ['atom']
#
#
#def get_unit_atom(universe, inplace=False):
#    '''
#    Compute the :class:`~atomic.atom.UnitAtom` for a given
#    :class:`~atomic.universe.Universe`.
#    '''
#    df = None
#    if universe.is_periodic():
#        if universe.is_variable_cell():
#            raise NotImplementedError()
#        else:
#            df = _compute_unit_atom_static_cell(universe)
#    if inplace:
#        universe._unit_atom = df
#    else:
#        return df
#
#
#def _compute_unit_atom_static_cell(universe):
#    '''
#    '''
#    rxyz = universe.frame._get_min_values('rx', 'ry', 'rz')
#    oxyz = universe.frame._get_min_values('ox', 'oy', 'oz')
#    return universe.atom._compute_unit_atom_static_cell(rxyz, oxyz)
#
#
#def get_projected_atom(universe, inplace=False):
#    '''
#    '''
#    prjd_atom = None
#    if universe.is_variable_cell():
#        raise NotImplementedError()
#    else:
#        prjd_atom = _compute_projected_atom_static_cell(universe)
#    if inplace:
#        universe._prjd_atom = prjd_atom
#    else:
#        return prjd_atom
#
#
#def _compute_projected_atom_static_cell(universe):
#    '''
#    '''
#    rxyz = universe.frame._get_min_values('rx', 'ry', 'rz')
#    xyz = universe.unit_atom._get_column_values('x', 'y', 'z')
#    df = project_coordinates(xyz, rxyz)
#    df = pd.DataFrame(df, columns=('x', 'y', 'z'))
#    df.index.names = ['prjd_atom']
#    df['frame'] = tile_i8(universe.atom['frame'].astype('i8').values, 27)
#    df['symbol'] = universe.atom['symbol'].astype('O').tolist() * 27
#    df['symbol'] = df['symbol'].astype('category')
#    df['atom'] = tile_i8(universe.atom.index.values, 27)
#    return ProjectedAtom(df)
#
