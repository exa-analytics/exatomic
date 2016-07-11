# -*- coding: utf-8 -*-
'''
Numerical Orbital Functions
=================================================
Building discrete molecular orbitals (for visualization) requires a complex set of operations that
are provided by this module and wrapped into a clean API.
'''
import sympy as sy
from exa import _conf, Series
from exa.algorithms import meshgrid3d


ex, ey, ez = sy.symbols('x y z', imaginary=False)


def generate_volume(universe, orbitals=None, **kwargs):
    '''
    '''
    if 'rmin' in kwargs:
        return _cubic_volume(universe, orbitals, **kwargs)
    else:
        raise NotImplementedError('Cubic volumes only')


def _cubic_volume(universe, orbitals, rmin, rmax, nr):
    '''
    Steps:
        1: Generate voluminated normalized cartesian basis functions
        2: Reduce them to spherical basis function set
        3) Combine into mos
        4) normalize mos
    '''
    # 1) Generate voluminated normalized cartesians
    basis_set_groups = universe.basis_set.groupby('basis_set')
    #for index, Ax, Ay, Az, bset in zip(universe.atom.index)
