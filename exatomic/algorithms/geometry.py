# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Geometry
======================
Functions for constructing molecular and solid state geometries with
symmetry adapted or crystalline structures.
"""
import numpy as np
import pandas as pd
from exa.util.units import Length


columns = ['x', 'y', 'z', 'symbol', 'frame', 'label']


def make_small_molecule(center, ligand, distance, geometry,
                        offset=None, plane=None, axis=None,
                        domains=None, unit='Angstrom',
                        angle=None):
    """
    A minimal molecule builder for simple one-center, homogeneous ligand
    molecules of various general chemistry molecular geometries. If 'domains'
    is not specified and geometry is ambiguous (like 'bent'),
    it just guesses the simplest geometry (smallest number of domains).

    Args:
        center (str): atomic symbol of central atom
        ligand (str): atomic symbol of ligand atoms
        distance (float): distance between central atom and ligand
        geometry (str): molecular geometry
        offset (np.array): 3-array of position of central atom
        plane (str): cartesian plane of molecule (eg. for 'square_planar')
        axis (str): cartesian axis of molecule (eg. for 'linear')
        domains (int): number of electronic domains
        unit (str): unit of distance (default 'Angstrom')

    Returns:
        df (:class:`~exatomic.core.atom.Atom`): Atom table of small molecule
    """
    distance *= Length[unit, 'au']
    funcs = {2: _2_domain, 3: _3_domain, 4: _4_domain, 5: _5_domain, 6: _6_domain}
    if domains is not None:
        return funcs[domains](center, ligand, distance, geometry, offset, plane, axis, angle)
    # 2 domains
    if geometry in ['linear', 'bent']:
        return _2_domain(center, ligand, distance, geometry, offset, plane, axis, angle)
    # 3 domains
    if geometry in ['trigonal_planar', 'trigonal_pyramidal', 't_shaped']:
        return _3_domain(center, ligand, distance, geometry, offset, plane, axis, angle)
    # 4 domains
    if geometry in ['tetrahedral', 'square_planar', 'seesaw']:
        return _4_domain(center, ligand, distance, geometry, offset, plane, axis, angle)
    # 5 domains
    if geometry in ['trigonal_bipyramidal', 'square_pyramidal']:
        return _5_domain(center, ligand, distance, geometry, offset, plane, axis, angle)
    # 6 domains
    if geometry in ['octahedral']:
        return _6_domain(center, ligand, distance, geometry, offset, plane, axis, angle)
    raise NotImplementedError


def _2_domain(center, ligand, distance, geometry, offset, plane, axis, angle):
    if axis is None: axis = 'z'
    if geometry == 'linear':
        cart = ['x', 'y', 'z']
        arr = np.array([0, 0, 0])
        arr[cart.index(axis)] = distance
        origin = np.array([0., 0., 0.])
        if offset is not None:
            arr += offset
            origin += offset
        geom = [[origin[0], origin[1], origin[2], center, 0, 0]]
        cnt = 1
        for xi, yi, zi in [arr, -arr]:
            geom.append([xi, yi, zi, ligand, 0, cnt])
            cnt += 1
        return pd.DataFrame(geom, columns=columns)
    else:
        raise NotImplementedError

def _3_domain(center, ligand, distance, geometry, offset, plane, axis, angle):
    if geometry == 'trigonal_pyramidal':
        raise NotImplementedError('trigonal pyramidal not supported yet')
    if geometry == 'trigonal_planar':
        raise NotImplementedError('trigonal planar not supported yet')
    if geometry == 'bent':
        origin = np.array([0., 0., 0.])
        arr = np.array([distance, distance, 0.])
        if offset is not None:
            raise NotImplementedError('bent and offset not supported yet')
        geom = [[origin[0], origin[1], origin[2], center, 0, 0]]
        cnt = 1
        for angle in [0, 2 * np.pi / 3]:
            geom.append([arr[0] * np.cos(angle), arr[1] * np.sin(angle), arr[2], ligand, 0, cnt])
            cnt += 1
        return pd.DataFrame(geom, columns=columns)

def _4_domain(center, ligand, distance, geometry, offset, plane, axis, angle):
    if geometry == 'bent':
        origin = np.array([0., 0., 0.])
        arr = np.array([distance, distance, 0.])
        if offset is not None:
            raise NotImplementedError('bent and offset not supported yet')
        geom = [[origin[0], origin[1], origin[2], center, 0, 0]]
        cnt = 1
        for angle in [0, 109.5 * np.pi / 180]:
            geom.append([arr[0] * np.cos(angle), arr[1] * np.sin(angle), arr[2], ligand, 0, cnt])
            cnt += 1
        return pd.DataFrame(geom, columns=columns)
    if geometry == 'tetrahedral':
        raise NotImplementedError('tetrahedral not supported yet')
    if geometry == 'square_planar':
        cart = ['x', 'y', 'z']
        if plane is None:
            plane = 'xy'
            if axis == 'x':
                plane = 'yz'
            elif axis == 'y':
                plane = 'xz'
        if plane not in ['xy', 'yx', 'xz', 'zx', 'yz', 'zy']:
            raise NotImplementedError('pick a cartesian plane, eg. yz')
        ax1, ax2 = plane
        origin = np.array([0., 0., 0.])
        arr1 = np.array([0., 0., 0.])
        arr2 = np.array([0., 0., 0.])
        arr1[cart.index(ax1)] = distance
        arr2[cart.index(ax2)] = distance
        if offset is not None:
            origin += offset
            arr1 += offset
            arr2 += offset
        geom = [[origin[0], origin[1], origin[2], center, 0, 0]]
        cnt = 1
        for xi, yi, zi in [arr1, -arr1, arr2, -arr2]:
            geom.append([xi, yi, zi, ligand, 0, cnt])
            cnt += 1
        return pd.DataFrame(geom, columns=columns)
    if geometry == 'seesaw':
        cart = ['x', 'y', 'z']
        if offset is not None:
            raise NotImplementedError('seesaw and offset no bueno at the moment')
        if axis is None:
            axis = 'z'
        arr1 = np.array([0., 0., 0.])
        arr2 = np.array([0., 0., 0.])
        arr1[cart.index(axis)] = distance
        for car in cart:
            if car != axis:
                arr2[cart.index(car)] = distance
        origin = np.array([0., 0., 0.])
        geom = [[origin[0], origin[1], origin[2], center, 0, 0]]
        cnt = 1
        for xi, yi, zi in [arr1, -arr1]:
            geom.append([xi, yi, zi, ligand, 0, cnt])
            cnt += 1
        if axis == 'z':
            for angle in [0, 2 * np.pi / 3]:
                geom.append([arr2[0] * np.cos(angle),
                             arr2[1] * np.sin(angle), arr2[2], ligand, 0, cnt])
                cnt += 1
        elif axis == 'y':
            for angle in [0, 2 * np.pi / 3]:
                geom.append([arr2[0] * np.cos(angle), arr2[1],
                             arr2[2] * np.sin(angle), ligand, 0, cnt])
                cnt += 1
        elif axis == 'x':
            for angle in [0, 2 * np.pi / 3]:
                geom.append([arr2[0], arr2[1] * np.cos(angle),
                                      arr2[2] * np.sin(angle), ligand, 0, cnt])
                cnt += 1
        return pd.DataFrame(geom, columns=columns)


def _5_domain(center, ligand, distance, geometry, offset, plane, axis, angle):
    raise NotImplementedError('5 coordinate complexes not implemented yet')

def _6_domain(center, ligand, distance, geometry, offset, plane, axis, angle):
    if geometry != 'octahedral':
        raise NotImplementedError('only octahedral geometry supported currently')
    origin = np.array([0., 0., 0.])
    x = np.array([distance, 0., 0.])
    nx = np.array([-distance, 0., 0.])
    y = np.array([0., distance, 0.])
    ny = np.array([0., -distance, 0.])
    z = np.array([0., 0., distance])
    nz = np.array([0., 0., -distance])
    if offset is not None:
        origin += offset
        x += offset
        y += offset
        z += offset
        nx += offset
        ny += offset
        nz += offset
    geom = [[origin[0], origin[1], origin[2], center, 0, 0]]
    cnt = 1
    for xi, yi, zi in [x, nx, y, ny, z, nz]:
        geom.append([xi, yi, zi, ligand, 0, cnt])
        cnt += 1
    return pd.DataFrame(geom, columns=columns)
