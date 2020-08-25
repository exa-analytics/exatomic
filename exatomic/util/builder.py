# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Construction of Molecules
################################
This module provides methods for constructing molecular structures
"""
from exa.util import isotopes
from exa.util.units import Length
from exatomic.core.atom import Atom
from exatomic.core.universe import Universe


def _builder(atom=None, unit="Angstrom", **kwargs):
    """
    """
    atom = Atom(atom, columns=("frame", "x", "y", "z", "Z"))
    conv = Length[unit, "au"]
    for q in ("x", "y", "z"):
        atom[q] *= conv
    return Universe(atom=atom, **kwargs)


def monatomic(element, unit="Angstrom", **kwargs):
    """
    """
    if not isinstance(element, isotopes.Element):
        element = isotopes.get(element)
    atom = [(0, 0.0, 0.0, 0.0, element), ]
    return _builder(atom=atom, unit=unit, **kwargs)


def diatomic(element_a, element_b, length, unit="Angstrom",
             center=(0.0, 0.0, 0.0), orient="z", **kwargs):
    """
    Construct a diatomic molecule.

    Args:
        element_a (str, int): String abbreviation or element Z number
        element_b (str, int): String abbreviation or element Z number
        length (float): Bond length (default unit Angstrom)
        unit (str): Unit of length
        origin (tuple): Point position of the origin
        orient (str): Direction in which to orient the molecule

    Returns:
        uni (:class:`~exatomic.core.universe.Universe`): Constructed universe containing the diatomic
    """
    if not isinstance(element_a, isotopes.Element):
        element_a = isotopes.get(element_a)
    if not isinstance(element_b, isotopes.Element):
        element_b = isotopes.get(element_b)
    z = length/2
    atom = [(0, 0.0, 0.0, -z, element_a.Z),
            (0, 0.0, 0.0, z, element_b.Z)]
    return _builder(atom=atom, unit=unit, **kwargs)
