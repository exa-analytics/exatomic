# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Base Functionality
############################
This module provides conversion/mappers between element symbols, Z, mass, and
radius.

Attributes:
    sym2z: Mapper between element symbols and Z number
    z2sym: Mapper from Z to element symbol
    sym2mass: Mapper from symbol to mass
    sym2radius: Mapper from symbol to radius
    sym2color: Mapper from symbol to color
"""
import os
from exa.util import isotopes


# Taken from exa
def staticdir():
    """Return the location of the static data directory."""
    root = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(root, "static")


def resource(name):
    """
    Return the full path of a named resource in the static directory.

    If multiple files with the same name exist, **name** should contain
    the first directory as well.

    .. code-block:: python

        resource("myfile")
        resource("test01/test.txt")
        resource("test02/test.txt")
    """
    for path, dirs, files in os.walk(staticdir()):
        if name in files:
            return os.path.abspath(os.path.join(path, name))


isotopedf = isotopes.as_df()
sym2z = isotopedf.drop_duplicates("symbol").set_index("symbol")["Z"].to_dict()
z2sym = {v: k for k, v in sym2z.items()}
sym2mass = {}
sym2radius = {}
sym2color = {}
for k, v in vars(isotopes).items():
    if isinstance(v, isotopes.Element):
        sym2mass[k] = v.mass
        sym2radius[k] = v.radius
        sym2color[k] = v.color
