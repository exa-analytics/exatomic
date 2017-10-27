# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Base Variables
############################
This module provides some global configuration-like variables for the exatomic
package. Some variables are used as internal only, e.g. for compilation using
`numba`_. Other variables may be useful at the user level, e.g. conversions
and mappers between element symbols, Z, mass, etc.

Attributes:
    sym2z: Mapper between element symbols and Z number
    z2sym: Mapper from Z to element symbol
    sym2mass: Mapper from symbol to mass
    sym2radius: Mapper from symbol to radius
    sym2color: Mapper from symbol to color
"""
import os
from platform import system
from exa.util import isotopes


# For numba compiled functions
sysname= system().lower()
nbpll = True if "linux" in sysname else False
nbtgt = "parallel" if nbpll else "cpu"
nbche = False if nbpll else True


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


# Element mappers
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
