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
from exa.util import isotopes


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
