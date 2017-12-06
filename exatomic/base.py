# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Base Functionality
############################
"""
from exa.util import isotopes
from platform import system

# For numba compiled functions
sysname= system().lower()
nbpll = True if "linux" in sysname else False
nbtgt = "parallel" if nbpll else "cpu"

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
        sym2color[k] = '#' + v.color[-2:] + v.color[3:5] + v.color[1:3]
