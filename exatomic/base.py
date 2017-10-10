# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Global Functionality
########################
"""
from exa.util import isotopes


# Mappers
sym2z = isotopes.as_df()
sym2z = sym2z.drop_duplicates("symbol").set_index("symbol")["Z"].to_dict()
z2sym = {v: k for k, v in sym2z.items()}
