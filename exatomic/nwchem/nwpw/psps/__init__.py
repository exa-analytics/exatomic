# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
If the ``print debug`` option is used in a plane wave calculation, the permanent
directory or scratch directory will contain actual pseudized (and all-electron)
wave forms and potentials relevant for the pseudopotential/plane wave method.
`NWChem`_ supports two types of pseudopotentials, the norm-conserving approach
and projector augmented wave method. This sub-package provides parsing functionality
for this (typically debug) data.
"""
from .base import NWChemPSPs
#from .ae import AEOutput
#from .ps import PAWOutput
#from .pspfile import PAWPSP
