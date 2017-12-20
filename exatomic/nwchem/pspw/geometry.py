# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Parser for Output File Geometry
#################################
Standalone parser for the geometry block (printed as part of the NWChem input
module block).
"""
import re
from exatomic.nwchem.core.geometry import Geometry as _Geometry


class Geometry(_Geometry):
    """Parser for plane wave geometry sections."""
    pass
