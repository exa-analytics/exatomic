# -*- coding: utf-8 -*-
'''
atomic
============
This package provides the atomic container and associated functionality.
'''

import os as _os
import numpy as _np
import pandas as _pd
from .xyz import read_xyz, write_xyz

# Aliases
__atomic_version__ = (0, 1, 0)    # atomic VERSION NUMBER
__version__ = '.'.join((str(v) for v in __atomic_version__))
