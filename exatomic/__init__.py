# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Exatomic
#############
This package provides a unified data format for organizing, analyzing, and
visualizing data coming from the most common computational chemistry software
programs.

Note:
    Although the package is called exatomic, some class and function names are
    shortened to just atomic.

Warning:
    This package uses the `atomic`_ unit system - all quantities appear in
    atomic units!

.. _atomic: https://en.wikipedia.org/wiki/Atomic_units
"""
__exatomic_version__ = (0, 3, 6)
__version__ = '.'.join((str(v) for v in __exatomic_version__))


from exa.relational import Isotope, Length, Energy, Time, Amount, Constant, Mass
from exatomic import _config
from exatomic import error

# User API
from exatomic.container import Universe, basis_function_contributions
from exatomic.editor import Editor
from exatomic.filetypes import XYZ, Cube

from exatomic import test
from exatomic.algorithms import neighbors
from exatomic.algorithms import diffusion
from exatomic.algorithms import pcf
