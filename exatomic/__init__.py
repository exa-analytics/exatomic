# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Exatomic
#############
This package provides a unified data format for organizing, analyzing, and
visualizing data coming from the most common computational chemistry software
programs.

Note:
    Althought the package is called exatomic, some class and function names are
    shortened to just atomic.
'''
__exatomic_version__ = (0, 2, 4)
__version__ = '.'.join((str(v) for v in __exatomic_version__))


from exa.relational import Isotope, Length, Energy, Time, Amount, Constant, Mass
from exatomic import _config as config
from exatomic import error

# User API
from exatomic.container import Universe
from exatomic.editor import Editor

from exatomic.filetypes import XYZ




#from exatomic import _config
#_config.update_config()
#from exatomic.universe import Universe
#from exatomic.editor import Editor
#from exatomic.filetypes import XYZ, write_xyz, Cube
#from exatomic.algorithms import nearest_molecules, einstein_relation, radial_pair_correlation
#from exatomic import tests, _install
#
#if _config.config['exa_persistent'] == False:
#    _install.install()
#
