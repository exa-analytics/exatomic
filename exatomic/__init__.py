# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
A unified data platform for computational and theoretical chemists and
physicists.

Supported software
####################
- :mod:`~exatomic.adf.__init__`: `Amsterdam Density Functional`_

.. _Amsterdam Density Functional: https://www.scm.com
"""
def _jupyter_nbextension_paths():
    """Jupyter notebook extension directory paths."""
    return [{
        'section': "notebook",
        'src': "static",
        'dest': "jupyter-exatomic",
        'require': "jupyter-exatomic/extension"
    }]
#
#
#<<<<<<< HEAD
#from exatomic._version import __version__
#from exatomic.container import Universe
#from exatomic.xyz import XYZ
#=======
#from ._version import __exatomic_version__
#__version__ = __exatomic_version__
#
#try:
#    from exa.cms import (Length, Mass, Time, Current, Amount, Luminosity, Isotope,
#                         Dose, Acceleration, Charge, Dipole, Energy, Force,
#                         Frequency, MolarMass)
#except:
#    from exa.relational import Isotope, Length, Energy, Time, Amount, Constant, Mass
#from exatomic import _config
#from exatomic import error
#
## User API
#from exatomic.container import Universe, basis_function_contributions
#from exatomic.editor import Editor
#from exatomic.filetypes import XYZ, Cube
#
#from exatomic import tests
#from exatomic.algorithms import delocalization
#from exatomic.algorithms import neighbors
#from exatomic.algorithms import diffusion
#from exatomic.algorithms import pcf
#
#from exatomic import molcas
#from exatomic import nwchem
#from exatomic import gaussian
#from exatomic import adf
#from exatomic import nbo
#from exatomic import mpl
#<<<<<<< HEAD
#>>>>>>> 811f6aaae1e1aef968c27a34842d5ad9e7267217
#=======
#
#from exatomic.widget import TestContainer, TestUniverse, UniverseWidget
#>>>>>>> 454ebaa9677a776a535abb28f528efefabda52c5
