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
        'src': "_static",
        'dest': "jupyter-exatomic",
        'require': "jupyter-exatomic/extension"
    }]


#from exatomic._version import __version__, version_info
# Data objects
#from .atom import Atom





try:
    from exa.cms import (Length, Mass, Time, Current, Amount, Luminosity, Isotope,
                         Dose, Acceleration, Charge, Dipole, Energy, Force,
                         Frequency, MolarMass)
except:
    from exa.relational import Isotope, Length, Energy, Time, Amount, Constant, Mass
from exatomic import _config
from exatomic import error
from exatomic.editor import Editor


#from exatomic import molcas
#from exatomic import nwchem
#from exatomic import gaussian
#from exatomic import adf
#from exatomic import nbo
#from exatomic import mpl
