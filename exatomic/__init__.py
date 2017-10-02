# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
A unified data anlaysis and visualization platform for computational and
theoretical chemists, physicists, etc. Support for molecular geometry and
orbital visualization is provided via the `Jupyter`_ notebook, a web-browser
based interactive (multi-programming language) environment.

.. extended description (todo)


Supported Software
---------------------
The list below contains third-party software that is supported by this package.
For specific features supported (per software), see the appropriate description
below.

- :mod:`~exatomic.adf.__init__`: `Amsterdam Density Functional`_
- :mod:`~exatomic.gaussian.__init__`: `Gaussian`_
- :mod:`~exatomic.molcas.__init__`: `OpenMolcas`_
- :mod:`~exatomic.nbo.__init__`: `NBO`_
- :mod:`~exatomic.nwchem.__init__`: `NWChem`_
- :mod:`~exatomic.qe.__init__`: `Quantum ESPRESSO`_

.. _Jupyter: https://jupyter.org
.. _Amsterdam Density Functional: https://www.scm.com
.. _Gaussian: http://gaussian.com/
.. _OpenMolcas: https://gitlab.com/Molcas/OpenMolcas
.. _NBO: http://nbo6.chem.wisc.edu/
.. _NWChem: http://www.nwchem-sw.org/index.php/Main_Page
.. _Quantum ESPRESSO: http://www.quantum-espresso.org/
"""
def _jupyter_nbextension_paths():
    """Jupyter notebook extension directory paths."""
    return [{
        'section': "notebook",
        'src': "static",
        'dest': "jupyter-exatomic",
        'require': "jupyter-exatomic/extension"
    }]

from ._version import __version__
from .interfaces import XYZ














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
