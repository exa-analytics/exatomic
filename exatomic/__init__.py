# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
A unified data analysis and visualization platform for computational and
theoretical chemists, physicists, etc. Support for molecular geometry and
orbital visualization is provided via the `Jupyter`_ notebook (a web-browser
based interactive - multi-programming language - environment).

.. extended description (todo)

Note:
    This package uses the `atomic`_ unit system (Hartree) by default.

Supported Software
---------------------
The list below contains third-party software for which partial support, by this
package, is provided. For specific support features, see the module specific
documentation.

- :ref:`adf-label`: `Amsterdam Density Functional`_
- :mod:`~exatomic.gaussian`: `Gaussian`_
- :mod:`~exatomic.molcas`: `OpenMolcas`_
- :mod:`~exatomic.nbo`: `NBO`_
- :ref:`nwchem-label`: `NWChem`_
- :mod:`~exatomic.qe`: `Quantum ESPRESSO`_
- :mod:`~exatomic.interfaces`: Additional 3rd party support

.. _Jupyter: https://jupyter.org
.. _Amsterdam Density Functional: https://www.scm.com
.. _Gaussian: http://gaussian.com/
.. _OpenMolcas: https://gitlab.com/Molcas/OpenMolcas
.. _NBO: http://nbo6.chem.wisc.edu/
.. _NWChem: http://www.nwchem-sw.org/index.php/Main_Page
.. _Quantum ESPRESSO: http://www.quantum-espresso.org/
.. _atomic: https://en.wikipedia.org/wiki/Atomic_units
"""
def _jupyter_nbextension_paths():
    """Jupyter notebook extension directory paths."""
    return [{
        'section': "notebook",
        'src': "static/nbextension",
        'dest': "exatomic",
        'require': "exatomic/extension"
    }]


__js_version__ = "0.4.0"
from ._version import __version__
from .core.universe import Universe
from .xyz import XYZ
from .util import builder
