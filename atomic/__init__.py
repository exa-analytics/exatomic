# -*- coding: utf-8 -*-
import sys as _sys                              # HACK for development
import os as _os
_sys.path.insert(0, _os.sep.join(
    (_os.path.dirname(_os.path.realpath(__file__)), '..', '..', 'exa'))
)
print(_sys.path)


__atomic_version__ = (0, 1, 0)                  # atomic VERSION NUMBER
__version__ = '.'.join((str(v) for v in __atomic_version__))


from exa import _pd, _np                        # imports from exa
from exa import run_unittests, run_doctests
from exa.relational.isotopes import Isotope
from exa.relational.units import Length, Energy
from exa.relational.constants import Constant

from atomic.xyz import read_xyz, write_xyz      # atomic imports
from atomic.pdb import read_pdb, write_pdb
from atomic.cube import read_cubes
from atomic import tests
