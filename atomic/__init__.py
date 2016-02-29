# -*- coding: utf-8 -*-
import sys as _sys                              # HACK for development
import os as _os
_sys.path.insert(0, _os.sep.join(
    (_os.path.dirname(_os.path.realpath(__file__)), '..', '..', 'exa'))
)


__atomic_version__ = (0, 1, 0)                  # atomic VERSION NUMBER
__version__ = '.'.join((str(v) for v in __atomic_version__))


from exa.relational.isotope import Isotope
from exa.relational.unit import Length, Energy
from exa.relational.constant import Constant


from atomic.universe import Universe            # atomic imports
from atomic.cube import read_cubes
from atomic.xyz import read_xyz
from atomic import algorithms
#from atomic.pdb import read_pdb, write_pdb
#from atomic.cube import read_cubes
#import atomic.tests
#import atomic.algorithms.tests

del universe, xyz, frame
