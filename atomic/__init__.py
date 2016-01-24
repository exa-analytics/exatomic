# -*- coding: utf-8 -*-
'''
atomic
===================

'''
__atomic_version__ = (0, 1, 0)                  # atomic VERSION NUMBER
__version__ = '.'.join((str(v) for v in __atomic_version__))


from exa.relational.isotopes import Isotope
from exa.relational.units import Length, Energy
from exa.relational.constants import Constant

from atomic.universe import Universe            # atomic imports
from atomic.xyz import read_xyz
#from atomic.pdb import read_pdb, write_pdb
#from atomic.cube import read_cubes
#import atomic.tests
#import atomic.algorithms.tests

del universe, xyz, twobody
