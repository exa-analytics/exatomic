# -*- coding: utf-8 -*-
__atomic_version__ = (0, 1, 0)                  # atomic VERSION NUMBER
__version__ = '.'.join((str(v) for v in __atomic_version__))


from exa.relational.isotope import Isotope
from exa.relational.unit import Length, Energy
from exa.relational.constant import Constant


from atomic.universe import Universe            # atomic imports
from atomic.xyz import read_xyz
from atomic import algorithms

del universe, xyz, frame
