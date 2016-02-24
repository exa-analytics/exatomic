# -*- coding: utf-8 -*-
__atomic_version__ = (0, 1, 0)                  # atomic VERSION NUMBER
__version__ = '.'.join((str(v) for v in __atomic_version__))


from exa.config import Config
from exa.relational.isotope import Isotope
from exa.relational.unit import Length, Energy
from exa.relational.constant import Constant


from atomic.universe import Universe            # atomic imports
from atomic.xyz import read_xyz
from atomic import algorithms
from atomic.formula import SimpleFormula

if Config._temp:
    from exa.tools import install_notebook_widgets
    from exa.relational import create_all
    from atomic.tools import update_config
    update_config()
    create_all()
    install_notebook_widgets(Config.atomic['nbext'], Config.atomic['extensions'])
    del create_all


del universe, xyz, frame, atom, two, Config
