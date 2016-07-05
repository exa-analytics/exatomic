# -*- coding: utf-8 -*-
__exatomic_version__ = (0, 2, 2)                  # exatomic VERSION NUMBER
__version__ = '.'.join((str(v) for v in __exatomic_version__))


from exa.relational import Isotope, Length, Energy, Time, Amount, Constant
from exatomic import _config
_config.update_config()
from exatomic.frame import Frame
from exatomic.atom import Atom
from exatomic.universe import Universe
from exatomic.editor import Editor
from exatomic.filetypes import XYZ, write_xyz, Cube
from exatomic.algorithms import nearest_molecules, einstein_relation, radial_pair_correlation
from exatomic import tests, _install


if _config.config['exa_persistent'] == False:
    _install.install()
