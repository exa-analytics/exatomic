# -*- coding: utf-8 -*-
__atomic_version__ = (0, 2, 0)                  # atomic VERSION NUMBER
__version__ = '.'.join((str(v) for v in __atomic_version__))


from exa.relational import Isotope, Length, Energy, Time, Amount, Constant


from atomic._config import _conf
from atomic.frame import Frame
from atomic.atom import Atom
from atomic.universe import Universe
from atomic.editor import Editor
from atomic.formula import SimpleFormula
from atomic.filetypes import XYZ, write_xyz, Cube
from atomic import tests


if not _conf['exa_persistent']:
    from atomic._install import install
    install()
