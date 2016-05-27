# -*- coding: utf-8 -*-
__atomic_version__ = (0, 2, 1)                  # atomic VERSION NUMBER
__version__ = '.'.join((str(v) for v in __atomic_version__))


# exatomic depends strongly on exa
from exa.relational import Isotope, Length, Energy, Time, Amount, Constant
from exa.relational.isotope import init_mappers


from exatomic._config import global_config
from exatomic.frame import Frame
from exatomic.atom import Atom
from exatomic.molecule import Molecule
from exatomic.basis import PlanewaveBasisSet, GaussianBasisSet, SlaterBasisSet
from exatomic.universe import Universe
from exatomic.editor import Editor
from exatomic.formula import SimpleFormula
from exatomic.filetypes import XYZ, write_xyz, Cube
from exatomic.algorithms import nearest_molecules, einstein_relation, radial_pair_correlation
from exatomic._install import install
from exatomic import tests


if global_config['exa_persistent'] == False:
    install(False)


# Commonly used dataframe mappers not initialized by default in exa
init_mappers()


del init_mappers
