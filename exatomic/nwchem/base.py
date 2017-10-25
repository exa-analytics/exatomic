# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Base Parsers (Mixins)
########################################
This module provides mixin classes for output (wrapping) parsers (i.e. API
exposed to the user).
"""
import pandas as pd
from exa import Parser, Typed
# Data objects
from exatomic.core.atom import Atom
from exatomic.core.basis import GaussianBasisSet
from exatomic.core.orbital import Coefficient
# Parsers
from .basis import BasisSet
from .geometry import Geometry
from .orbital import MOVectors
from .header import Header


class _GTO_Output(Parser):
    """
    Base parser for NWChem output files: to be used as a mixin for output
    parsers for GTO code outputs.

    In some cases the parser cannot correctly assign frame indexes for the
    atom table. The **frame_map** attributed can be used to fix this problem.

    Attributes:
        frame_map (iterable): Correctly ordered frame indexes
    """
    atom = Typed(Atom, doc="Full atom table from all 'frames'.")
    basis_set = Typed(GaussianBasisSet, doc="Gaussian basis set description.")
    coefficient = Typed(Coefficient, doc="Full molecular orbital coefficient table.")

    def parse_atom(self):
        """
        Generate the complete :class:`~exatomic.core.atom.Atom` table
        for the entire calculation.

        Warning:
            The frame index is arbitrary! Map it to the correct
            values as required!
        """
        atoms = []
        for i, sec in enumerate(self.get_sections(Geometry)):
            atom = sec.atom
            fdx = i if self.frame_map is None else self.frame_map[i]
            atom['frame'] = fdx
            atoms.append(atom)
        self.atom = pd.concat(atoms, ignore_index=True)

    def parse_basis_set(self):
        """
        """
        key = self.sections[self.sections['parser'] == BasisSet].index[0]
        self.basis_set = self.get_section(key).basis_set

    def parse_coefficient(self):
        """
        Complete :class:`~exatomic.core.orbital.Coefficient` table.

        Warning:
            The frame index is arbitrary! Map it to the correct
            values as required!
        """
        coefs = []
        frames = sorted(self.atom['frame'].unique())
        for i, sec in enumerate(self.get_sections(MOVectors)):
            c = sec.coefficient
            fdx = frames[i]
            c['frame'] = fdx
            coefs.append(c)
        self.coefficient = pd.concat(coefs, ignore_index=True)

    def __init__(self, *args, **kwargs):
        frame_map = kwargs.pop("frame_map", None)
        super(_GTO_Output, self).__init__(*args, **kwargs)
        self.frame_map = frame_map

_GTO_Output.add_parsers(Header, Geometry, BasisSet, MOVectors)
