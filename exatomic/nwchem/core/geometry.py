# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Parser for Output File Geometry
#################################
Standalone parser for the geometry block (printed as part of the NWChem input
module block).
"""
import re
import pandas as pd
from exa import Parser, Typed
from exatomic.core.atom import Atom


class Geometry(Parser):
    """Parser for the 'Geometry' section of NWChem output files."""
    _start = re.compile("^\s*Geometry \".*\" -> \".*\"")
    _end = re.compile("^\s*(Lattice Parameters|Atomic Mass)")
    _int0 = 7
    _int1 = -2
    _int2 = 3
    _cols = ("number", "tag", "charge", "x", "y", "z")
    _wids = (6, 17, 10, 15, 15, 15)
    atom = Typed(Atom, doc="Table of nuclear coordinates.")

    def parse_atom(self, length=None):
        """
        Create an :class:`~exatomic.core.atom.Atom` object.

        Read in the block using pandas' fixed width format reader and convert
        the second column (_cols) to symbol. Pass the data to the
        :class:`~exatomic.core.atom.Atom` which converts units and dtypes.

        Args:
            length (str): Unit of length for coordinates
        """
        if length is None:
            length = "Angstrom" if "angstrom" in str(self[self._int2]).lower() else "au"
        atom = pd.read_fwf(self[self._int0:self._int1].to_stream(),
                           widths=self._wids, names=self._cols)
        atom[self._cols[0]] = atom[self._cols[0]].astype(int)
        # Helper function to identify symbols from tags
        get_symbol = lambda x: "".join([a for a in x if a.isalpha()])
        atom['symbol'] = atom[self._cols[1]].apply(get_symbol)
        self.atom = Atom.from_xyz(atom, unit=length)
