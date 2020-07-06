# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Atomic Positions Parser
###############################
"""
#import re
#import pandas as pd
#from exa import Parser
#from exa.typed import Typed
#from exatomic.core.atom import Atom
#
#
#class AtomicPositions(Parser):
#    """Parser for the 'ATOMIC_POSITIONS' section of an output."""
#    _start = re.compile(r"^\s*ATOMIC_POSITIONS|atomic_positions")
#    _int0 = 1
#    _cols = ("symbol", "x", "y", "z")
#    atom = Typed(Atom, doc="Table of nuclear coordinates.")
#
#    def parse_atom(self, length=None):
#        """
#        Parse the atom dataframe.
#
#        Args:
#            length (str): String length unit
#        """
#        if length is None and "angstrom" in str(self[0]).lower():
#            length = "Angstrom"
#        else:
#            length = "au"
#        if "end" in str(self[-2]).lower():
#            slce = slice(self._int0, -2)
#        else:
#            slce = slice(self._int0, -1)
#        atom = pd.read_csv(self[slce].to_stream(), delim_whitespace=True,
#                           names=self._cols)
#        self.atom = Atom.from_xyz(atom, unit=length)
#
#    def _parse_end(self, starts):
#        """Find the next blank line."""
#        return [self.next_blank_line(cursor=i[0]) for i in starts]
