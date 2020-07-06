## -*- coding: utf-8 -*-
## Copyright (c) 2015-2020, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#Parser for ADF Output
##########################
#Parser for the output of the DIRAC program (part of the ADF suite).
#"""
#import pandas as pd
#from exa import Parser, Matches, Typed
#from .scf import SCF
#from .geometry import Geometry
#from exatomic.core import Atom
#from exatomic.base import sym2z
#
#
#
#class ADF(Parser):
#    """
#    """
#    _start = "*                              |     A D F     |                              *"
#    _stop = -1
#    atom = Typed(Atom)
#
#    def _parse_stops_1(self, starts):
#        """Find the end of the section."""
#        key = "Hash table lookups:"
#        matches = [self.find_next(key, cursor=s[0]) for s in starts]
#        return Matches(key, *matches)
#
#    def parse_atom(self):
#        self.parse()
#
#    def _parse(self):
#        self._parse_atom()
#
#    def _parse_atom(self):
#        """Parse the atom table."""
#        secs = self.sections[self.sections['parser'] == Geometry].index
#        atom = []
#        for i, j in enumerate(secs):
#            atm = self.get_section(j).atom
#            atm['frame'] = i
#            atom.append(atm)
#        atom = pd.concat(atom, ignore_index=True)
#        atom['Z'] = atom['symbol'].map(sym2z)
#        del atom['symbol']
#        self.atom = Atom(atom)
#
#
#ADF.add_parsers(SCF, Geometry)
