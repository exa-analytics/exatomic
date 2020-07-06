# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Geometry Parser ('A D F' Calculations)
#########################################
"""
#import re
#import pandas as pd
#from exa import Parser, Typed
#from exatomic.core.atom import Atom
#
#
#class Geometry(Parser):
#    """
#    Parser for the data block labeled 'Coordinates (Cartesian)' in 'A D F' output sections.
#    """
#    _start = " Coordinates (Cartesian)"
#    atom = Typed(Atom, doc="Coordinates")
#
#    def _parse_end(self, starts):
#        return [self.find_next_blank_line(cursor=i[0]) for i in starts]




#    _stop = -1
#    _skey = re.compile(" Number of elements of the density matrix on this node|^1")
#    _skey1 = "----"
#    _swidths = (7, 4, 12, 12, 12, 15, 12, 12, 7, 8, 8)
#    _snames = ("label", "symbol", "x", "y", "z", "dx", "dy", "dz", "lx", "ly", "lz")
#    atom = Typed(DataFrame)
#
#    def parse_atom(self):
#        self.parse()
#
#    def _parse_stops_1(self, starts):
#        """
#        """
#        return Matches(self._skey, *[self.regex_next(self._skey, cursor=s[0]) for s in starts])
#
#    def _parse(self):
#        """
#        """
#        start, stop = [m.num for m in self.find(self._skey1).as_matches().matches]
#        atom = pd.read_fwf(self[start+1:stop].to_stream(), widths=self._swidths,
#                           names=self._snames)
#        for i in range(5, len(atom.columns)):
#            del atom[atom.columns[-1]]
#        self.atom = atom
