## -*- coding: utf-8 -*-
## Copyright (c) 2015-2020, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#Parser for ADF's SCF Section
################################
#Parses the 'S C F' section of an ADF block output.
#"""
#from exa import Parser, Matches, Match
#
#
#class SCF(Parser):
#    """
#    """
#    _start = " S C F"
#    _stop = -1
#
#    def _parse_stops_1(self, starts):
#        """
#        """
#        key = "  Integrated fractional orbital density"
#        matches = []
#        for start in starts:
#            m = self.find_next(key, cursor=start[0])
#            if m is None:
#                matches.append(Match(len(self), None))
#            else:
#                matches.append(m)
#        return Matches(key, *matches)
