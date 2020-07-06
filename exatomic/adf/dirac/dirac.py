## -*- coding: utf-8 -*-
## Copyright (c) 2015-2020, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#ADF Dirac Parser
##########################
#Parser for the output of the DIRAC program (part of the ADF suite).
#"""
#from exa import Parser, Matches
#
#
#class DIRAC(Parser):
#    """
#    """
#    _start =  "*                            |     D I R A C     |                            *"
#    _stop = -1
#
#    def _parse_stops_1(self, starts):
#        """Find the end of the section."""
#        key = "Hash table lookups:"
#        matches = [self.find_next(key, cursor=s[0]) for s in starts]
#        return Matches(key, *matches)
