# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Parser for ADF's SCF Section
###############################
Parses the 'S C F' section of an ADF block output.
"""
from exa import Parser, Matches


class SCF(Parser):
    """
    """
    _start = " S C F"
    _stop = -1

    def _parse_stops_1(self, starts):
        """
        """
        key = "  Integrated fractional orbital density"
        return Matches(key, *[self.find_next(key, cursor=s[0]) for s in starts])
