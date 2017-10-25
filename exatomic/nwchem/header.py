# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Parser for Output File Header
#################################
Standalone parser for the calculation header.
"""
from exa import Parser, Typed


class Header(Parser):
    """
    Parser for the header of an NWChem calculation.

    Constructs a dictionary of 'metadata' related to calculation
    being performed such as code version, time and date, and processor count.
    Additionally, the input deck is occasionally printed here.
    """
    _ek = "   NWChem Input Module"
    _i0 = 0
    _i1 = -1
    _key0 = " = "
    info = Typed(dict, doc="Metadata about the calculation as a dict")

    def parse_info(self):
        """
        Parse metadata about the calculation from the header (code version,
        dates, etc.)

        Iterate line by line to see if the delimter (_key0) is present. If
        it is, split the line taking the first and last items as key,value
        pairs.
        """
        info = {}
        for line in self:
            cnt = line.count(self._key0)
            if cnt > 0:
                split = line.split(self._key0)
                key = split[self._i0].strip()
                value = split[self._i1].strip()
                info[key] = value
        self.info = info

    def _parse_both(self):
        """Used by internals for parsing 'sections'."""
        end = self.find_next(self._ek, cursor=0)[0]
        return [(0, self.lines[0])], [(end, self.lines[end])]
