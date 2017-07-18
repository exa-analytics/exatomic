# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Standard Output Parser
#############################
This module provides an output parser for the standard output file coming from
``pw.x`` calculations. The standard output prints information about the cell,
size of the basis, and self consistent field loop data.
"""
from exa import Sections
from .header import Header
from .footer import Footer
from .scf import SCF


class PWOutput(Sections):
    """
    """
    _key_delims = ("Self-consistent Calculation", "Writing output data file ")
    _key_s = SCF
    _key_h = Header
    _key_f = Footer

    def _parse(self):
        """Identify subsections of the composite output"""
        delims = self.find(*self._key_delims, text=False)
        self.delims = delims
        starts = [0] + delims[self._key_delims[0]] + [delims[self._key_delims[1]][-1]]
        ends = starts[1:]
        ends.append(len(self))
        parsers = [self._key_h] + [self._key_s]*len(delims[self._key_delims[0]]) + [self._key_f]
        self._sections_helper(parser=parsers, start=starts, end=ends)
