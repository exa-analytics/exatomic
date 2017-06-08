# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
ADF Output Editor
###################################
Parser for "A D F" output sections of a ADF calculation. The main object that
should be used is :class:`~exatomic.adf.adf.output.ADF`
"""
from exa import Sections
from .results import RESULTS


parser_aliases2 = {'RESULTS': RESULTS}


class ADF(Sections):
    """
    Output parser for an 'A D F' calculation.
    """
    _key_delim0 = "^\s{40,}\*[ A-Z]{15,30}\*$"
    _key_delim1 = "\(LOGFILE\)"
    _key_start_minus = -1
    _key_starts = 0
    _key_names = "HEADER"    # Custom name for the first part of an ADF output section

    def _parse(self):
        """
        """
        delims = self.regex(self._key_delim0, self._key_delim1, text=False)
        starts = [self._key_starts]
        names = [self._key_names]
        ends = []
        for i in delims[self._key_delim0] + delims[self._key_delim1]:
            start = i + self._key_start_minus
            name = str(self[start]).replace("(", "").replace(")", "")
            name = name.replace("*", "").strip().replace(" ", "")
            starts.append(start)
            names.append(parser_aliases2[name] if name in parser_aliases2 else name)
            ends.append(start)
        ends.append(len(self))
        self._sections_helper(start=starts, end=ends, parser=names)
