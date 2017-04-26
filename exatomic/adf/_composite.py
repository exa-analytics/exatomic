# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Composite ADF Output File Editor
###################################
"""
import pandas as pd
from exa.core import Sections
from .mixin import OutputMixin
from .nmr.output import NMROutput


class CompositeOutput(Sections, OutputMixin):
    """
    Most general ADF output file parser.

    This object accepts the most general type of ADF output file that contains
    output structures from many different calculations.
    """
    name = "CompositeOutput"
    description = "Generic ADF output file parser"
    _key_exe_delim = " *   Amsterdam Density Functional  (ADF)"
    _key_exe_plus = 7
    _key_exe_minus = -4
    _key_title_rep = (("*", ""), ("|", ""))
    _key_parser_rep = (" ", "")

    def _parse(self):
        """Identifies all sub-sections (calculations) in the composite output file."""
        # Find executable tags
        delims = self.find(self._key_exe_delim, text=False)[self._key_exe_delim]
        # Determine calculation names
        parser_names = []
        titles = []
        for i in delims:
            title = str(self[i + self._key_exe_plus])
            for args in self._key_title_rep:
                title = title.replace(*args)
            titles.append(title.strip())
            parser = title.replace(*self._key_parser_rep)
            parser_names.append(parser)
        starts = [i + self._key_exe_minus for i in delims]
        ends = starts[1:]
        ends.append(len(self))
        dct = {'parser': parser_names, 'start': starts, 'end': ends, 'title': titles}
        self._sections_helper(pd.DataFrame.from_dict(dct))


CompositeOutput.add_section_parsers(NMROutput)
