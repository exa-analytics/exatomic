# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Composite ADF Output File Editor
###################################
The :class:`~exatomic.adf.output.CompositeOutput` object is a general entry
point for ADF output files. Most output files can be read by this class.

.. code-block:: python

    outputfile = "path-to-file"
    out = CompositeOutput(outputfile)
    out.sections    # List the specific output sections detected
"""
from exa import Sections
from .mixin import OutputMixin
from .dirac import DIRAC
from .nmr import NMR
from .adf import ADF


parser_aliases = {'DIRAC': DIRAC, 'NMR': NMR, 'ADF': ADF}


class CompositeOutput(Sections, OutputMixin):
    """
    Most general ADF output file parser.

    This object accepts the most general type of ADF output file that contains
    output structures from many different calculations.
    """
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
            parser_names.append(parser_aliases.get(parser, parser))
        starts = [i + self._key_exe_minus for i in delims]
        ends = starts[1:]
        ends.append(len(self))
        self._sections_helper(parser=parser_names, start=starts, end=ends, title=titles)
