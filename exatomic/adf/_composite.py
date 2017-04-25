# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Composite ADF Output File Editor
###################################
"""
import pandas as pd
from exa.core import Sections
from .nmr.output import NMROutput


class CompositeOutput(Sections):
    """
    Generic ADF output file containing many types of executable calls.
    """
    name = "Composite ADF output"
    description = "General parser for ADF output files."
    _key_exe_delim = " *   Amsterdam Density Functional  (ADF)"
    _key_exe_plus = 7
    _key_exe_minus = -4

    def _parse(self):
        """Identify all sections in the composite output."""
        delims = self.find(self._key_exe_delim, which='lineno')[self._key_exe_delim]
        titles = []
        for i in delims:
            title = str(self[i + self._key_exe_plus])
            title = title.replace("*", "").replace("|", "").strip()
            title = title.replace(" ", "_")
            titles.append(title)
        names = [self._gen_sec_attr_name(i) for i in range(len(titles))]
        starts = [i + self._key_exe_minus for i in delims]
        ends = starts[1:]
        ends.append(len(self))
        self.sections = pd.DataFrame.from_dict({"name": names, "titles": titles,
                                                "starts": starts, "ends": ends})


CompositeOutput.add_section_parsers(NMROutput)
