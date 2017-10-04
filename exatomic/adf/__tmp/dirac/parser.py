## -*- coding: utf-8 -*-
## Copyright (c) 2015-2017, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#Composite ADF Output File Editor
####################################
#The :class:`~exatomic.adf.output.CompositeOutput` object is a general entry
#point for ADF output files. Most output files can be read by this class.
#
#.. code-block:: python
#
#    outputfile = "path-to-file"
#    out = CompositeOutput(outputfile)
#    out.sections    # List the specific output sections detected
#"""
#import re
#import numpy as np
#import pandas as pd
#from exa import Sections, TypedProperty, DataFrame, Parser
#
#
#class DIRACArray(Parser):
#    _key_n = 9
#    _key_pat = "(\d)([+-])(\d)"
#    _key_sub = r"\1E\2\3"
#    name = TypedProperty(str)
#    title = TypedProperty(str)
#    array = TypedProperty(pd.Series)
#
#    def _parse(self):
#        """
#        """
#        self.title = str(self[0]).strip()
#        array = self[2:].to_data("fwf", widths=[2]+[14]*8, names=range(self._key_n))
#        del array[0]
#        array = array.applymap(lambda x: re.sub(self._key_pat, self._key_sub, str(x))).astype(float)
#        array = array.values.ravel()
#        self.array = array[~np.isnan(array)]
#
#
#class DIRAC(Sections):
#    """
#    A parser for the 'D I R A C' section(s) of an ADF calculation.
#
#    The 'dirac' program solves the all-electron radial problem for a variety
#    of Hamiltonians (default is the Dirac-Slater Hamiltonian with Slater's
#    alpha value modified to 0.7).
#    """
#    _key_d0 = "^1"
#    _key_d1 = "^\n, [ -]0"
#    _key_d2 = "Divide hfi by"
#    _key_d3 = "D I R A C   E X I T"
#    _key_def = ("header", "DiracInput", "Orbital", "footer")
#    orbital = TypedProperty(DataFrame)
#
#    def _parse(self):
#        """Identify sub-sections for standard and debug output."""
#        delims = self.regex(self._key_d0, self._key_d1, self._key_d2,
#                            self._key_d3, text=False)
#        starts = [0]
#        parsers = []
#        titles = []
#        self.delims = delims
#        i = 0
#        for no in delims[self._key_d0]:
#            starts.append(no)
#            parsers.append(self._key_def[i])
#            titles.append(self._key_def[i])
#            i += 1
#        parsers.append(self._key_def[i])
#        titles.append(self._key_def[i])
#        i += 1
#        if self._key_d1 in delims:
#            for no in delims[self._key_d1]:
#                starts.append(no-2)
#                titles.append(str(self[no-2]))
#                parsers.append(DIRACArray)
#            no = delims[self._key_d2][0]
#            starts.append(no - 3)
#            titles.append(str(self[no - 2]))
#            parsers.append(None)
#        no = delims[self._key_d3][0]
#        starts.append(no - 2)
#        titles.append(str(self[no]))
#        parsers.append(None)
#        ends = starts[1:]
#        ends.append(len(self))
#        self._sections_helper(start=starts, end=ends, parser=parsers, title=titles)
