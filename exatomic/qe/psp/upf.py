## -*- coding: utf-8 -*-
## Copyright (c) 2015-2020, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#Unified Pseudopotential Format (UPF)
###########################################
#Parsers for UPF files.
#"""
#import numpy as np
#import pandas as pd
#from exa import Sections, Parser, TypedProperty
#
#
#class UPFArray(Parser):
#    """
#    """
#    _key_d0 = ">"
#    _key_d1 = "="
#    _key_s = ("<", "")
#    _key_r = 4
#    array = TypedProperty(pd.Series)
#    meta = TypedProperty(dict)
#    name = TypedProperty(str)
#
#    def _parse(self):
#        """Identify data containing rows and add metadata from tags."""
#        self.name = str(self[0]).strip().split()[0].replace(*self._key_s)
#        ends = self.find(self._key_d0, text=False)[self._key_d0]
#        bounds = (ends[0] + 1, ends[1])
#        array = self[ends[0]+1:ends[1]].to_data(delim_whitespace=True, names=range(self._key_r)).values.astype(float).ravel()
#        array = array[~np.isnan(array)]
#        self.array = array
#        text = str(self[:ends[0]+1]).split()
#        meta = {}
#        for txt in text:
#            if self._key_d1 in txt:
#                k, v = txt.strip().split(self._key_d1)
#                meta[k] = v
#        self.meta = meta
#
#
#class UPF(Sections):
#    """
#    """
#    _key_d0 = "<[A-z0-9_\.]+"
#    _key_d1 = "<\/[A-z0-9_\.]+"
#    _key_ck = 16
#
#    def _parse(self):
#        """Parse the xml-like format by matching tags."""
#        size = None
#        starts = []
#        ends = []
#        titles = []
#        parsers = []
#        tags = self.regex(self._key_d0, self._key_d1)
#        pop = tags[self._key_d1][:]
#        for stline, sttxt in tags[self._key_d0]:
#            for endline, endtxt in pop:
#                if sttxt[1:] == endtxt[2:]:
#                    starts.append(stline-1)
#                    ends.append(endline)
#                    titles.append(sttxt[1:])
#                    line0 = self._lines[stline-1]
#                    line1 = self._lines[stline]
#                    if (((">" in line0 and "<" not in line1 and ">" not in line1) or
#                         (">" not in line0 and ">" in line1)) and
#                        ("E+" in line1 or "E-" in line1)):
#                        parsers.append(UPFArray)
#                    else:
#                        parsers.append(None)
#                    pop.remove((endline, endtxt))
#                    break
#        self._sections_helper(start=starts, end=ends, title=titles, parser=parsers)
