## -*- coding: utf-8 -*-
## Copyright (c) 2015-2020, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#PSLibrary Job File
######################
#Job files (extension '.job') are used inside the `pslibrary`_ package to house
#input data for pseudopotential generation using the atomic sub-package ('ld1.x')
#within the `Quantum ESPRESSO`_ quantum chemistry suite of tools. See for example,
#`this`_ job file.
#
#.. _pslibrary: https://github.com/dalcorso/pslibrary
#.. _Quantum ESPRESSO: https://github.com/QEF/q-e
#.. _this: https://github.com/dalcorso/pslibrary/blob/master/paw_ps_collection.job
#"""
#import re
#import numpy as np
#import pandas as pd
#from exa import isotopes, Sections, Parser, TypedProperty, DataFrame
#
#
#class Element(Parser):
#    """A single element's input file in the composite job file."""
#    _key_config = "config"
#    _key_ae_dct = {}
#    _key_mrk = "["
#    _key_resplit = re.compile("([1-9]*)([spdfghjklmn])([0-9-.]*)")
#    _key_symbol = "title"
#    _key_zed = "zed"
#    _key_ps = "/"
#    _key_ps_cols = ("n", "l_sym", "nps", "l", "occupation",
#                    "energy", "rcut_nc", "rcut", "misc")
#    _key_ps_dtypes = [np.int64, "O", np.int64, np.int64, np.float64,
#                      np.float64, np.float64, np.float64, np.float64]
#    ae = TypedProperty(DataFrame)
#    ps = TypedProperty(DataFrame)
#    z = TypedProperty(int)
#    symbol = TypedProperty(str)
#
#    def _parse(self):
#        if str(self[0]).startswith("#"):
#            return
#        found = self.find(self._key_config, self._key_symbol,
#                          self._key_zed, self._key_ps)
#        config = found[self._key_config][-1][1].split("=")[1]
#        config = config.replace("'", "").replace(",", "").split(" ")
#        nvals = []
#        angmoms = []
#        occs = []
#        for item in config:
#            if "[" in item:
#                continue
#            try:
#                nval, angmom, occ = self._key_resplit.match(item.lower()).groups()
#                nvals.append(nval)
#                angmoms.append(angmom)
#                occs.append(occ)
#            except AttributeError:
#                pass
#        self.ae = pd.DataFrame.from_dict({'n': nvals, 'l': angmoms, 'occupation': occs})
#        self.symbol = found[self._key_symbol][-1][1].split("=")[1].replace("'", "").replace(",", "").title()
#        element = getattr(isotopes, self.symbol)
#        self.z = element.Z
#        ps = []
#        for line in self[found[self._key_ps][-1][0]:]:
#            if "#" in line:
#                continue
#            ls = line.split()
#            if len(ls) > 7:
#                dat = list(self._key_resplit.match(ls[0].lower()).groups())[:-1]
#                dat += ls[1:]
#                ps.append(dat)
#        self.ps = pd.DataFrame(ps, columns=self._key_ps_cols)
#        for i, col in enumerate(self.ps.columns):
#            self.ps[col] = self.ps[col].astype(self._key_ps_dtypes[i])
#
#
#class PSLJobFile(Sections):
#    """Input 'job' file in the pslibrary"""
#    name = "pslibrary job file"
#    description = "Parser for pslibrary input files"
#    _key_sep = "EOF"
#    _key_parser = Element
#
#    def _parse(self):
#        """Parse input data from pslibrary"""
#        delims = self.find(self._key_sep, text=False)[self._key_sep]
#        starts = delims[::2]
#        ends = delims[1::2]
#        names = [self._key_parser]*len(ends)
#        self._sections_helper(parser=names, start=starts, end=ends)
