## -*- coding: utf-8 -*-
## Copyright (c) 2015-2017, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#EFG Section Parser
####################################
#Parser for the "EFG" (sub)section of an "A D F" block. The parser in this
#module should not typically be used directly. Rather users should open their
#file via the :class:`~exatomic.adf.adf.output.ADF` or
#:class:`~exatomci.adf.output.CompositeOutput` parsers.
#"""
#import re
#import pandas as pd
#from exa import Sections, Parser, TypedProperty, DataFrame
#
#
#class AtomEFG(Parser):
#    """
#    A subsection of the EFG results giving results for a specific atom
#    """
#    ds = TypedProperty(pd.Series, docs="Labeled array of EFG for a specific atom")
#    _key_delim = "="
#    _key_data_fwf = "fwf"
#    _key_info = slice(1, 6)
#    _key_info_widths = [23, 47]
#    _key_info_index = ["isotope", "I", "mu", "g_n", "Q"]
#    _key_info_cols = ["del", "val"]
#    _key_xyz = 8
#    _key_xyz_cols = [1, 2, 5]
#    _key_xyz_names = ["label", "symbol", "input", "x", "y", "z"]
#    _key_xyz_rep = (" cm2", "")
#    _key_efg_xyz = ["x", "y", "z"]
#    _key_efg_123 = ["11", "22", "33"]
#    _key_efg = slice(11, 15)
#    _key_efg_del_col = 0
#    _key_efg_widths = [9, 14, 14, 14]
#    _key_efg_cart_prefix = "V"
#    _key_efg_pas_prefix = "Q"
#    _key_efg_pas = slice(17, 21)
#    _key_pas_line = 23
#    _key_qtens_line = 27
#    _key_lim = -1
#    _key_eta = "eta"
#    _key_eta_line = 29
#
#    @property
#    def _key_cart_index(self):
#        """
#        """
#        return [self._key_efg_cart_prefix+a+b for a in self._key_efg_xyz for b in self._key_efg_xyz]
#
#    @property
#    def _key_pas_index(self):
#        """
#        """
#        return [self._key_efg_pas_prefix+a+b for a in self._key_efg_xyz for b in self._key_efg_xyz]
#
#    @property
#    def _key_pas_names(self):
#        """
#        """
#        return [self._key_efg_cart_prefix+i for i in self._key_efg_123]
#
#    @property
#    def _key_qtens_names(self):
#        """
#        """
#        return [self._key_efg_pas_prefix+i for i in self._key_efg_123]
#
#    def _parse(self):
#        """
#        """
#        # Stuff between ----- lines
#        ds = self[self._key_info].to_data(self._key_data_fwf, widths=self._key_info_widths,
#                                          names=self._key_info_cols)[self._key_info_cols[1]]
#        ds.index = self._key_info_index
#        # Atom info line
#        numsym, xyz = str(self[self._key_xyz]).split(self._key_delim)
#        numsymsplit = numsym.split()
#        dat = [numsymsplit[i] for i in self._key_xyz_cols] + xyz.split()
#        for i, name in enumerate(self._key_xyz_names):
#            ds[name] = dat[i]
#        ds[self._key_efg_pas_prefix] = ds[self._key_efg_pas_prefix].replace(*self._key_xyz_rep)
#        # Cartesian EFG tensor (au, lab frame of reference)
#        array = self[self._key_efg].to_data(self._key_data_fwf, widths=self._key_efg_widths)
#        del array[array.columns[self._key_efg_del_col]]
#        ds = pd.concat((ds, pd.Series(array.values.ravel(), index=self._key_cart_index)))
#        # Principle axis Q-tensor (EFG)
#        array = self[self._key_efg_pas].to_data(self._key_data_fwf, widths=self._key_efg_widths)
#        del array[array.columns[self._key_efg_del_col]]
#        ds = pd.concat((ds, pd.Series(array.values.ravel(), index=self._key_pas_index)))
#        # EFG components in the principle axis system
#        array = str(self[self._key_pas_line]).split()[:self._key_lim]
#        names = self._key_pas_names
#        for i, item in enumerate(array):
#            ds[names[i]] = item
#        # Q-tensor principle components
#        array = str(self[self._key_qtens_line]).split()[:self._key_lim]
#        names = self._key_qtens_names
#        for i, item in enumerate(array):
#            ds[names[i]] = item
#        # Asymmetry parameter (eta)
#        ds[self._key_eta] = str(self[self._key_eta_line]).split()[self._key_lim]
#
#        self.ds = ds
#
#
#class EFG(Sections):
#    """
#    Parsers EFG section from EFG subsection of 'R E S U L T S' subsection of 'A D F' calculation.
#    """
#    df = TypedProperty(DataFrame, docs="Full table of EFG data")
#    _key_d0 = r"^\n -+$"
#    _key_parser = AtomEFG
#
#    def _parse(self):
#        """
#        """
#        delims = self.regex(self._key_d0, text=False)[self._key_d0]
#        starts = delims
#        ends = delims[1:]
#        ends.append(len(self))
#        names = [self._key_parser]*len(starts)
#        self._sections_helper(parser=names, start=starts, end=ends)
#
#    def _get_df(self):
#        """
#        """
#        dss = []
#        for section in self.itersections():
#            dss.append(section.ds)
#        self.df = pd.concat(dss, axis=1).T
##        for col in self.df.columns:
##            try:
##                self.df[col] = pd.to_numeric(self.df[col])
##            except ValueError:
##                try:
##                    self.df[col] = pd.to_numeric(self.df[col].str.replace(" e 10", "E"))
##                except ValueError:
##                    pass
#
#
#
