## -*- coding: utf-8 -*-
## Copyright (c) 2015-2017, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#NMR Nucleus Subsection
#######################
#This subsection contains calculation shielding tensors.
#"""
#import numpy as np
#import pandas as pd
#from exa import Sections, Parser
#from exa.typed import TypedProperty
#
#
#class NMRNucleusInfoParser(Parser):
#    """Parses the informational part of the 'N U C L E U S :' subsection of an ADF NMR output."""
#    _key_marker = ":"
#    _key_symmrk = "("
#    _key_repmrk = (")", "")
#    symbol = TypedProperty(str, docs="Isotope symbol")
#    adfid = TypedProperty(int, docs="Atom ID in adf executable")
#    nmrid = TypedProperty(int, docs="Atom ID in nmr executable")
#
#    def _parse(self):
#        """Get atom numbering"""
#        for text in self.find(self._key_marker, num=False)[self._key_marker][1:]:
#            txt, atom = text.split(self._key_marker)
#            symbol, number = atom.replace(*self._key_repmrk).split(self._key_symmrk)
#            if "ADF" in txt:
#                self.adfid = number
#            else:
#                self.nmrid = number
#        self.symbol = symbol.strip()
#
#
#class NMRNucleusTensorParserMixin(object):
#    """Common parsing attributes for these related section parsers."""
#    _key_forder = 'F'    # ravel order
#    # Single tensor fixed width fortran printing
#    _key_1tensor = [26, 10, 10, 10]
#    # Double tensor fixed width fortran printing
#    _key_2tensor = [6, 10, 10, 10, 10, 10, 10, 10]
#    # PAS principle fixed with fortran printing
#    _key_pas_widths = [36, 10, 10]
#    _key_first = slice(3, 12)
#    _key_second = slice(15, None)
#    _key_nwidths3 = 3
#    _key_nwidths3 = 3
#    _key_split = "="
#    _key_rep = ("isotropic", "")
#    _key_range3 = range(3)
#    _key_range6 = range(6)
#    _key_range8 = range(8)
#    _key_idx_pas = ["sigma11", "sigma22", "sigma33"]
#    _key_idx_iso = "iso"
#    _key_idx_ijk = ["x", "y", "z"]
#    _key_dtype = np.float64
#    ds = TypedProperty(pd.Series, "Shielding tensor array for the current atom")
#
#    @property
#    def _key_index(self):
#        """Build the dataseries index."""
#        index = []
#        for prefix in self._key_idx_prefix:
#            for alpha in self._key_idx_ijk:
#                for beta in self._key_idx_ijk:
#                    index.append(self._key_typ + prefix + alpha + beta)
#            if prefix != self._key_idx_prefix[-1]:
#                index.append(self._key_typ + prefix + self._key_idx_iso)
#        for value in self._key_idx_pas:
#            index.append(self._key_typ + value)
#        return index
#
#    def _finalize_parse(self, data=None):
#        """Parse the cartesian and principal axis representations and set the data."""
#        cart = self[self._key_cart].to_data("fwf", widths=self._key_1tensor,
#                                            names=self._key_range3).values.ravel(self._key_forder)
#        cartiso = str(self[self._key_cartiso]).split(self._key_split)[1].strip()
#        if data is None:
#            data = cart.tolist()
#        else:
#            data += cart.tolist()
#        data.append(cartiso)
#
#        pas = self[self._key_pas].to_data("fwf", widths=self._key_1tensor,
#                                          names=self._key_range3).values.ravel()
#        data += pas.tolist()
#        pas3 = self[self._key_pas3].to_data("fwf", widths=self._key_pas_widths,
#                                            names=self._key_range3).values.ravel()
#        data += pas3.tolist()
#
#        self.ds = pd.to_numeric(pd.Series(data, index=self._key_index), errors='coerce')
#
#
#class NMRNucleusParaParser(Parser, NMRNucleusTensorParserMixin):
#    """
#    Parses the paramagnetic NMR shielding tensors.
#    """
#    _key_b1u1 = slice(6, 9)
#    _key_b1u1iso = 10
#    _key_s1gauge = slice(14, 17)
#    _key_s1gaugeiso = 18
#    _key_cart = slice(25, 28)
#    _key_cartiso = 29
#    _key_pas = slice(41, 44)
#    _key_pas3 = 37
#    _key_idx_prefix = ["b1", "u1", "s1", "gauge", "cart", "pas"]
#    _key_typ = "para"
#
#    def _parse(self):
#        """Parse the paramagnetic shielding tensor data."""
#        b1u1 = self[self._key_b1u1].to_data("fwf", widths=self._key_2tensor,
#                                            names=self._key_range8).values.ravel(self._key_forder)
#        b1iso, u1iso = str(self[self._key_b1u1iso]).strip().split(self._key_split)[1:]
#        data = b1u1[self._key_first].tolist()
#        data.append(b1iso.replace(*self._key_rep).strip())
#        data += b1u1[self._key_second].tolist()
#        data.append(u1iso.strip())
#
#        s1gauge = self[self._key_s1gauge].to_data("fwf", widths=self._key_2tensor,
#                                                  names=self._key_range8).values.ravel(self._key_forder)
#        s1iso, gaugeiso = str(self[self._key_s1gaugeiso]).strip().split(self._key_split)[1:]
#        data += s1gauge[self._key_first].tolist()
#        data.append(s1iso.replace(*self._key_rep).strip())
#        data += s1gauge[self._key_second].tolist()
#        data.append(gaugeiso.strip())
#
#        self._finalize_parse(data)
#
#
#class NMRNucleusDiaParser(Parser, NMRNucleusTensorParserMixin):
#    """
#    description = "Parses the diamagnetic NMR shielding tensors."
#    """
#    _key_cv = slice(6, 9)
#    _key_cviso = 10
#    _key_cart = slice(17, 20)
#    _key_cartiso = 21
#    _key_pas = slice(33, 36)
#    _key_pas3 = 29
#    _key_idx_prefix = ["core", "valence", "cart", "pas"]
#    _key_typ = "dia"
#
#    def _parse(self):
#        """Parse the diagmentic shielding tensor data."""
#        cv = self[self._key_cv].to_data("fwf", widths=self._key_2tensor,
#                                        names=self._key_range8).values.ravel(self._key_forder)
#        ciso, viso = str(self[self._key_cviso]).strip().split(self._key_split)[1:]
#        data = cv[self._key_first].tolist()
#        data.append(ciso.replace(*self._key_rep))
#        data += cv[self._key_second].tolist()
#        data.append(viso.strip())
#
#        self._finalize_parse(data)
#
#
#class NMRNucleusTotParser(Parser, NMRNucleusTensorParserMixin):
#    """
#    Parses the total NMR shielding tensors.
#    """
#    _key_cart = slice(9, 12)
#    _key_cartiso = 13
#    _key_pas = slice(25, 28)
#    _key_pas3 = 21
#    _key_idx_prefix = ["cart", "pas"]
#    _key_typ = "tot"
#
#    def _parse(self):
#        """Parse total NMR shielding tensors."""
#        self._finalize_parse()
#
#
#class NMRNucleusParser(Sections):
#    """
#    Parses the 'N U C L E U S :' subsection of an ADF NMR output.
#
#    Collates individual arrays corresponding to different shielding tensor
#    data (e.g. paramagnetic, diamagnetic, total, etc.) into a single table.
#    """
#    _key_sep = "^=+$"
#    _key_start = 0
#    _key_sec_name_p = 2
#    _key_sec_name_id = 0
#    _key_title_rep0 = ("=== ", "")
#    _key_title_rep1 = ("UNSCALED: ", "")
#    _key_parser_name_rep = ("-", "")
#    _key_info_parser = NMRNucleusInfoParser
#    _key_info_name = "INFO"
#    _key_parsers = {'PARAMAGNETIC': NMRNucleusParaParser,
#                    'DIAMAGNETIC': NMRNucleusDiaParser,
#                    'TOTAL': NMRNucleusTotParser}
#
#    def _parse(self):
#        """Parser out what shielding tensor components are present."""
#        # First parse sub-sections
#        starts = [self._key_start] + self.regex(self._key_sep, text=False)[self._key_sep]
#        ends = starts[1:]
#        parsers = [self._key_info_parser]
#        titles = [self._key_info_name]
#        for start in ends:
#            title = str(self[start + self._key_sec_name_p])
#            title = title.replace(*self._key_title_rep0)
#            title = title.replace(*self._key_title_rep1)
#            titles.append(title)
#            parser = title.split()[self._key_sec_name_id]
#            parser = parser.replace(*self._key_parser_name_rep)
#            parsers.append(self._key_parsers[parser])
#        ends.append(len(self))
#        self._sections_helper(start=starts, end=ends, parser=parsers, title=titles)
#
#
