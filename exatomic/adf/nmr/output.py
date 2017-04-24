# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NMR Output Editors
######################
"""
import six
import numpy as np
import pandas as pd
from exa.core import SectionsMeta, Parser, Sections, DataSeries, DataFrame


class NMR(Sections):
    """ADF NMR output file parsing."""
    name = "ADF NMR output"
    description = "Parses an NMR output file"
    _key_sep = "################################################################################"
    _key_sec_names = ["metadata", "info"]
    _key_convergence = "NOT CONVERGED"

    @property
    def converged(self):
        """Checked for convergence."""
        if self._key_convergence in self:
            return False
        return True

    def _parse(self):
        """Determine all sections."""
        delims = self.find(self._key_sep, which='lineno')[self._key_sep][1:]
        starts = [delim + 1 for delim in delims]
        starts.insert(0, 0)
        ends = delims
        ends.append(len(self))
        names = self._key_sec_names + ["nucleus"]*(len(delims) - 1)
        self.sections = list(zip(names, starts, ends))


class NMRMetadataParserMeta(SectionsMeta):
    """Defines the objects parsed by NMRMetadataParser."""
    _descriptions = {}


class NMRMetadataParser(six.with_metaclass(NMRMetadataParserMeta, Parser)):
    """"""
    name = "metadata"
    description = "Parser for the title and citations section of an ADF NMR output."

    def _parse(self):
        pass


class NMRInfoParserMeta(SectionsMeta):
    """Defines the objects parsed by NMRInfoParser."""
    _descriptions = {}


class NMRInfoParser(six.with_metaclass(NMRInfoParserMeta, Parser)):
    """"""
    name = "info"
    description = "Parser for the information section (geometry and basis) of an ADF NMR output."

    def _parse(self):
        pass


class NMRNucleusParser(Sections):
    """
    Parses the 'N U C L E U S :' subsection of an ADF NMR output.
    """
    name = "nucleus"
    description = "Parses NMR shielding tensors from the nucleus section."
    _key_sep = "================================================================================"
    _key_sec_names = ["info", "paramagnetic", "diamagnetic", "total"]

    def _parse(self):
        """Get the section line numbers."""
        starts = self.find(self._key_sep, which='lineno')[self._key_sep]
        starts.insert(0, 0)
        ends = starts[1:]
        ends.append(len(self))
        self.sections = list(zip(self._key_sec_names, starts, ends))


class NMRNucleusInfoParserMeta(SectionsMeta):
    """Defines objects parsed by NMRNucleusInfoParser."""
    symbol = str
    adfid = int
    nmrid = int
    _descriptions = {'symbol': "Atom type", 'adfid': "Atom input number in ADF calculation",
                     'nmrid': "Internal NMR atom numbering"}


class NMRNucleusInfoParser(six.with_metaclass(NMRNucleusInfoParserMeta, Parser)):
    """Parses the informational part of the 'N U C L E U S :' subsection of an ADF NMR output."""
    name = "info"
    description = "Parses the atom numbering part of the 'N U C L E U S :' subsection of an ADF NMR output."
    _key_marker = ":"
    _key_symmrk = "("
    _key_repmrk = (")", "")

    def _parse(self):
        """Get atom numbering"""
        for text in self.find(self._key_marker, which='text')[self._key_marker][1:]:
            txt, atom = text.split(self._key_marker)
            symbol, number = atom.replace(*self._key_repmrk).split(self._key_symmrk)
            if "ADF" in txt:
                self.adfid = number
            else:
                self.nmrid = number
        self.symbol = symbol.strip()


class NMRNucleusTensorParserMeta(SectionsMeta):
    """Defines objects parsed by NMRNucleusParaParser"""
    ds = DataSeries
    _descriptions = {'ds': "Shielding tensor data"}


class NMRNucleusTensorParserMixin(object):
    """Common parsing attributes for these related section parsers."""
    _key_forder = 'F'    # ravel order
    # Single tensor fixed width fortran printing
    _key_1tensor = [26, 10, 10, 10]
    # Double tensor fixed width fortran printing
    _key_2tensor = [6, 10, 10, 10, 10, 10, 10, 10]
    # PAS principle fixed with fortran printing
    _key_pas_widths = [36, 10, 10]
    _key_first = slice(3, 12)
    _key_second = slice(15, None)
    _key_nwidths3 = 3
    _key_nwidths3 = 3
    _key_split = "="
    _key_rep = ("isotropic", "")
    _key_range3 = range(3)
    _key_range6 = range(6)
    _key_range8 = range(8)
    _key_idx_pas = ["sigma11", "sigma22", "sigma33"]
    _key_idx_iso = "iso"
    _key_idx_ijk = ["x", "y", "z"]
    _key_dtype = np.float64

    @property
    def _key_index(self):
        """Build the dataseries index."""
        index = []
        for prefix in self._key_idx_prefix:
            for alpha in self._key_idx_ijk:
                for beta in self._key_idx_ijk:
                    index.append(prefix + alpha + beta)
            if prefix != self._key_idx_prefix[-1]:
                index.append(prefix + self._key_idx_iso)
        for value in self._key_idx_pas:
            index.append(value)
        return index

    def _finalize_parse(self, data=None):
        """Parse the cartesian and principal axis representations and set the data."""
        cart = self[self._key_cart].to_data("fwf", widths=self._key_1tensor,
                                            names=self._key_range3).values.ravel(self._key_forder)
        cartiso = str(self[self._key_cartiso]).split(self._key_split)[1].strip()
        if data is None:
            data = cart.tolist()
        else:
            data += cart.tolist()
        data.append(cartiso)

        pas = self[self._key_pas].to_data("fwf", widths=self._key_1tensor,
                                          names=self._key_range3).values.ravel()
        data += pas.tolist()
        pas3 = self[self._key_pas3].to_data("fwf", widths=self._key_pas_widths,
                                            names=self._key_range3).values.ravel()
        data += pas3.tolist()

        self.ds = pd.to_numeric(pd.Series(data, index=self._key_index), errors='coerce')


class NMRNucleusParaParser(six.with_metaclass(NMRNucleusTensorParserMeta, Parser,
                                              NMRNucleusTensorParserMixin)):
    """
    """
    name = "paramagnetic"
    description = "Parses the paramagnetic NMR shielding tensors."
    _key_b1u1 = slice(7, 10)
    _key_b1u1iso = 11
    _key_s1gauge = slice(15, 18)
    _key_s1gaugeiso = 19
    _key_cart = slice(26, 29)
    _key_cartiso = 30
    _key_pas = slice(42, 45)
    _key_pas3 = 38
    _key_idx_prefix = ["b1", "u1", "s1", "gauge", "cart", "pas"]

    def _parse(self):
        """
        Parse the paramagnetic shielding tensor.
        """
        b1u1 = self[self._key_b1u1].to_data("fwf", widths=self._key_2tensor,
                                            names=self._key_range8).values.ravel(self._key_forder)
        b1iso, u1iso = str(self[self._key_b1u1iso]).strip().split(self._key_split)[1:]
        data = b1u1[self._key_first].tolist()
        data.append(b1iso.replace(*self._key_rep).strip())
        data += b1u1[self._key_second].tolist()
        data.append(u1iso.strip())

        s1gauge = self[self._key_s1gauge].to_data("fwf", widths=self._key_2tensor,
                                                  names=self._key_range8).values.ravel(self._key_forder)
        s1iso, gaugeiso = str(self[self._key_s1gaugeiso]).strip().split(self._key_split)[1:]
        data += s1gauge[self._key_first].tolist()
        data.append(s1iso.replace(*self._key_rep).strip())
        data += s1gauge[self._key_second].tolist()
        data.append(gaugeiso.strip())

        self._finalize_parse(data)


class NMRNucleusDiaParser(six.with_metaclass(NMRNucleusTensorParserMeta, Parser,
                                             NMRNucleusTensorParserMixin)):
    """
    """
    name = "diamagnetic"
    description = "Parses the diamagnetic NMR shielding tensors."
    _key_cv = slice(7, 10)
    _key_cviso = 11
    _key_cart = slice(18, 21)
    _key_cartiso = 22
    _key_pas = slice(34, 37)
    _key_pas3 = 30
    _key_idx_prefix = ["core", "valence", "cart", "pas"]

    def _parse(self):
        """Parse the diagmentic shielding tensor data."""
        cv = self[self._key_cv].to_data("fwf", widths=self._key_2tensor,
                                        names=self._key_range8).values.ravel(self._key_forder)
        ciso, viso = str(self[self._key_cviso]).strip().split(self._key_split)[1:]
        data = cv[self._key_first].tolist()
        data.append(ciso.replace(*self._key_rep))
        data += cv[self._key_second].tolist()
        data.append(viso.strip())

        self._finalize_parse(data)


class NMRNucleusTotParser(six.with_metaclass(NMRNucleusTensorParserMeta, Parser,
                                             NMRNucleusTensorParserMixin)):
    """
    """
    name = "total"
    description = "Parses the total NMR shielding tensors."
    _key_cart = slice(10, 13)
    _key_cartiso = 14
    _key_pas = slice(26, 29)
    _key_pas3 = 22
    _key_idx_prefix = ["cart", "pas"]

    def _parse(self):
        """Parse total NMR shielding tensors."""
        self._finalize_parse()



NMR.add_section_parsers(NMRMetadataParser, NMRInfoParser, NMRNucleusParser)
NMRNucleusParser.add_section_parsers(NMRNucleusInfoParser, NMRNucleusParaParser,
                                     NMRNucleusDiaParser, NMRNucleusTotParser)
