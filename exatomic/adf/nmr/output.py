# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NMR Output Editors
######################
This module provides parsing for an 'N M R' calculation block output from ADF's
``nmr`` executable. The :class:`~exatomic.adf.nmr.output.NMROutput` object
represents this output. The output block is divided into subsections that
contain specific information. An overview is given below.

.. code-block:: text

    metadata (version, citations, etc.)
    ####################################
    blank
    ####################################
    info (xyz, basis, etc.)
    ####################################
    nucleus 1
    ####################################
    nucleus 2
    ####################################
    nucleus ...
    logfile

Three unique parsing objects are required, one for  ``metadata``, ``info``, and
``nucleus`` subsections. Regardless of how many nuclei are calculated a single
parsing object (of Sections subtype) handles that region of the output.
"""
import six
import numpy as np
import pandas as pd
from exa import Sections, Parser
from exa.core import Meta
from exatomic.adf.mixin import OutputMixin


class NMROutputMeta(Meta):
    """
    Defines data objects parsed in this part of an ADF output file. Note that
    some objects are aggregated from individual subsection parsers.
    """
    shielding_tensors = pd.DataFrame
    _description = {'shielding_tensors': "Shielding tensor data per atom"}


class NMROutput(six.with_metaclass(NMROutputMeta, Sections, OutputMixin)):
    """
    The 'N M R' calculation section of a composite (or standalone) ADF output
    file.

    Attributes:
        sheilding_tensors (DataFrame): Shielding tensor data (per atom)
    """
    name = "NMR"
    description = "Parser for an 'N M R' calculation block"
    _key_delim0 = "^#+$"
    _key_delim1 = "\(LOGFILE\)"
    _key_sec_names = ["metadata", "None", "info"]    # Parser names
    _key_sec_titles = ["version", "None", "basis"]   # Section titles
    _key_start = 0
    _key_nuc_title = 1
    _key_nuc_title_rep = ("*", "")
    _key_nuc_name = "NUCLEUS"
    _key_log_name = "LOGFILE"
    _key_log_title = "log"
    _key_id_type = np.int64

    def _parse(self):
        """Determine all sections."""
        found = self.regex(self._key_delim0, self._key_delim1, text=False)
        # Start lines and end lines
        starts = [self._key_start] + found[self._key_delim0]
        m = len(self._key_sec_titles)
        n = len(starts)
        starts += found[self._key_delim1]
        ends = starts[1:]
        ends.append(len(self))
        # Parser names
        parsers = list(self._key_sec_names) + [self._key_nuc_name]*(n-m) + [self._key_log_name]
        # Title names
        titles = list(self._key_sec_titles)
        titles += [str(self[i+self._key_nuc_title]).replace(*self._key_nuc_title_rep) for i in starts[m:n]]
        titles.append(self._key_log_title)
        dct = {'parser': parsers, 'start': starts, 'end': ends, 'title': titles}
        self._sections_helper(dct)

    # Note that we use the hidden "automatic" getter prefix "_get...".
    # If the ``shielding_tensors`` attribute has not been created, it will be
    # automatically done when this function is automatically called.
    # See documentation of exa for more information.
    def _get_shielding_tensors(self):
        """Build the complete shielding tensor dataframe."""
        nuclei = self.sections[self.sections["parser"] == self._key_nuc_name]
        df = []
        for i in nuclei.index:
            nucleus = self.get_section(i)
            adfid = nucleus.section0.adfid
            nmrid = nucleus.section0.nmrid
            symbol = nucleus.section0.symbol
            shieldings = []
            for j in nucleus.sections.index[1:]:
                subsec = nucleus.get_section(j)
                if subsec is not None:
                    shieldings.append(subsec.ds)
            shieldings = pd.concat(shieldings)
            shieldings["adfid"] = adfid
            shieldings["nmrid"] = nmrid
            shieldings["symbol"] = symbol
            df.append(shieldings)
        shielding_tensors = pd.DataFrame(df)
        shielding_tensors['adfid'] = shielding_tensors['adfid'].astype(self._key_id_type)
        shielding_tensors['nmrid'] = shielding_tensors['nmrid'].astype(self._key_id_type)
        self.shielding_tensors = shielding_tensors


class NMRMetadataParserMeta(Meta):
    """Defines the objects parsed by NMRMetadataParser."""
    _descriptions = {}


class NMRMetadataParser(six.with_metaclass(NMRMetadataParserMeta, Parser)):
    """"""
    name = "metadata"
    description = "Parser for the title and citations section of an ADF NMR output."

    def _parse(self):
        pass


class NMRInfoParserMeta(Meta):
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

    Collates individual arrays corresponding to different shielding tensor
    data (e.g. paramagnetic, diamagnetic, total, etc.) into a single table.
    """
    name = "NUCLEUS"
    description = "Parses NMR shielding tensors from the nucleus section."
    _key_sep = "^=+$"
    _key_start = 0
    _key_first_sec_name = "info"
    _key_sec_name_p = 2
    _key_sec_name_id = 0
    _key_title_rep0 = ("=== ", "")
    _key_title_rep1 = ("UNSCALED: ", "")
    _key_parser_name_rep = ("-", "")

    def _parse(self):
        """Parser out what shielding tensor components are present."""
        # First parse sub-sections
        starts = [self._key_start] + self.regex(self._key_sep, text=False)[self._key_sep]
        ends = starts[1:]
        parsers = [self._key_first_sec_name]
        titles = [self._key_first_sec_name]
        for start in ends:
            title = str(self[start + self._key_sec_name_p])
            title = title.replace(*self._key_title_rep0)
            title = title.replace(*self._key_title_rep1)
            titles.append(title)
            parser = title.split()[self._key_sec_name_id]
            parser = parser.replace(*self._key_parser_name_rep)
            parsers.append(parser)
        ends.append(len(self))
        dct = {'start': starts, 'end': ends, 'parser': parsers, 'title': titles}
        self._sections_helper(dct)


class NMRNucleusInfoParserMeta(Meta):
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
        for text in self.find(self._key_marker, num=False)[self._key_marker][1:]:
            txt, atom = text.split(self._key_marker)
            symbol, number = atom.replace(*self._key_repmrk).split(self._key_symmrk)
            if "ADF" in txt:
                self.adfid = number
            else:
                self.nmrid = number
        self.symbol = symbol.strip()


class NMRNucleusTensorParserMeta(Meta):
    """Defines objects parsed by NMRNucleusParaParser"""
    ds = pd.Series
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
                    index.append(self._key_typ + prefix + alpha + beta)
            if prefix != self._key_idx_prefix[-1]:
                index.append(self._key_typ + prefix + self._key_idx_iso)
        for value in self._key_idx_pas:
            index.append(self._key_typ + value)
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
    name = "PARAMAGNETIC"
    description = "Parses the paramagnetic NMR shielding tensors."
    _key_b1u1 = slice(6, 9)
    _key_b1u1iso = 10
    _key_s1gauge = slice(14, 17)
    _key_s1gaugeiso = 18
    _key_cart = slice(25, 28)
    _key_cartiso = 29
    _key_pas = slice(41, 44)
    _key_pas3 = 37
    _key_idx_prefix = ["b1", "u1", "s1", "gauge", "cart", "pas"]
    _key_typ = "para"

    def _parse(self):
        """Parse the paramagnetic shielding tensor data."""
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
    name = "DIAMAGNETIC"
    description = "Parses the diamagnetic NMR shielding tensors."
    _key_cv = slice(6, 9)
    _key_cviso = 10
    _key_cart = slice(17, 20)
    _key_cartiso = 21
    _key_pas = slice(33, 36)
    _key_pas3 = 29
    _key_idx_prefix = ["core", "valence", "cart", "pas"]
    _key_typ = "dia"

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
    name = "TOTAL"
    description = "Parses the total NMR shielding tensors."
    _key_cart = slice(9, 12)
    _key_cartiso = 13
    _key_pas = slice(25, 28)
    _key_pas3 = 21
    _key_idx_prefix = ["cart", "pas"]
    _key_typ = "tot"

    def _parse(self):
        """Parse total NMR shielding tensors."""
        self._finalize_parse()


NMRNucleusParser.add_section_parsers(NMRNucleusInfoParser, NMRNucleusParaParser,
                                     NMRNucleusDiaParser, NMRNucleusTotParser)
NMROutput.add_section_parsers(NMRMetadataParser, NMRInfoParser, NMRNucleusParser)
