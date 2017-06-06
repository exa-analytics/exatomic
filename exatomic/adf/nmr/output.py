# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NMR Output Editor
######################
This module houses the main output editor for an 'N M R" calculation. The output
block structure is summarized below.

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
import numpy as np
import pandas as pd
from exa.typed import TypedProperty
from exa.core.parser import Sections
from exa.core.dataframe import DataFrame
from exatomic.atom import Atom
from exatomic.adf.mixin import OutputMixin
from .out import NMRMetadataParser, InfoParser, NMRNucleusParser


class NMR(Sections, OutputMixin):
    """
    The 'N M R' calculation section of a composite (or standalone) ADF output
    file.

    Attributes:
        shielding_tensors (DataFrame): Shielding tensor data (per atom)
    """
    _key_delim0 = "^#+$"
    _key_delim1 = "\(LOGFILE\)"
    _key_sec_names = [NMRMetadataParser, "None", InfoParser]    # Static section parsers
    _key_sec_titles = ["version", "None", "basis"]              # Static section titles
    _key_start = 0
    _key_nuc_title = 1
    _key_nuc_title_rep = ("*", "")
    _key_nuc_name = NMRNucleusParser
    _key_log_name = "LOGFILE"
    _key_log_title = "log"
    _key_id_type = np.int64
    atom = TypedProperty(Atom)
    shielding_tensors = TypedProperty(DataFrame, docs="Full shielding tensor data")

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

    # Getting the atom object is far easier
    def _get_atom(self):
        """Find the reference to the atom table."""
        pass
