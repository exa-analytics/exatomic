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
import re
import six
import pandas as pd
from exa.core import Meta, Sections#, Parser


class DIRACMeta(Meta):
    """Defines data objects of :class:`~exatoimc.adf.dirac.parser.DIRAC`."""
    orbital = pd.DataFrame


class DIRAC(six.with_metaclass(DIRACMeta, Sections)):
    """
    A parser for the 'D I R A C' section(s) of an ADF calculation.

    The 'dirac' program solves the all-electron radial problem for a variety
    of Hamiltonians (default is the Dirac-Slater Hamiltonian with Slater's
    alpha value modified to 0.7).
    """
    _key_d0 = re.compile("^[01] ")        # Default printed sections
    _key_d1 = re.compile("^\n, [ -]0")    # Debug printed wave forms

    def _parse(self):
        """
        """
        pass
        #secs = self.regex(self._key_d0, self._key_d1)
