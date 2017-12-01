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
from exa import Parser
from .adf import Output as ADF



class Output(Parser):
    """
    Parser for a composite output ADF output file that may contain 'D I R A C',
    'A D F', 'N M R', etc. calculations.

    This object accepts the most general type of ADF output file that contains
    output structures from many different calculations.
    """
    pass


Output.add_parsers(ADF)
