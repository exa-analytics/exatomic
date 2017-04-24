# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Composite ADF Output File Editor
###################################
"""
from exa.core import Sections


class CompositeADFOutput(Sections):
    """
    Generic ADF output file containing many types of executable calls.
    """
    name = "Composite ADF output"
    description = "General parser for ADF output files."
