# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
ADF Composite Output
#########################
This module provides the primary (user facing) output parser.
"""
from exa import Parser
from .dirac import DIRAC
from .adf import ADF


class Output(Parser):
    """
    The ADF output parser.
    """
    pass


Output.add_parsers(DIRAC, ADF)
