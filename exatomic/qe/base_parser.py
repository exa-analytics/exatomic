# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Base Output Parser
#############################
A few of the output files from various executables provided as part of the
`QE`_ suite share an output format. This module provides an (abstract) base
parser that contains shared functionality. Users should use the executable
specific output parsers, for example :class:`~exatomic.qe.pw.output.PWOutput`.
"""
from exa.core.parser import Sections, Parser


class PWCPSections(Sections):
    pass

