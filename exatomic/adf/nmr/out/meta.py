# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NMR Output Subsection
######################
The metadata subsection is the first subsection of an 'N M R' calculation.
It contains information about the citations and license.
"""
import six
from exa.core import Meta, Parser


class NMRMetadataParserMeta(Meta):
    """Defines the objects parsed by NMRMetadataParser."""
    _descriptions = {}


class NMRMetadataParser(six.with_metaclass(NMRMetadataParserMeta, Parser)):
    """"""
    name = "metadata"
    description = "Parser for the title and citations section of an ADF NMR output."

    def _parse(self):
        pass


