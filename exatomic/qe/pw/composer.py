# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
PW Input Composer
##########################

"""
from exa.typed import cta


class PWInputMeta(ComposerMeta):
    """
    """
    control = dict
    system = dict
    electrons = dict
    ions = dict
    cell = dict


class PWInput(six.with_metaclass(PWInputMeta, PWCPComposer)):
    """
    Input file composer for Quantum ESPRESSO's pw.x module.
    """
    control = cta("control", dict)
    system = cta("system", dict)
    electrons = cta("electrons", dict)
    ions = cta("ions", dict)
    cell = cta("cell", dict)
    _template = _pwtemplate
