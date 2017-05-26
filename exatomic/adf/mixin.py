# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
ADF Input and Output Mixin Classes
###################################
Utility class providing common functionality for ADF input and output
editors and parsers.
"""


class OutputMixin(object):
    """Common properties and methods relevant to all output objects."""
    _key_convergence = "NOT CONVERGED"

    @property
    def converged(self):
        """
        Any easy way to check for (SCF) convergence.

        Returns:
            convered (bool): True if converged.
        """
        if self._key_convergence in self:
            return False
        return True
