# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
ADF Input and Output Mixin Classes
###################################
Utility class providing common functionality for ADF input and output
editors and parsers.
"""


class _OutputMixin(object):
    """Contains some common properties and methods for ADF output objects."""
    _key_convergence = "NOT CONVERGED"

    @property
    def converged(self):
        """
        Check for convergence.

        Returns:
            convered (bool): True if converged.
        """
        if self._key_convergence in self:
            return False
        return True


