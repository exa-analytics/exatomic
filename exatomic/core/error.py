# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Exceptions
###############
"""
from exa.core.error import ExaException


class AtomicException(ExaException):
    """
    The exatomic exception.
    """
    pass


class StringFormulaError(AtomicException):
    """
    The string representation of a :class:`~exatomic.formula.SimpleFormula` has
    syntax described in :class:`~exatomic.formula.SimpleFormula`.
    """
    _msg = 'Incorrect formula syntax for {} (syntax example H(2)O(1)).'

    def __init__(self, formula):
        msg = self._msg.format(formula)
        super().__init__(msg)


class ClassificationError(AtomicException):
    """
    Raised when a classifier for :func:`~exatomic.molecule.Molecule.add_classification`
    used incorrectly.
    """
    def __init__(self):
        super().__init__(msg='Classifier must be a tuple of the form ("identifier", "label", exact).')


class PeriodicUniverseError(AtomicException):
    """
    Raised when the code is asked to perform a periodic simulation specific
    operation on a free boundary condition :class:`~exatomic.container.Universe`.
    """
    def __init__(self):
        super().__init__(msg='Not a periodic boundary condition Universe?')


class FreeBoundaryUniverseError(AtomicException):
    def __init__(self):
        super().__init__(msg='Not a free boundary condition Universe?')


class BasisSetNotFoundError(AtomicException):
    def __init__(self):
        super().__init__(msg='Not basis set table present in Universe?')
