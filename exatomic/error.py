# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Exceptions
###############
'''
from exa.error import ExaException


class AtomicException(ExaException):
    '''
    The exatomic exception.
    '''
    pass


class StringFormulaError(AtomicException):
    '''
    The string representation of a :class:`~exatomic.formula.SimpleFormula` has
    syntax described in :class:`~exatomic.formula.SimpleFormula`.
    '''
    _msg = 'Incorrect formula syntax for {} (syntax example H(2)O(1)).'

    def __init__(self, formula):
        msg = self._msg.format(formula)
        super().__init__(msg)


class ClassificationError(AtomicException):
    '''
    Raised when a classifier for :func:`~exatomic.molecule.Molecule.add_classification`
    used incorrectly.
    '''
    _msg = 'Classifier must be a tuple of the form ("identifier", "label", exact).'
