# -*- coding: utf-8 -*-
'''
Errors, Warnings, and Exceptions for atomic
============================================
'''
from exa.errors import ExaException


class PeriodicError(ExaException):
    '''
    Raised when a periodic operation is attempted on a non-periodic system.
    '''
    msg = 'Periodic system? Possible missing a "periodic" (=True) in the Frame table?'
