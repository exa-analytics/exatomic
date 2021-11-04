# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Exceptions
#################################
"""
import re


class ExaException(Exception):
    """
    Exception with support for logging.
    """
    def __init__(self, msg):
        spacer = '\n' + ' ' * len(self.__class__.__name__) + '  '    # Align the message
        msg = re.sub(r'\s*\n\s*', spacer, msg)
        super().__init__(msg)


class RequiredColumnError(ExaException):
    """
    :class:`~exa.core.numerical.DataFrame` column error.
    """
    _msg = 'Missing required column(s), {0}, for creation of class {1} object.'

    def __init__(self, missing, clsname):
        msg = self._msg.format(missing, clsname)
        super().__init__(msg)

