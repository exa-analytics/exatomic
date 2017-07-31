# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Base DataFrame
######################
"""
from exa import DataFrame as _DataFrame


class DataFrame(_DataFrame):
    def __init__(self, *args, **kwargs):
        super(DataFrame, self).__init__(*args, **kwargs)
        self.index.name = self.__class__.__name__.lower()
