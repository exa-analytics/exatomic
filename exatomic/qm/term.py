# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
r"""
Term Symbol
#######################################
Term symbols define specific stationary states which are eigenfunctions
of the total angular momentum quantum number.
"""
from exa import DataFrame, Column, Index


class State(DataFrame):
    pass


class AtomicState(State):
    """
    """
    n = Column(int)
    l = Column(int)

    def symbolic_l(self):
        return self['l'].map(l_lower_map)
