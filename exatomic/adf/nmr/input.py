# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NMR Input Editors
######################
"""
import six
import numpy as np
import pandas as pd
from exa.core import Editor


class NMRInput(Editor):
    @classmethod
    def from_universe(cls, universe):
        raise NotImplementedError()
