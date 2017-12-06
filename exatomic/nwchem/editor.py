# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NWChem Editor
##################
"""
import pandas as pd
from exatomic import Editor

class Editor(Editor):

    def __init__(self, *args, **kwargs):
        super(Editor, self).__init__(*args, **kwargs)
        if self.meta is not None:
            self.meta.update({'program': 'nwchem'})
        else:
            self.meta = {'program': 'nwchem'}
