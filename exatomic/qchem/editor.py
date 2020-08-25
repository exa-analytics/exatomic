# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Q-Chem Editor
#######################
Base class on top of exatomic.Editor for Q-Chem Editors
"""
#import pandas as pd
from exatomic import Editor

class Editor(Editor):

    def __init__(self, *args, **kwargs):
        super(Editor, self).__init__(*args, **kwargs)
        if self.meta is None: self.meta = {'program': 'qchem',
                                           'qchem': True}
        else: self.meta.update({'program': 'qchem',
                                'qchem': True})
