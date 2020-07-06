# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Gaussian Editor
#######################
Base class on top of exatomic.Editor for Gaussian Editors
"""
#import pandas as pd
from exatomic import Editor

class Editor(Editor):

    def __init__(self, *args, **kwargs):
        super(Editor, self).__init__(*args, **kwargs)
        if self.meta is None: self.meta = {'program': 'gaussian',
                                           'gaussian': True}
        else: self.meta.update({'program': 'gaussian',
                                'gaussian': True})
