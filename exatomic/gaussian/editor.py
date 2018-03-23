# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Gaussian Editor
#######################
Base class on top of exatomic.Editor for Gaussian Editors
"""
import pandas as pd
from exatomic import Editor

class Editor(Editor):

    def __init__(self, *args, **kwargs):

        super(Editor, self).__init__(*args, **kwargs)

        if self.meta is not None:
            self.meta.update({'program': 'gaussian'})
        else:
            self.meta = {'program': 'gaussian'}
