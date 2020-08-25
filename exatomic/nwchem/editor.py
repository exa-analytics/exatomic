# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NWChem Editor
##################
"""
from exatomic import Editor as _Editor


class Editor(_Editor):
    def __init__(self, *args, **kwargs):
        super(Editor, self).__init__(*args, **kwargs)
        if self.meta is None: self.meta = {'program': 'nwchem',
                                           'gaussian': True}
        else: self.meta.update({'program': 'nwchem',
                                'gaussian': True})
