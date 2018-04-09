# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Base ADF editor
##################
"""
from exatomic import Editor as AtomicEditor


class Editor(AtomicEditor):
    def __init__(self, *args, **kwargs):
        super(Editor, self).__init__(*args, **kwargs)
        if self.meta is None:
            self.meta = {'program': 'adf',
                         'gaussian': False}
        else:
            self.meta.update({'program': 'adf',
                              'gaussian': False})
