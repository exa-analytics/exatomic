# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Base Molcas Editor
##################
"""
from exatomic import Editor as AtomicEditor


class Editor(AtomicEditor):
    _to_universe = AtomicEditor.to_universe

    def to_universe(self, *args, **kwargs):
        uni = self._to_universe(self, *args, **kwargs)
        try:
            uni.occupation_vector = self.occupation_vector
        except AttributeError:
            pass
        return uni

    def __init__(self, *args, **kwargs):
        super(Editor, self).__init__(*args, **kwargs)
        if self.meta is None: self.meta = {'program': 'molcas'}
        else: self.meta.update({'program': 'molcas'})
