# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Base Molcas Editor
##################
"""
from exatomic import Editor as AtomicEditor


class Editor(AtomicEditor):
    _to_universe = AtomicEditor.to_universe

    def to_universe(self, *args, **kws):
        kwargs = {}
        for attr in ['momatrix', 'orbital', 'overlap']:
            kwargs[attr] = getattr(self, attr, None)
        kws.update(kwargs)
        return self._to_universe(*args, **kws)

    def __init__(self, *args, **kwargs):
        super(Editor, self).__init__(*args, **kwargs)
        if self.meta is None: self.meta = {'program': 'molcas',
                                           'gaussian': True}
        else: self.meta.update({'program': 'molcas',
                                'gaussian': True})
