# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Atomic Editor
###################
This module provides a text file editor that can be used to transform commonly
found file formats directly into :class:`~exatomic.container.Universe` objects.
'''
from exa.editor import Editor as BaseEditor
from exatomic.container import UniverseTypedMeta, Universe
from exatomic.frame import compute_frame_from_atom


class Editor(BaseEditor, metaclass=UniverseTypedMeta):
    '''
    Base atomic editor class for converting between file formats and to (or
    from) :class:`~exatomic.container.Universe` objects.
    '''
    def parse_frame(self):
        '''
        Create a minimal_frame table.
        '''
        self.frame = compute_frame_from_atom(self.atom)

    def to_universe(self, *args, **kwargs):
        '''
        Convert the editor to a :class:`~exatomic.container.Universe` object.
        '''
        to_parse = [func.replace('parse_', '') for func in vars(self.__class__).keys() if func[:5] == 'parse']
        kwargs.update({attr: getattr(self, attr) for attr in to_parse if hasattr(self, attr)})
        kwargs.update({'frame': self.frame})
        return Universe(*args, **kwargs)


