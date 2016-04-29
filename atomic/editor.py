# -*- coding: utf-8 -*-
'''
Atomic Editor
====================================
'''
from exa.editor import Editor as _Editor
from atomic.universe import Universe
from atomic.frame import minimal_frame


class Editor(_Editor):
    '''
    Editor specific to the atomic package.
    '''
    @property
    def frame(self):
        '''
        The :class:`~atomic.frame.Frame` dataframe parsed from the current editor.
        '''
        if self._frame is None:
            self.parse_frame()
        return self._frame

    @property
    def atom(self):
        '''
        The :class:~atomic.atom.Atom` dataframe parsed from the current editor.
        '''
        if self._atom is None:
            self.parse_atom()
        return self._atom

    def parse_frame(self):
        '''
        By default, generate a minimal frame from the atom dataframe.
        '''
        self._frame = minimal_frame(self.atom)

    def to_universe(self, **kwargs):
        '''
        Create a :class:`~atomic.universe.Universe` from the editor object.

        Warning:
            This operation does not make a copy of any dataframes (e.g. atom).
            Any changes made to the resulting universe object will be reflected
            in the original editor. To obtain the "original" dataframes, simply
            run the parsing functions "parse_*" again. This will only affect
            the editor object; it will not affect the universe object.
        '''
        return Universe(frame=self.frame, atom=self.atom, meta=self.meta, **kwargs)

    def __init__(self, *args, atom=None, frame=None, two=None, field=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._atom = atom
        self._frame = frame
        self._two = two
        self._field = field
