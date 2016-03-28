# -*- coding: utf-8 -*-
'''
Atomic Editor
====================================

'''
from exa.editor import Editor as _Editor
from atomic.universe import Universe


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

    def to_universe(self, **kwargs):
        '''
        Create a :class:`~atomic.universe.Universe` from the editor object.
        '''
        return Universe(frame=self.frame, atom=self.atom, meta=self.meta, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._atom = None
        self._frame = None
