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
    def atom_count(self):
        '''
        '''
        if self._atom is None:
            self.parse_atom()
        return len(self._atom)

    @property
    def frame(self):
        '''
        '''
        if self._frame is None:
            self.parse_frame()
        return self._frame

    @property
    def atom(self):
        '''
        '''
        if self._atom is None:
            self.parse_atom()
        return self._atom

    def to_universe(self):
        '''
        Create a :class:`~atomic.universe.Universe` from the editor object.
        '''
        atom = self.atom
        frame = self.frame
        meta = self.meta
        return Universe(frame=frame, atom=atom, meta=meta)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._atom = None
        self._frame = None
