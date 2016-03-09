# -*- coding: utf-8 -*-
'''
Atomic Editor
====================================

'''
from exa.editor import Editor
from atomic.universe import Universe


class AtomicEditor(Editor):
    '''
    Editor specific to the atomic package.
    '''
    def to_universe(self):
        '''
        Create a :class:`~atomic.universe.Universe` from the editor object.
        '''
        atom = self.atom
        frame = self.frame
        meta = self.meta
        return Universe(frame=frame, atom=atom, meta=meta)
