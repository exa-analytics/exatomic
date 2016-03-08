# -*- coding: utf-8 -*-
'''
Atomic Editor
====================================

'''
from exa.editor import Editor


class AtomicEditor(Editor):
    '''
    Editor specific to the atomic package.
    '''
    @property
    def atom(self):
        '''
        Create the atom dataframe from the editor.
        '''
        return self._parse_atom()

    @property
    def frame(self):
        '''
        Create the frame dataframe from the editor.
        '''
        return self._parse_frame()
