# -*- coding: utf-8 -*-
'''
Atomic Editor
====================================

'''
from exa import Editor


class AtomicEditor(Editor):
    '''
    Editor specific to the atomic package.
    '''
    @property
    def atom(self):
        '''
        Create the atom dataframe from the editor.
        '''
        raise NotImplementedError()

    @property
    def frame(self):
        '''
        Create the frame dataframe from the editor.
        '''
        raise NotImplementedError()
