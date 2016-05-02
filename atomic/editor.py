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
    # Hidden bound methods for simple API
    def _last_num_from_regex(self, regex):
        return int(list(regex.items())[0][1].split()[-1])

    def _lineno_from_regex(self, regex):
        return list(regex.items())[0][0]

    def _pandas_csv(self, flslice, ncol):
        return pd.read_csv(flslice, delim_whitespace=True, 
                           names=range(ncol)).stack().values   

    def _patterned_array(self, regex, dim, ncols):
        first = self._lineno_from_regex(regex) + 1
        last = first + int(np.ceil(dim / ncols))
        return self._pandas_csv(StringIO('\n'.join(self[first:last])), ncols)
                                        

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
