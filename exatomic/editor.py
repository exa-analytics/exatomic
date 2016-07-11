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


class Editor(BaseEditor, metaclass=UniverseTypedMeta):
    '''
    Base atomic editor class for converting between file formats and to (or
    from) :class:`~exatomic.container.Universe` objects.
    '''
    def to_universe(self, *args, **kwargs):
        '''
        Convert the editor to a :class:`~exatomic.container.Universe` object.
        '''
        return Universe(*args, frame=self.frame, atom=self.atom, **kwargs)


#import numpy as np
#import pandas as pd
#from io import StringIO
#from exa.editor import Editor as ExaEditor
#from exatomic.container import Universe, UniverseTypedMeta
#from exatomic.frame import compute_frame_from_atom
#from exatomic.basis import CartesianGTFOrder, SphericalGTFOrder, lmap
#
#
#class Editor(ExaEditor):
#    '''
#    Editor specific to the exatomic package. Property definitions and follow the same naming
#    convention as :class:`~exatomic.universe.Universe`. Note that only "raw" dataframes (those
#    whose data can only be obtained from the raw input) are acccessible; computed dataframes
#    (such :class:`~exatomic.atom.ProjectedAtom`) are available after creating a unvierse
#    (:func:`~exatomic.editor.Editor.to_universe`).
#    '''
#    @property
#    def frame(self):
#        '''
#        The :class:`~exatomic.frame.Frame` dataframe parsed from the current editor.
#        '''
#        if self._frame is None:
#            self.parse_frame()
#        return self._frame
#
#    @property
#    def atom(self):
#        '''
#        The :class:~exatomic.atom.Atom` dataframe parsed from the current editor.
#        '''
#        if self._atom is None:
#            self.parse_atom()
#        return self._atom
#
#    @property
#    def unit_atom(self):
#        if self._unit_atom is None:
#            self.parse_unit_atom()
#        return self._unit_atom
#
#    @property
#    def visual_atom(self):
#        if self._visual_atom is None:
#            self.parse_visual_atom()
#        return self._visual_atom
#
#    @property
#    def two(self):
#        if self._two is None:
#            self.parse_two()
#        return self._two
#
#    @property
#    def periodic_two(self):
#        if self._periodic_two is None:
#            self.parse_two()
#        return self._periodic_two
#
#    @property
#    def molecule(self):
#        if self._molecule is None:
#            self.parse_molecule()
#        return self._molecule
#
#    @property
#    def basis_set(self):
#        if self._basis_set is None:
#            self.parse_basis_set()
#        return self._basis_set
#
#    @property
#    def orbital(self):
#        if self._orbital is None:
#            self.parse_orbital()
#        return self._orbital
#
#    @property
#    def momatrix(self):
#        if self._momatrix is None:
#            self.parse_momatrix()
#        return self._momatrix
#
#    @property
#    def field(self):
#        if self._field is None:
#            self.parse_field()
#        return self._field
#
#    @property
#    def field_values(self):
#        if self._field is None:
#            self.parse_field()
#        if self._field is not None:
#            return self.field.field_values
#
#    @property
#    def basis_set_summary(self):
#        if self._basis_set_summary is None:
#            self.parse_basis_set_summary()
#        return self._basis_set_summary
#
#    # Placeholder functions; all parsing functions follow the same template as for frame:
#    # self.frame is a property returns self._frame
#    # self.parse_frame() sets self._frame, returns nothing
#    def parse_frame(self):
#        '''
#        By default, generate a minimal frame from the atom dataframe.
#        '''
#        self._frame = minimal_frame(self.atom)
#
#    def parse_two(self):
#        return
#
#    def parse_molecule(self):
#        return
#
#    def parse_field(self):
#        return
#
#    def parse_basis_set(self):
#        return
#
#    def parse_orbital(self):
#        return
#
#    def parse_unit_atom(self):
#        return
#
#    def parse_visual_atom(self):
#        return
#
#    def parse_momatrix(self):
#        return
#
#    def parse_basis_set_summary(self):
#        return
#
#    def to_universe(self, **kwargs):
#        '''
#        Create a :class:`~exatomic.universe.Universe` from the editor object.
#
#        Warning:
#            This operation does not make a copy of any dataframes (e.g. atom).
#            Any changes made to the resulting universe object will be reflected
#            in the original editor. To obtain the "original" dataframes, simply
#            run the parsing functions "parse_*" again. This will only affect
#            the editor object; it will not affect the universe object.
#        '''
#        return Universe(atom=self.atom, frame=self.frame, field=self.field)
#        spherical_gtf_order = None
#        cartesian_gtf_order = None
#        if self._sgtfo_func is not None:
#            lmax = self.basis_set['shell'].map(lmap).max()
#            spherical_gtf_order = SphericalGTFOrder.from_lmax_order(lmax, self._sgtfo_func)
#        if self._cgtfo_func is not None:
#            lmax = self.basis_set['shell'].map(lmap).max()
#            cartesian_gtf_order = CartesianGTFOrder.from_lmax_order(lmax, self._cgtfo_func)
#        return Universe(frame=self._frame, atom=self._atom, meta=self.meta, field=self.field,
#                        orbital=self.orbital, basis_set=self.basis_set, molecule=self.molecule,
#                        two=self.two, periodic_two=self.periodic_two, unit_atom=self.unit_atom,
#                        momatrix=self.momatrix, #spherical_gtf_order=spherical_gtf_order,
#                        #cartesian_gtf_order=cartesian_gtf_order,
#                        basis_set_summary=self._basis_set_summary, **kwargs)
#
#    def _last_num_from_regex(self, regex, typ=int):
#        return typ(list(regex.items())[0][1].split()[-1])
#
#    def _last_nums_from_regex(self, regex, typ=float):
#        return [typ(i[1].split()[-1]) for i in list(regex.items())]
#
#    def _lineno_from_regex(self, regex):
#        return list(regex.items())[0][0]
#
#    def _linenos_from_regex(self, regex):
#        return list(regex.keys())
#        #return [i[0] for i in list(regex.items())]
#
#    def _pandas_csv(self, flslice, ncol):
#        return pd.read_csv(flslice, delim_whitespace=True,
#                           names=range(ncol)).stack().values
#
#    def _patterned_array(self, regex, dim, ncols):
#        first = self._lineno_from_regex(regex) + 1
#        last = first + int(np.ceil(dim / ncols))
#        return self._pandas_csv(StringIO('\n'.join(self[first:last])), ncols)
#
#    def __init__(self, *args, sgtfo_func=None, cgtfo_func=None, **kwargs):
#        super().__init__(*args, **kwargs)
#        self.meta = {'program': None}
#        self._atom = None
#        self._frame = None
#        self._field = None
#        self._orbital = None
#        self._momatrix = None
#        self._field = None
#        self._two = None
#        self._periodic_two = None
#        self._unit_atom = None
#        self._visual_atom = None
#        self._molecule = None
#        self._basis_set = None
#        self._basis_set_summary = None
#        self._sgtfo_func = sgtfo_func
#        self._cgtfo_func = cgtfo_func
#
