# -*- coding: utf-8 -*-
'''
Atomic Editor
====================================
'''
from exa.editor import Editor as ExaEditor
from atomic.universe import Universe
from atomic.frame import minimal_frame


class Editor(ExaEditor):
    '''
    Editor specific to the atomic package. Property definitions and follow the same naming
    convention as :class:`~atomic.universe.Universe`. Note that only "raw" dataframes (those
    whose data can only be obtained from the raw input) are acccessible; computed dataframes
    (such :class:`~atomic.atom.ProjectedAtom`) are available after creating a unvierse
    (:func:`~atomic.editor.Editor.to_universe`).
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

    @property
    def unit_atom(self):
        if self._unit_atom is None:
            self.parse_unit_atom()
        return self._unit_atom

    @property
    def visual_atom(self):
        if self._visual_atom is None:
            self.parse_visual_atom()
        return self._visual_atom

    @property
    def two(self):
        if self._two is None:
            self.parse_two()
        return self._two

    @property
    def periodic_two(self):
        if self._periodic_two is None:
            self.parse_two()
        return self._periodic_two

    @property
    def molecule(self):
        if self._molecule is None:
            self.parse_molecule()
        return self._molecule

    @property
    def basis(self):
        if self._basis is None:
            self.parse_basis()
        return self._basis

    @property
    def orbital(self):
        if self._orbital is None:
            self.parse_orbital()
        return self._orbital

    @property
    def orbital_coefficient(self):
        if self._orbital_coefficient is None:
            self.parse_orbital_coefficient()
        return self._orbital_coefficient

    @property
    def field(self):
        if self._field is None:
            self.parse_field()
        return self._field

    @property
    def field_values(self):
        if self._field is None:
            self.parse_field()
        if self._field is not None:
            return self.field.field_values

    # Placeholder functions; all parsing functions follow the same template as for frame:
    # self.frame is a property returns self._frame
    # self.parse_frame() sets self._frame, returns nothing
    def parse_frame(self):
        '''
        By default, generate a minimal frame from the atom dataframe.
        '''
        self._frame = minimal_frame(self.atom)

    def parse_two(self):
        return

    def parse_molecule(self):
        return

    def parse_field(self):
        return

    def parse_basis(self):
        return

    def parse_orbital(self):
        return

    def parse_unit_atom(self):
        return

    def parse_visual_atom(self):
        return

    def parse_orbital_coefficient(self):
        return

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
        return Universe(frame=self.frame, atom=self.atom, meta=self.meta, field=self.field,
                        orbital=self.orbital, basis=self.basis, molecule=self.molecule,
                        two=self.two, periodic_two=self.periodic_two, unit_atom=self.unit_atom,
                        orbital_coefficient=None, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._atom = None
        self._frame = None
        self._field = None
        self._orbital = None
        self._orbital_coefficient = None
        self._field = None
        self._two = None
        self._periodic_two = None
        self._unit_atom = None
        self._visual_atom = None
        self._molecule = None
        self._basis = None
