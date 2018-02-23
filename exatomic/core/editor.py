# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Atomic Editor
###################
This module provides a text file editor that can be used to transform commonly
found file formats directly into :class:`~exatomic.container.Universe` objects.
"""
import six
from exa import Editor as _Editor
from exa import TypedMeta
from .universe import Universe
from .frame import compute_frame_from_atom


class Editor(six.with_metaclass(TypedMeta, _Editor)):
    """
    Base atomic editor class for converting between file formats and to (or
    from) :class:`~exatomic.container.Universe` objects.

    Note:
        Functions defined in the editor that generate typed attributes (see
        below) should be names "parse_{data object name}".

    See Also:
        For a list of typed attributes, see :class:`~exatomic.container.Universe`.
    """
    _getter_prefix = "parse"

    def parse_frame(self):
        """
        Create a minimal :class:`~exatomic.frame.Frame` from the (parsed)
        :class:`~exatomic.atom.Atom` object.
        """
        self.frame = compute_frame_from_atom(self.atom)


    def to_universe(self, name=None, description=None, meta=None, verbose=True,
                    kwargs=None):
        """
        Convert the editor to a :class:`~exatomic.container.Universe` object.
        """
        if hasattr(self, 'meta') and self.meta is not None:
            if meta is not None:
                meta.update(self.meta)
            else:
                meta = self.meta
        if kwargs is None:
            kwargs = {'name': name,
                      'description': description,
                      'meta': meta}
        else:
            kwargs.update({'name': name,
                           'description': description,
                           'meta': meta})
        attrs = [attr.replace('parse_', '')
                 for attr in vars(self.__class__).keys()
                 if attr.startswith('parse_')]
        for attr in attrs:
            result = None
            try:
                result = getattr(self, attr)
            except Exception as e:
                if verbose:
                    if not str(e).startswith('Please compute'):
                        print('parse_{} failed with: {}'.format(attr, e))
            if result is not None:
                kwargs[attr] = result
        return Universe(**kwargs)
