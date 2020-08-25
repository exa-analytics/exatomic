# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Atomic Editor
###################
This module provides a text file editor that can be used to transform commonly
found file formats directly into :class:`~exatomic.container.Universe` objects.
"""
import six
import pandas as pd
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
        For a list of typed attributes, see :class:`~exatomic.core.universe.Universe`.
    """
    _getter_prefix = "parse"

    def parse_frame(self):
        """
        Create a minimal :class:`~exatomic.frame.Frame` from the (parsed)
        :class:`~exatomic.core.atom.Atom` object.
        """
        self.frame = compute_frame_from_atom(self.atom)

    def to_universe(self, **kws):
        """
        Convert the editor to a :class:`~exatomic.core.universe.Universe` object.

        Args:
            name (str): Name
            description (str): Description of parsed file
            meta (dict): Optional dictionary of metadata
            verbose (bool): Verbose information on failed parse methods
            ignore (bool): Ignore failed parse methods
        """
        name = kws.pop("name", None)
        description = kws.pop("description", None)
        meta = kws.pop("meta", None)
        verbose = kws.pop("verbose", True)
        ignore = kws.pop("ignore", False)
        if hasattr(self, 'meta') and self.meta is not None:
            if meta is not None:
                meta.update(self.meta)
            else:
                meta = self.meta
        kwargs = {'name': name, 'meta': meta,
                  'description': description}
        attrs = [attr.replace('parse_', '')
                 for attr in vars(self.__class__).keys()
                 if attr.startswith('parse_')]
        extras = {key: val for key, val in vars(self).items()
                  if isinstance(val, pd.DataFrame)
                  and key[1:] not in attrs}
        for attr in attrs:
            result = None
            try:
                result = getattr(self, attr)
            except Exception as e:
                if not ignore:
                    if not str(e).startswith('Please compute'):
                        print('parse_{} failed with: {}'.format(attr, e))
            if result is not None:
                kwargs[attr] = result
        kwargs.update(kws)
        kwargs.update(extras)
        return Universe(**kwargs)
