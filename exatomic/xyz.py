# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
XYZ Editor
##################
An editor for the `XYZ`_ file format.

.. _XYZ: https://en.wikipedia.org/wiki/XYZ_file_format
"""
import numpy as np
import pandas as pd
from exa import units
from exa.core.parser import Parser
from exa.typed import TypedProperty
from .atom import Atom, frame_index


class XYZ(Parser):
    """
    A parser/composer for the `XYZ`_ file format.

    This class can parser in simple XYZ and trajectory XYZ files. Occasionally
    additional columns will be present in xyz-like files and can be handled by
    keyword arguments. XYZ files can be constructed ('composed') using the
    classmethods, ``from_universe``, ``from_atom``, etc.

    .. code-block:: python

        from exa import units

        # File in atomic length units (Bohr)
        xyz = XYZ(xyzfile, unit=units.au_length)

        # File with extra columns
        xyz = XYZ(xyzfile, columns=("symbol", "Z", "x", "y", "z"))

    .. _XYZ: https://en.wikipedia.org/wiki/XYZ_file_format
    """
    _parse_unit = units.Angstrom
    _parse_columns = ("symbol", "x", "y", "z")
    comment_lines = TypedProperty(list, "List of lines with comments")
    atom = TypedProperty(Atom, "Atom dataframe of absolute nuclear coordinates")

    @property
    def comments(self):
        """Get a dictionary of frame numbers and comments."""
        return {i: str(self[k]) for i, k in enumerate(self.comment_lines)}

    @classmethod
    def from_universe(cls, universe):
        """Create an XYZ editor from a universe."""
        raise NotImplementedError()

    @classmethod
    def from_atom(cls, atom):
        """Create an XYZ editor from an :class:`~exatomic.atom.Atom` table."""
        raise NotImplementedError()

    def to_atom(self):
        """Create an :class:`~exatomic.atom.Atom` table."""
        raise NotImplementedError()

    def write(self, path, trajectory=True, float_format="%    .10f"):
        """
        Write the file (or files) to disk.

        Args:
            path (str): File or directory path
            trajectory (bool): If true, write trajectory-like XYZ file
            float_format (str): Formatting for numbers
        """
        raise NotImplementedError()

    def _parse(self):
        """
        The parser assumes that the number of atoms does not vary if multiple frames,
        but if they do a slow parsing algorithm is used that goes xyz-block by xyz-block.
        """
        nat = int(str(self[0]).split()[0])
        nat2 = nat + 2
        # Check if we have an integer number of frames
        if np.mod(len(self), nat2) == 0:
            # To confirm that all frames have the same number of atoms,
            # we compare the lines.
            if not all(int(line.split()[0]) == nat for line in self[::nat2]):
                self._parse_variable()           # Logic for the current function
                return                           # ends here, so we return.
            nframe = len(self)//nat2
        else:
            self._parse_variable()    # Similarly, if it is not obvious how many
            return                    # frames/atoms there are use the slow approach.
        rows = frame_index(nframe, nat)
        atom = self[rows].to_data(delim_whitespace=True, names=self._parse_columns)
        atom['frame'] = [i for i in range(nframe) for _ in range(nat)]
        self.atom = atom
        if self._parse_unit != units.Angstrom:
            factor = units.Angstrom._value/self._parse_unit._value
            for r in ("x", "y", "z"):
                self.atom[r] *= factor
        self.comment_lines = [i*nat2 + 1 for i in range(nframe)]

    def _parse_variable(self):
        """
        Parses a trajectory like XYZ file that has variable atom counts in each frame.
        No assumptions are made about the length of the file, or number of atoms per
        frame.
        """
        cursor = 0    # Cursor line number
        frame = 0     # Frame index
        n = len(self)
        nat = 0
        atoms = []
        comments = []
        # Inspect line by line
        while cursor + nat < n:
            ls = str(self[cursor]).strip().split()
            # If we find what we think is a nat, then parse
            if len(ls) >= 1 and ls[0].isdigit():
                nat = int(ls[0])
                comments.append(cursor + 1)
                atom = self[cursor+2:cursor+nat+2].to_data(delim_whitespace=True,
                                                           names=self._parse_columns)
                atom['frame'] = frame
                atoms.append(atom)
                cursor += nat + 2
                nat = 0
                frame += 1
            else:
                cursor += 1
        self.atom = pd.concat(atoms, ignore_index=True)
        if self._parse_unit != units.Angstrom:
            factor = units.Angstrom._value/self._parse_unit._value
            for r in ("x", "y", "z"):
                self.atom[r] *= factor
        self.comment_lines = comments

    def __init__(self, *args, **kwargs):
        unit = kwargs.pop("unit", XYZ._parse_unit)
        columns = kwargs.pop("columns", XYZ._parse_columns)
        super(XYZ, self).__init__(*args, **kwargs)
        self._parse_unit = unit
        self._parse_columns = columns
