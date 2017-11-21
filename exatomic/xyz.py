## -*- coding: utf-8 -*-
## Copyright (c) 2015-2017, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
"""
XYZ Parser
##################
A parser for the `XYZ`_ file format (for storing atomic coordinates).

.. _XYZ: https://en.wikipedia.org/wiki/XYZ_file_format
"""
import re
import numpy as np
import numba as nb
import pandas as pd
from operator import itemgetter
from exa import Parser, Typed
from exatomic.base import nbpll, nbche
from exatomic.core.atom import Atom


@nb.jit(nopython=True, nogil=True, parallel=nbpll, cache=nbche)
def _build_indexes(start, stop):
    """Given the sections of an XYZ file, build the indexes."""
    m = len(start)
    n = stop[-1] - 2*m     # Total number of atoms (in all frames)
    idx = np.empty((n, ), dtype=np.int64)
    fdx = idx.copy()
    cdx = np.empty((len(start), ), dtype=np.int64)
    k = 0    # Frame counter
    j = 0    # Atom counter
    for i, sta in enumerate(start):
        for v in range(sta+2, stop[i]):
            idx[j] = v
            fdx[j] = k
            j += 1
        k += 1
        cdx[i] = sta + 1
    return idx, fdx, cdx


class XYZ(Parser):
    """
    Parser for the XYZ file format.

    The symbol column gets mapped onto the proton number, 'Z'. For extended
    XYZ-like files with custom columns pass the column names as needed.

    .. code-block:: python

        xyz = XYZ(file)
        xyz.atom              # Default XYZ format
        # Support for additional formats
        xyz = XYZ(file)
        xyz.parse(("symbol", "charge", "x", "y", "z"))
        xyz.atom              # The atom table has an additional column 'charge'
    """
    atom = Typed(Atom, doc="Table of nuclear coordinates")
    comments = Typed(dict, doc="Dictionary of comments")

    def parse_atom(self, columns=("symbol", "x", "y", "z")):
        """Perform complete parsing."""
        idx, fdx, cdx = _build_indexes(self.sections['startline'].values,
                                       self.sections['endline'].values)
        atom = pd.DataFrame([l.split() for l in itemgetter(*idx)(self.lines)],
                             columns=columns)
        atom['frame'] = fdx
        self.comments = {i: self.lines[j] for i, j in enumerate(cdx)}
        self.atom = Atom.from_xyz(atom)

    def parse_comments(self):
        self.parse_atom()

    @classmethod
    def from_universe(cls, uni):
        """
        Create an XYZ editor-like object from a universe.

        Args:
            uni (:class:`~exatomic.core.universe.Universe`): The universe

        Returns:
            xyz (:class:`~exatomic.xyz.XYZ`): XYZ editor
        """
        text = ""
        uni.atom['symbol'] = uni.atom.get_symbols()
        for fdx, frame in uni.atom.groupby("frame"):
            xyz = frame[['symbol', 'x', 'y', 'z']].to_csv(sep=" ", header=False, index=False)
            text += "{}\n{}\n{}\n".format(len(frame), "Frame: "+str(fdx), xyz)
        ed = cls(text)
        del uni.atom['symbol']
        ed.atom = uni.atom
        return ed

    def _parse_both(self):
        """
        Parse the starting points and stopping points by counting.
        """
        n = len(self)
        cnat = 0
        nat = int(str(self[0]))
        cnat += nat
        starts = [(0, "")]
        ends = [(nat+1, "")]
        while cnat + 2*len(ends) < n:
            try:
                line_num = cnat + 2*len(ends)
                text = str(self[line_num])
                nat = int(text)
                starts.append((line_num, ""))
                ends.append((line_num+nat+1, ""))
                cnat += nat
            except IndexError:
                break
        return starts, ends
