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
from exatomic.base import sym2z, nbpll, nbche
from exatomic.core.atom import Atom


@nb.jit(nopython=True, nogil=True, parallel=nbpll, cache=nbche)
def _build_indexes(start, stop):
    """Given the sections of an XYZ file, build the indexes."""
    m = len(start)
    n = stop[-1] - 2*m
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
    _start = re.compile("^\s*(\d+)")
    atom = Typed(Atom, doc="Table of nuclear coordinates")
    comments = Typed(dict, doc="Dictionary of comments")

    def _parse_end(self, starts):
        """Stop when the number of atoms plus two is found."""
        matches = []
        for k, v in starts:
            n = k + int(v.split()[0]) + 1
            text = self.lines[n]
            matches.append((n, text))
        return matches

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
