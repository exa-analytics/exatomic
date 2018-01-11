# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Parser for Output File C Matrix
################################
Standalone parser for the molecular orbital coefficient matrix (if present)
in NWChem output calculations. To enable this (in NWChem), the input file
should contain the **print high** directive or the calculation block should
contain the **print "final vectors analysis"**.
"""
import numpy as np
import pandas as pd
from exa import Parser, Typed
from exatomic.core.orbital import Coefficient, _gen_mo_index


class MOVectors(Parser):
    """
    Parser for NWChem's molecular orbital coefficient matrix output.
    """
    _start = "Final MO vectors"
    _ek = ("center of mass",
           "---------------------------------------------------------")
    _i0 = 6
    _i1 = -1
    _cols = (0, 1, 2, 3, 4, 5, 6)
    _wids = (6, 12, 12, 12, 12, 12, 12)
    coefficient = Typed(Coefficient, doc="Molecular orbital coefficients.")

    def parse_coefficient(self):
        """
        Parse molecular orbital coefficient matrix.

        Read in the entire listing (7 columns varying number of lines).
        Identify the number of basis functions by the first column's unique
        values then unravel the coefficient values and index them appropriately.
        """
        # Read in the mangled table
        c = pd.read_fwf(self[self._i0:self._i1].to_stream(),
                        names=self._cols, widths=self._wids)
        # The following sets text to null so that we correctly count nbas
        c[0] = pd.to_numeric(c[0], errors='coerce')
        # Remove null lines
        idx = c[c[0].isnull()].index.values
        c = c[~c.index.isin(idx)]
        # The size of the basis is given by the informational numbers
        nbas = c[0].unique().shape[0]
        # Remove that column; it doesn't contain coefficients only sequential integers
        del c[0]
        n = c.shape[0]//nbas
        coefs = []
        # The loop below is like numpy.array_split (same speed, but
        # fully compatible with pandas)
        for i in range(n):
            coefs.append(c.iloc[i*nbas:(i+1)*nbas, :].astype(float).dropna(axis=1).values.ravel("F"))
        # Concatenate coefficients
        c = np.concatenate(coefs)
        orbital, chi = _gen_mo_index(nbas)
        df = pd.DataFrame.from_dict({'orbital': orbital, 'chi': chi, 'c': c})
        df['frame'] = 0
        self.coefficient = df

    def _parse_end(self, starts):
        return [self.find_next(*self._ek, cursor=i[0]) for i in starts]
