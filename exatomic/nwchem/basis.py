# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Parser for Output File Basis Set Block
########################################
Standalone parser for the basis set block of NWChem output files.
"""
import re
from exa import Parser, Typed, DataFrame
from exatomic.core.basis import GaussianBasisSet, str2l


class BasisSet(Parser):
    """
    Parser for the NWChem's printing of basis sets.
    """
    _start = re.compile("^\s*Basis \"")
    _ek = re.compile("^ Summary of \"")
    _k = "spherical"
    _k0 = "Exponent"
    _k1 = "("
    _i0 = 2
    _i1 = 4
    _cols = ("function", "l", "a", "d", "tag")
    basis_set = Typed(GaussianBasisSet, doc="Gaussian basis set description.")

    def _parse_end(self, starts):
        return [self.regex_next(self._ek, cursor=i[0]) for i in starts]

    def parse_basis_set(self):
        """
        """
        data = []
        tag = None
        spherical = True if self._k in self.lines[0] else False
        for line in self:
            if self._k0 not in line:
                ls = line.split()
                if len(ls) == self._i0 and self._k1 in line:
                    tag = ls[0]
                elif len(ls) == self._i1:
                    data.append(ls+[tag])
        basis_set = DataFrame(data, columns=self._cols)
        basis_set['l'] = basis_set['l'].str.lower().map(str2l)
        self.basis_set = GaussianBasisSet(basis_set, spherical=spherical, order=lambda x: x)

