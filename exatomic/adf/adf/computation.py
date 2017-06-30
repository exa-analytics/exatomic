# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Computation Section
###################################
Parsers related to the 'C O M P U T A T I O N' subsection of an ADF calculation.
"""
import pandas as pd
from exa import Sections, TypedProperty, DataFrame, Parser


class ComputationMultipoleParser(Parser):
    """
    """
    multipole_charges = TypedProperty(DataFrame,
                                      docs="Multipole derived atomic charges (au)")
    energy_gradients = TypedProperty(DataFrame,
                                     docs="Energy gradients wrt nuclear displacements (au/angstrom)")
    convergence_info = TypedProperty(pd.Series, docs="Geometry convergence information")
    coordinates = TypedProperty(DataFrame, docs="Atomic coordinates")
    _key_d_multi = ("Multipole derived", 5, ("label", "symbol", "MDC-m", "MDC-d", "MDC-q"))
    _key_d_nrggrad = ("Energy gradients", 6, ("label", "symbol", "x", "y", "z"))
    _key_d_conv = ("Geometry Convergence after", 2, 8, 30)
    _key_d_coord = ("Coordinates", 6, ("label", "symbol", "x", "y", "z",
                                       "x ang", "y ang", "z ang", "gvar0",
                                       "gvar1", "gvar2"))

    def _parse(self):
        """
        """
        found = self.find(self._key_d_multi[0], self._key_d_nrggrad[0],
                          self._key_d_conv[0], self._key_d_coord[0])
        self.found = found
        # Multipole charges
        start = found[self._key_d_multi[0]][0][0] + self._key_d_multi[1]
        end = start
        for line in self[start:]:
            if not line.strip()[0].isdigit():
                break
            end += 1
        self.multipole_charges = self[start:end].to_data(delim_whitespace=True,
                                                         names=self._key_d_multi[2])
        nat = len(self.multipole_charges)
        # Energy gradients
        start = found[self._key_d_nrggrad[0]][0][0] + self._key_d_nrggrad[1]
        end = start + nat
        self.energy_gradients = self[start:end].to_data(delim_whitespace=True,
                                                        names=self._key_d_nrggrad[2])
        # Convergence info
        start = found[self._key_d_conv[0]][0][0] + self._key_d_conv[1]
        conv_idx = []
        conv_dat = []
        for i in range(start, start+self._key_d_conv[2]):
            line = str(self[i])
            conv_idx.append(line[:self._key_d_conv[3]].strip())
            conv_dat.append(line[self._key_d_conv[3]:].strip().split()[0])
        self.convergence_info = pd.Series(conv_dat, index=conv_idx)
        # Coordinates
        try:
            start = found[self._key_d_coord[0]][0][0] + self._key_d_coord[1]
            end = start + nat
            self.coordinates = self[start:end].to_data(delim_whitespace=True,
                                                   names=self._key_d_coord[2])
        except IndexError:
            pass    # Converged



class Computation(Sections):
    """
    Parser for the 'C O M P U T A T I O N' subsection of an ADF calculation.
    """
    _key_delim = "^[ ]+=+$"
    _key_1 = 1
    _key_2 = 2
    _key_mapper0 = {' M U L T I P O L E   D E R I V E D   C H A R G E   A N A L Y S I S': ComputationMultipoleParser}

    def _parse(self):
        """
        """
        delims = self.regex("^[ ]+=+$", text=False)["^[ ]+=+$"]
        mains = []
        for i in range(len(delims) - self._key_1):
            if delims[i] + self._key_2 == delims[i+self._key_1]:
                mains.append(delims[i])
        titles = self[mains]._lines
        parsers = [self._key_mapper0.get(title, title) for title in titles]
        starts = mains
        ends = mains[1:]
        ends.append(len(self))
        self._sections_helper(start=starts, end=ends, parser=parsers, title=titles)
