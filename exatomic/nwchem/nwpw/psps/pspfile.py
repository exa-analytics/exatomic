# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
PSP File Edtor
##########################
These parsing editors are targeted at parsing pseudized wave function and potential
data.
"""
import six
import numpy as np
import pandas as pd
from exa.special import LazyFunction
from exa.tex import text_value_cleaner
from exa.core import Meta, Parser, DataFrame
from exatomic.nwchem.nwpw.pseudopotentials.grid import scaled_logspace
angular_momentum_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "j": 7}
angular_momentum_map_inv = {v: k for k, v in angular_momentum_map.items()}


class PAWPSPMeta(Meta):
    """Defines data objects for PAWOutput."""
    fields = DataFrame
    data = DataFrame
    info = dict
    grid = LazyFunction
    _descriptions = {'data': "Pseudized channel data",
                     'info': "Atom, charge, and core info",
                     'grid': "Grid information",
                     'fields': "Stuff"}

class PAWPSP(six.with_metaclass(PAWPSPMeta, Parser)):
    """
    """
    _key_info_names = ["psp_type", "atom_name", "valence_z", "rmin",
                       "rmax", "nr", "nbasis", "rcuts", "max_i_r",
                       "comment", "core_kinetic_energy"]
    def _parse(self):
        info = {}
        for i, name in enumerate(self._key_info_names):
            info[name] = text_value_cleaner(str(self[i]))
        self.info = info
        #n = self.info['nbasis']
        nr = self.info['nr']
        self.grid = LazyFunction(scaled_logspace, self.info['rmin'], 1.005, nr-1)
        i += 1
        start = i + self.info['nbasis']
        self.data = pd.read_csv(self[i:start].to_stream(), delim_whitespace=True,
                                names=('n', 'eigenvalue', 'nps', 'l'))
        self.data['rcut'] = self.info['rcuts'].split()
        self.data['rcut'] = self.data['rcut'].astype(np.float64)
        nls = (self.data['n'].astype(str) + self.data['l'].map(angular_momentum_map)).tolist()
        fields = pd.DataFrame()
        for name in [r"$\psi_nl(r)$", r"$\psi^{'}_nl(r)$", r"$\tilde{\psi}_nl(r)$",
                        r"$\tilde{\psi}^{'}_nl(r)$", r"$\tilde{p}_nl(r)$"]:
            for nl in nls:
                end = start + nr
                wave = pd.read_csv(self[start:end].to_stream(), delim_whitespace=True)
                wave_name = name.replace('nl', "{" + nl + "}")
                fields[wave_name] = wave
                start += nr
        for name in [r"$\rho_{\text{core}}(r)/(4\pi)$",
                     r"$\rho_{\text{core, ps}}(r)/(4\pi)$",
                     r"$V_{ps}(r)$"]:
            end = start + nr
            rho = pd.read_csv(self[start:end].to_stream(), delim_whitespace=True)
            fields[name] = rho
            start += nr
        self.info[r"$\sigma_{comp}$"] = text_value_cleaner(str(self[start]))
        start += 1
        self.info["Z"] = text_value_cleaner(str(self[start]))
        start += 1
        for nl in nls:
            end = start + nr
            wave = pd.read_csv(self[start:end].to_stream(), delim_whitespace=True)
            wave_name = r"$\tilde{p}_{" + nl + ",0}(r)$"
            fields[wave_name] = wave
            start += nr
        fields.index = self.grid()
        self.fields = fields
