# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Comprehensive Planewave (DFT) Output Parser
############################################
"""
import pandas as pd
from exa import Parser, Typed
from exatomic.core.atom import Atom
from .geometry import Geometry


class Output(Parser):
    """
    """
    atom = Typed(Atom, doc="Full atom table from all 'frames'.")
    frame_map = Typed((list, tuple), doc="Correctly ordered frames")

    def parse_atom(self):
        """
        Generate the complete :class:`~exatomic.core.atom.Atom` table
        for the entire calculation.

        Warning:
            The frame index is arbitrary! Map it to the correct
            values as required!
        """
        atoms = []
        for i, sec in enumerate(self.get_sections(Geometry)):
            atom = sec.atom
            fdx = i if self.frame_map is None else self.frame_map[i]
            atom['frame'] = fdx
            atoms.append(atom)
        self.atom = pd.concat(atoms, ignore_index=True)

    def __init__(self, *args, **kwargs):
        frame_map = kwargs.pop("frame_map", None)
        super(Output, self).__init__(*args, **kwargs)
        self.frame_map = frame_map


Output.add_parsers(Geometry)
