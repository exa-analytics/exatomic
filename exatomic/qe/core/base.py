# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Base Output Parser
###############################
"""
#import pandas as pd
#from exa import Parser
#from exa.typed import Typed
#from exatomic.core.atom import Atom
#from .atomic_positions import AtomicPositions
#
#
#class Output(Parser):
#    """
#    """
#    atom = Typed(Atom, doc="Full table of all atom coordinates (frames)")
#
#    def parse_atom(self):
#        """Generate the complete atom table."""
#        atoms = []
#        for i, sec in enumerate(self.get_sections(AtomicPositions)):
#            atom = sec.atom
#            atom['frame'] = i
#            atoms.append(atom)
#        self.atom = pd.concat(atoms, ignore_index=True)
#
#
#Output.add_parsers(AtomicPositions)
