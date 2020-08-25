## -*- coding: utf-8 -*-
## Copyright (c) 2015-2020, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#PW SCF Iteration Parser
##############################
#"""
#from exa.typed import TypedProperty
#from exa.core.parser import Sections, Parser
#
#
#class Iterations(Parser):
#    def _parse(self):
#        pass
#
#
#class Eigenvalues(Parser):
#    def _parse(self):
#        pass
#
#
#class Occupations(Parser):
#    def _parse(self):
#        pass
#
#
#class Energies(Parser):
#    def _parse(self):
#        pass
#
#
#class Forces(Parser):
#    def _parse(self):
#        pass
#
#
#class AtomPos(Parser):
#    def _parse(self):
#        pass
#
#
#class SCF(Sections):
#    """
#    """
#    _key_parsers = [Iterations, Eigenvalues, Occupations, Energies, Forces,
#                    AtomPos]
#    _key_delims = ("End of self-consistent calculation",
#                   "occupation numbers",
#                   "the Fermi energy",
#                   "Forces acting on atoms",
#                   "ATOMIC_POSITIONS")
#
#    def _parse(self):
#        """Identify subsections"""
#        delims = self.find(*self._key_delims, text=False)
#        self.delims = delims
#        starts = [0]
#        parsers = [self._key_parsers[0]]
#        for i, value in enumerate(delims.values(), start=1):
#            if len(value) == 1:
#                starts += value
#                parsers.append(self._key_parsers[i])
#        ends = starts[1:]
#        ends.append(len(self))
#        self._sections_helper(parser=parsers, end=ends, start=starts)
#
