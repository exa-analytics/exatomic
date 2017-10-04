## -*- coding: utf-8 -*-
## Copyright (c) 2015-2017, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#Composer for 'dirac'
####################################
#The 'dirac' executable performs all electron calculations, prerequisite for
#'adf' calculations.
#"""
#import six
#from exa import Composer, TypedProperty
#
#
#_template = """{title}
#{line2}
#{line3}
#{line4}
#{line5}
#{orbitals}
#{line_1}
#"""
#
#
#class DiracInput(Composer):
#    """
#    Composer for the 'dirac' executable of the ADF modeling suite.
#
#    The 'dirac' program performs all electron calculations on atoms. It solves
#    the radial quantum problem. The input file format can be found in the
#    source code of the 'dirac.f90' file.
#    """
#    joiner = " "
#    delimter = None
#    line2 = TypedProperty((list, tuple, str))
#    line3 = TypedProperty((list, tuple, str))
#    line4 = TypedProperty((list, tuple, str))
#    line5 = TypedProperty((list, tuple, str))
#    orbitals = TypedProperty((list, tuple, str))
#    line_1 = TypedProperty((list, tuple, str), docs="Final line of DIRAC input")
#
#    def __init__(self, title, line2, line3, line4, line5, orbitals, line_1):
#        self.title = title
#        self.line2 = line2
#        self.line3 = line3
#        self.line4 = line4
#        self.line5 = line5
#        self.orbitals = orbitals
#        self.line_1 = line_1
