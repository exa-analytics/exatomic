# -*- coding: utf-8 -*-
## Copyright (c) 2015-2016, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#QE Output Editor
#====================================
#"""
#from exatomic.frame import minimal_frame
#from exqe.editor import Editor
#
#
#class QEOutput(Editor):
#    """
#    Many QE output files have similar syntax.
#    """
#    def parse_frame(self):
#        """
#        Parse the :class:`~atomic.frame.Frame` dataframe.
#        """
#        frame = minimal_frame(self.atom)
#        frame['energy'] = [float(v.split()[-2]) for v in self.find(_frame01)[_frame01].values()]
#        self._frame = frame
#
#
#_frame01 = '!'
#
