# -*- coding: utf-8 -*-
#'''
#PW Module Input File Editor
#====================================
#
#'''
#import numpy as np
#from exa import DataFrame
#from exatomic import Length, Frame, Atom
#from exqe.input import QEInput
#from exqe.types import lengths
#
#
#class PWInput(QEInput):
#    '''
#    Editor representation of QE's pw.x input file.
#    '''
#    def parse_frame(self):
#        '''
#        Parse the :class:`~atomic.frame.Frame` dataframe and store
#        it locally, accessible through the ".frame" property.
#        '''
#        nat = len(self.atom)
#        df = DataFrame.from_dict({'atom_count': [nat]})
#        self._frame = Frame(df)
#
#    def parse_cell(self):
#        '''
#        Determine the type of unit cell being used.
#        '''
#        ibrav = int(list(self.find('ibrav').values())[0].split('=')[1])
#        if ibrav == 1:
#            a = np.float64(list(self.find('celldm(1)').values())[0].split('=')[1])
#            frame = self.frame
#            frame['xi'] = a
#            frame['xj'] = 0.0
#            frame['xk'] = 0.0
#            frame['yi'] = 0.0
#            frame['yj'] = a
#            frame['yk'] = 0.0
#            frame['zi'] = 0.0
#            frame['zj'] = 0.0
#            frame['zk'] = a
#            frame['ox'] = 0.0
#            frame['oy'] = 0.0
#            frame['oz'] = 0.0
#            self._frame = frame
#        else:
#            raise NotImplementedError()
#
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        self._atom = None
#        self._frame = None
#
