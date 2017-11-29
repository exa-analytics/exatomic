# -*- coding: utf-8 -*-
#'''
#Classes for pw.x
#=======================================
#'''
#import pprint
#from exatomic import Universe
#from exatomic.frame import _min_frame_from_atom
#from exqe.classes import classes
#
#
#class PWInput:
#    '''
#    '''
#    _order = ['control', 'system', 'electrons', 'ions', 'cell', 'atomic_species',
#              'atomic_positions', 'k_points', 'cell_parameters', 'occupations',
#              'constraints', 'atomic_forces']
#
#    def to_universe(self):
#        '''
#        '''
#        atom = self.atomic_positions
#        atom['frame'] = 0
#        atom['label'] = range(len(atom))
#        atom.index.names = ['atom']
#        frame = _min_frame_from_atom(atom)
#        return Universe(frame=frame, atom=atom)
#
#    def __str__(self):
#        '''
#        '''
#        blocks = []
#        for block_name in self._order:
#            block = self[block_name]
#            if block is not None:
#                blocks.append(str(block))
#        return '\n'.join(blocks)
#
#    def __getitem__(self, key):
#        return self.__dict__[key]
#
#    def __setitem__(self, key, value):
#        setattr(self, key, value)
#
#    def __init__(self, control=None, system=None, electrons=None, ions=None,
#                 cell=None, atomic_species=None, atomic_positions=None,
#                 k_points=None, cell_parameters=None, occupations=None,
#                 constraints=None, atomic_forces=None):
#        '''
#        '''
#        self.control = control
#        self.system = system
#        self.electrons = electrons
#        self.ions = ions
#        self.cell = cell
#        self.atomic_species = atomic_species
#        self.atomic_positions = atomic_positions
#        self.k_points = k_points
#        self.cell_parameters = cell_parameters
#        self.occupations = occupations
#        self.constraints = constraints
#        self.atomic_forces = atomic_forces
#
#    def __repr__(self):
#        obj = str(self).split('\n')
#        pprint.pprint(obj, indent=0, width=128)
#        return 'PWInput(len: {0})'.format(len(obj))
#
#
#class PWOutput:
#    '''
#    '''
#    def to_universe(self):
#        uni = Universe(atom=self.atom, frame=self.frame)
#        return uni
#
#    def __init__(self, timings=None, atom=None, frame=None, scf=None,
#                 orbital=None):
#        self.timings = timings
#        self.atom = atom
#        self.frame = frame
#        self.scf = scf
#        self.orbital = orbital
#
