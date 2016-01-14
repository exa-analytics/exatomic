# -*- coding: utf-8 -*-
'''
Universe
====================
The atomic container object.
'''
from exa import Container
from exa.relational.base import Column, Integer, ForeignKey
from atomic.atom import Atom
from atomic.atom import compute_twobody as _compute_twobody
from atomic.twobody import TwoBody
from atomic.frame import Frame


class Universe(Container):
    '''
    A collection of atoms.

    An :class:`~atomic.universe.Universe` represents a collection of time
    dependent frames themselves containing atoms and molecules. A frame can
    be thought of as a snapshot in time, though the frame axis is not required
    to be time. Each frame has information about atomic positions, energies,
    bond distances, energies, etc. The following table outlines the structures
    provided by this container. A description of the index or columns can be
    found in the corresponding dataframe link.

    +--------------------------------------------+--------------+---------------------------------+
    | Attribute (DataFrame)                      | Dimensions   | Required Columns                |
    +============================================+==============+=================================+
    | atoms (:class:`~atomic.atom.Atom`)         | frame, atom  | symbol, x, y, z                 |
    +--------------------------------------------+--------------+---------------------------------+
    | twobody (:class:`~atomic.twobody.TwoBody`) | frame, index | atom1, atom2, symbols, distance |
    +--------------------------------------------+--------------+---------------------------------+
    '''
    cid = Column(Integer, ForeignKey('container.pkid'), primary_key=True)
    frame_count = Column(Integer)
    __mapper_args__ = {'polymorphic_identity': 'universe'}

    def compute_cell_magnitudes(self):
        '''
        Compute periodic cell dimension magnitudes.

        See Also:
            :func:`~atomic.frame.Frame.cell_mags`
        '''
        self.frames.cell_mags()

    def compute_twobody(self, k=None, bond_extra=0.45, dmax=13.0, dmin=0.3):
        '''
        Compute two body information from the current atom dataframe.

        This function does not return the computed dataframe but rather
        attaches it directly to the active universe object (**obj.twobody**).

        Args:
            k (int): Number of distances (per atom) to compute
            bond_extra (float): Extra distance to include when determining bonds (see above)
            dmax (float): Max distance of interest (larger distances are ignored)
            dmin (float): Min distance of interest (smaller distances are ignored)

        See Also:
            :func:`~atomic.atom.compute_twobody`.
        '''
        self.two = _compute_twobody(self.atoms, *args, **kwargs)

    @classmethod
    def from_xyz(cls, path, unit='A'):
        raise NotImplementedError()

    def __init__(self, atoms=None, frames=None, twobody=None,
                 molecule=None, **kwargs):
        '''
        '''
        super().__init__(**kwargs)
        self.atoms = Atom(atoms)
        self.frames = Frame(frames)
        self.twobody = TwoBody(twobody)
