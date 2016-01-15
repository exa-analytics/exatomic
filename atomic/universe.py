# -*- coding: utf-8 -*-
'''
Universe
====================
The atomic container object.
'''
from exa import Container
from exa.relational.base import Column, Integer, ForeignKey
from atomic.atom import Atom, SuperCellAtom, VisualAtom
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

    Warning:
        The correct way to set DataFrame object is as follows:

        .. code-block:: Python

            universe = atomic.Universe()
            df = pd.DataFrame()
            universe['atoms'] = df
            or
            setattr(universe, 'atoms', df)

        Avoid setting objects using the **__dict__** attribute as follows:

        .. code-block:: Python

            universe = atomic.Universe()
            df = pd.DataFrame()
            universe.atoms = df

        (This is used in **__init__** where type control is enforced.)
    '''
    cid = Column(Integer, ForeignKey('container.pkid'), primary_key=True)
    frame_count = Column(Integer)
    __mapper_args__ = {'polymorphic_identity': 'universe'}
    __dfclasses__ = {'atoms': Atom, 'frames': Frame, 'pbcatoms': SuperCellAtom,
                     'visatoms': VisualAtom}

    def cell_mags(self):
        '''
        Compute periodic cell dimension magnitudes.

        See Also:
            :func:`~atomic.frame.Frame.cell_mags`
        '''
        self.frames.cell_mags()

    def get_twobody(self, k=None, bond_extra=0.45, inplace=False,
                    dmax=13.0, dmin=0.3):
        '''
        Compute two body information from the current atom dataframe.

        This function does not return the computed dataframe but rather
        attaches it directly to the active universe object (**obj.twobody**).

        Args:
            k (int): Number of distances (per atom) to compute
            bond_extra (float): Extra distance to include when determining bonds (see above)
            dmax (float): Max distance of interest (larger distances are ignored)
            dmin (float): Min distance of interest (smaller distances are ignored)
            inplace (bool): Perform action in place (default False)

        See Also:
            :func:`~atomic.atom.compute_twobody`.
        '''
        twobody = _compute_twobody(self, k=k, bond_extra=bond_extra, dmax=dmax, dmin=dmin)
        if inplace:
            self.twobody = twobody
        else:
            return twobody

    @classmethod
    def from_xyz(cls, path, unit='A'):
        raise NotImplementedError()

    def __init__(self, atoms=None, frames=None, twobody=None,
                 molecule=None, **kwargs):
        super().__init__(**kwargs)
        self.atoms = Atom(atoms)
        self.frames = Frame(frames)
