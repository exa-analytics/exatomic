# -*- coding: utf-8 -*-
'''
Universe
====================
The atomic container object.
'''
from exa import Container
from exa.relational.base import Column, Integer, ForeignKey
from atomic.atom import Atom, SuperAtom, VisualAtom, PrimitiveAtom
from atomic.atom import compute_primitive, compute_supercell, compute_twobody
from atomic.frame import Frame
from atomic.twobody import TwoBody, PeriodicTwoBody
from atomic.orbitals import Orbital
from atomic.molecule import Molecule


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

    +-------------------------------------------------------+--------------+---------------------------------+
    | Attribute (DataFrame)                                 | Dimensions   | Required Columns                |
    +=======================================================+==============+=================================+
    | atoms (:class:`~atomic.atom.Atom`)                    | frame, atom  | symbol, x, y, z                 |
    +-------------------------------------------------------+--------------+---------------------------------+
    | twobody (:class:`~atomic.twobody.TwoBody`)            | frame, index | atom1, atom2, symbols, distance |
    +-------------------------------------------------------+--------------+---------------------------------+
    | eigenvalues (:class:`~atomic.eigenvalues.EigenValue`) | frame, index | energy                          |
    +-------------------------------------------------------+--------------+---------------------------------+

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
    __dfclasses__ = {'atoms': Atom, 'frames': Frame, 'pbcatoms': SuperAtom,
                     'visatoms': VisualAtom}

    def get_cell_mags(self, inplace=False):
        '''
        Compute periodic cell dimension magnitudes in place.

        See Also:
            :func:`~atomic.frame.Frame.cell_mags`
        '''
        return self.frames.cell_mags(inplace=inplace)

    def get_primitive_atoms(self, inplace=False):
        '''
        Compute primitive atom positions in place.

        See Also:
            :func:`~atomic.atom.compute_primitive`
        '''
        patoms = compute_primitive(self)
        if inplace:
            self._primitive_atoms = patoms
        else:
            return patoms

    def get_super_atoms(self, inplace=False):
        '''
        Compute the super cell atom positions from the primitive atom positions.

        See Also:
            :func:`~atomic.atom.compute_supercell`
        '''
        obj = compute_supercell(self)
        if inplace:
            self._super_atoms = obj
        else:
            return obj

    def get_twobody(self, inplace=False, **kwargs):
        '''
        Compute two body information from the current atom dataframe.

        This function does not return the computed dataframe but rather
        attaches it directly to the active universe object (**obj.twobody**).

        See Also:
            :func:`~atomic.atom.compute_twobody`.
        '''
        data = compute_twobody(self, **kwargs)
        if inplace:
            tb = data
            if isinstance(data, tuple):
                self._super_atoms = data[0]
                tb = data[1]
            if isinstance(tb, PeriodicTwoBody):
                self._periodic_twobody = tb
            else:
                self._twobody = tb
        else:
            return data

    @property
    def twobody(self):
        '''
        '''
        if len(self._twobody) == 0 and len(self._periodic_twobody) == 0:
            self.get_twobody(inplace=True)
        if len(self._twobody) == 0:
            return self._periodic_twobody
        else:
            return self._twobody

    @property
    def periodic_twobody(self):
        '''
        '''
        return self.twobody

    @property
    def primitive_atoms(self):
        '''
        '''
        if len(self._primitive_atoms) == 0:
            self.get_primitive_atoms(inplace=True)
        return self._primitive_atoms

    @property
    def super_atoms(self):
        '''
        '''
        if len(self._super_atoms) == 0:
            self.get_super_atoms(inplace=True)
        return self._super_atoms

    @classmethod
    def from_xyz(cls, path, unit='A'):
        raise NotImplementedError()

    def __init__(self, atoms=None, frames=None, twobody=None, molecule=None,
                 primitive_atoms=None, super_atoms=None, periodic_twobody=None,
                 orbitals=None, **kwargs):
        super().__init__(**kwargs)
        self.atoms = Atom(atoms)
        self.frames = Frame(frames)
        self.orbitals = Orbital(orbitals)
        self._primitive_atoms = PrimitiveAtom(primitive_atoms)
        self._super_atoms = SuperAtom(super_atoms)
        self._periodic_twobody = PeriodicTwoBody(periodic_twobody)
        self._twobody = TwoBody(twobody)
        self._molecule = Molecule(molecule)
