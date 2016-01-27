# -*- coding: utf-8 -*-
'''
Universe
====================
The atomic container object.
'''
from traitlets import Unicode, List
from sqlalchemy import Column, Integer, ForeignKey
from exa import Container
from atomic.frame import Frame, minimal_frame
from atomic.atom import Atom, UnitAtom, ProjectedAtom
from atomic.atom import _compute_projected_non_var_cell
from atomic.atom import get_unit_atom as _get_unit_atom
#from atomic.atom import Atom, SuperAtom, VisualAtom, PrimitiveAtom
#from atomic.atom import compute_primitive, compute_supercell
#from atomic.frame import Frame
#from atomic.twobody import TwoBody, PeriodicTwoBody
#from atomic.twobody import compute_twobody, compute_bond_counts
#from atomic.orbitals import Orbital
#from atomic.molecule import Molecule, PeriodicMolecule, _periodic_molecules
#from atomic.algorithms.nearest import get_nearest_neighbors as _get_nearest_neighbors


class Universe(Container):
    '''
    A collection of atoms, molecules, electronic data, and other relevant
    information from an atomistic simulation.
    '''
    # Relational information
    cid = Column(Integer, ForeignKey('container.pkid'), primary_key=True)
    frame_count = Column(Integer)
    __mapper_args__ = {'polymorphic_identity': 'universe'}
    __dfclasses__ = {'_frame': Frame, '_atom': Atom, '_unit_atom': UnitAtom}

    # DOMWidget settings
    _view_module = Unicode('nbextensions/exa/atomic/universe').tag(sync=True)
    _view_name = Unicode('UniverseView').tag(sync=True)
    _center = Unicode().tag(sync=True)
    _camera = Unicode().tag(sync=True)
    _framelist = List().tag(sync=True)

    def compute_cell_magnitudes(self, inplace=True):
        '''
        See Also:
            :func:`~atomic.atom.Atom.get_unit_cell_magnitudes`
        '''
        return self._frame.get_unit_cell_magnitudes(inplace)

    # DataFrame properties
    @property
    def frame(self):
        if len(self._frame) == 0 and len(self._atom) > 0:
            self._frame = minimal_frame(self.atom)
        return self._frame

    @property
    def atom(self):
        return self._atom

    @property
    def unit_atom(self):
        '''
        
        '''
        if len(self._unit_atom) == 0:
            self._unit_atom = _get_unit_atom(self)
        atom = self.atom.copy()
        atom.update(self._unit_atom)
        return atom

    @property
    def prjd_atom(self):
        '''
        '''
        if len(self._prjd_atom) == 0:
            self._prjd_atom = _compute_projected_non_var_cell(self)
        return self._prjd_atom

    def __init__(self, frame=None, atom=None, unit_atom=None, prjd_atom=None,
                 two=None, **kwargs):
        '''
        The universe container represents all of the atoms, bonds, molecules,
        orbital/densities, etc. present within an atomistic simulations.
        '''
        super().__init__(**kwargs)
        self._frame = Frame(frame)
        self._atom = Atom(atom)
        self._unit_atom = UnitAtom(unit_atom)
        self._prjd_atom = ProjectedAtom(prjd_atom)
        self._update_all_traits()
        self._center = (self._atom[['x', 'y', 'z']].mean() / 2).to_json(orient='values')
        self._camera = (self._atom[['x', 'y', 'z']].max() + 12).to_json(orient='values')
        if len(self._frame) > 0:
            self._framelist = self._frame.index.tolist()


def concat(universes):
    '''
    Concatenate a collection of universes.
    '''
    raise NotImplementedError()



#    def __init__(self, atom=None, frame=None, two=None, molecule=None,
#                 primitive_atoms=None, super_atoms=None, periodic_twobody=None,
#                 periodic_molecules=None, orbitals=None, **kwargs):
#        super().__init__(**kwargs)
#        self.atoms = Atom(atoms)
#        self.frames = Frame(frames)
#        self.orbitals = Orbital(orbitals)
#        self._primitive_atoms = PrimitiveAtom(primitive_atoms)
#        self._super_atoms = SuperAtom(super_atoms)
#        self._periodic_twobody = PeriodicTwoBody(periodic_twobody)
#        self._twobody = TwoBody(twobody)
#        self._molecules = Molecule(molecule)
##        self._periodic_molecules = PeriodicMolecule(periodic_molecules)

#    def get_cell_mags(self, inplace=False):
#        '''
#        Compute periodic cell dimension magnitudes in place.
#
#        See Also:
#            :func:`~atomic.frame.Frame.cell_mags`
#        '''
#        return self.frames.cell_mags(inplace=inplace)
#
#    def get_primitive_atoms(self, inplace=False):
#        '''
#        Compute primitive atom positions in place.
#
#        See Also:
#            :func:`~atomic.atom.compute_primitive`
#        '''
#        patoms = compute_primitive(self)
#        if inplace:
#            self._primitive_atoms = patoms
#        else:
#            return patoms
#
#    def get_super_atoms(self, inplace=False):
#        '''
#        Compute the super cell atom positions from the primitive atom positions.
#
#        See Also:
#            :func:`~atomic.atom.compute_supercell`
#        '''
#        obj = compute_supercell(self)
#        if inplace:
#            self._super_atoms = obj
#        else:
#            return obj
#
#    def get_twobody(self, inplace=False, **kwargs):
#        '''
#        Compute two body information from the current atom dataframe.
#
#        This function does not return the computed dataframe but rather
#        attaches it directly to the active universe object (**obj.twobody**).
#
#        See Also:
#            :func:`~atomic.atom.compute_twobody`.
#        '''
#        data = compute_twobody(self, **kwargs)
#        if inplace:
#            tb = data
#            if isinstance(data, tuple):
#                self._super_atoms = data[0]
#                tb = data[1]
#            if isinstance(tb, PeriodicTwoBody):
#                self._periodic_twobody = tb
#            else:
#                self._twobody = tb
#        else:
#            return data
#
#    def get_bond_counts(self, inplace=False):
#        '''
#        '''
#        counts = compute_bond_counts(self)
#        if inplace:
#            self.atoms['bond_count'] = counts
#        else:
#            return counts, periodic
#
#    def get_molecules(self, inplace=False):
#        '''
#        '''
#        obj = _periodic_molecules(self)
#        if inplace == True:
#            self['_molecules'] = obj
#        else:
#            return obj
#
#    def get_nearest_neighbors(self, count, solute, solvent):
#        '''
#        '''
#        kwargs = _get_nearest_neighbors(self, count, solute, solvent)
#        return self.__class__(**kwargs)
#
#    @property
#    def atom(self):
#        return self._atom
#
#    @property
#    def unit_atom(self):
#        a = self.atom.copy()
#        a.update(self._unit_atom)
#        return a
#
#
#    @property
#    def twobody(self):
#        '''
#        '''
#        if len(self._twobody) == 0 and len(self._periodic_twobody) == 0:
#            self.get_twobody(inplace=True)
#        if len(self._twobody) == 0:
#            return self._periodic_twobody
#        else:
#            return self._twobody
#
#    @property
#    def periodic_twobody(self):
#        '''
#        '''
#        return self.twobody
#
#    @property
#    def primitive_atoms(self):
#        '''
#        '''
#        if len(self._primitive_atoms) == 0:
#            self.get_primitive_atoms(inplace=True)
#        return self._primitive_atoms
#
#    @property
#    def super_atoms(self):
#        '''
#        '''
#        if len(self._super_atoms) == 0:
#            self.get_super_atoms(inplace=True)
#        return self._super_atoms
#
#    @property
#    def molecules(self):
#        '''
#        '''
#        if len(self._molecules) == 0:
#            self.get_molecules(inplace=True)
#        return self._molecules
#
#    @property
#    def periodic_molecules(self):
#        '''
#        '''
#        if len(self._periodic_molecules) == 0:
#            self.get_molecules(inplace=True)
#        return self._molecules
#
#    @classmethod
#    def from_xyz(cls, path, unit='A'):
#        raise NotImplementedError()
#
#    def __init__(self, atom=None, frame=None, two=None, molecule=None,
#                 primitive_atoms=None, super_atoms=None, periodic_twobody=None,
#                 periodic_molecules=None, orbitals=None, **kwargs):
#        super().__init__(**kwargs)
#        self.atoms = Atom(atoms)
#        self.frames = Frame(frames)
#        self.orbitals = Orbital(orbitals)
#        self._primitive_atoms = PrimitiveAtom(primitive_atoms)
#        self._super_atoms = SuperAtom(super_atoms)
#        self._periodic_twobody = PeriodicTwoBody(periodic_twobody)
#        self._twobody = TwoBody(twobody)
#        self._molecules = Molecule(molecule)
##        self._periodic_molecules = PeriodicMolecule(periodic_molecules)
#

#    An :class:`~atomic.universe.Universe` represents a collection of time
#    dependent frames themselves containing atoms and molecules. A frame can
#    be thought of as a snapshot in time, though the frame axis is not required
#    to be time. Each frame has information about atomic positions, energies,
#    bond distances, energies, etc. The following table outlines the structures
#    provided by this container. A description of the index or columns can be
#    found in the corresponding dataframe link.
#
#    +-------------------------------------------------------+--------------+---------------------------------+
#    | Attribute (DataFrame)                                 | Dimensions   | Required Columns                |
#    +=======================================================+==============+=================================+
#    | atoms (:class:`~atomic.atom.Atom`)                    | frame, atom  | symbol, x, y, z                 |
#    +-------------------------------------------------------+--------------+---------------------------------+
#    | twobody (:class:`~atomic.twobody.TwoBody`)            | frame, index | atom1, atom2, symbols, distance |
#    +-------------------------------------------------------+--------------+---------------------------------+
#    | eigenvalues (:class:`~atomic.eigenvalues.EigenValue`) | frame, index | energy                          |
#    +-------------------------------------------------------+--------------+---------------------------------+
#
#    Warning:
#        The correct way to set DataFrame object is as follows:
#
#        .. code-block:: Python
#
#            universe = atomic.Universe()
#            df = pd.DataFrame()
#            universe['atoms'] = df
#            or
#            setattr(universe, 'atoms', df)
#
#        Avoid setting objects using the **__dict__** attribute as follows:
#
#        .. code-block:: Python
#
#            universe = atomic.Universe()
#            df = pd.DataFrame()
#            universe.atoms = df
#
#        (This is used in **__init__** where type control is enforced.)
