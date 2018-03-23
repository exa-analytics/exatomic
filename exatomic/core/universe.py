# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
The Atomic Universe
#########################
The :class:`~exatomic.container.Universe` object is a subclass of
:class:`~exa.container.Container` that stores data coming from computational
chemistry experiments in a unified and systematic way. Data is organized into
"frames". A frame is an axis that can represent time (e.g. molecular dynamics
simulations), step number (e.g. geometry optimization), or an arbitrary index
(e.g. density functional theory exchange correlation functional).
"""
import six
import pandas as pd
from exa import DataFrame, Container, TypedMeta
from .frame import Frame, compute_frame_from_atom
from .atom import Atom, UnitAtom, ProjectedAtom, VisualAtom, Frequency
from .two import (AtomTwo, MoleculeTwo, compute_atom_two,
                  _compute_bond_count, _compute_bonds)
from .molecule import (Molecule, compute_molecule, compute_molecule_com,
                       compute_molecule_count)
from .field import AtomicField
from .orbital import Orbital, Excitation, MOMatrix, DensityMatrix
from .basis import Overlap, BasisSet, BasisSetOrder
from exatomic.algorithms.orbital import add_molecular_orbitals
from exatomic.algorithms.basis import Basis



class Meta(TypedMeta):
    atom = Atom
    frame = Frame
    atom_two = AtomTwo
    unit_atom = UnitAtom
    projected_atom = ProjectedAtom
    visual_atom = VisualAtom
    frequency = Frequency
    molecule = Molecule
    molecule_two = MoleculeTwo
    field = AtomicField
    orbital = Orbital
    momatrix = MOMatrix
    excitation = Excitation
    overlap = Overlap
    density = DensityMatrix
    basis_set_order = BasisSetOrder
    basis_set = BasisSet
    basis_dims = dict
    basis_functions = Basis
    contribution = DataFrame
    multipole = DataFrame


class Universe(six.with_metaclass(Meta, Container)):
    """
    The atomic container is called a universe because it represents everything
    known about the atomistic simulation (whether quantum or classical). This
    includes data such as atomic coordinates, molecular orbital energies, as
    well as (classical phenomena) such as two body distances, etc.

    Attributes:
        frame (:class:`~exatomic.core.frame.Frame`): State variables:
        atom (:class:`~exatomic.core.atom.Atom`): (Classical) atomic data (e.g. coordinates)
        atom_two (:class:`~exatomic.core.two.AtomTwo`): Interatomic distances
        molecule (:class:`~exatomic.core.molecule.Molecule`): Molecule information
        orbital (:class:`~exatomic.core.orbital.Orbital`): Molecular orbital information
        momatrix (:class:`~exatomic.core.orbital.MOMatrix`): Molecular orbital coefficient matrix
    """
    _cardinal = "frame"
    _getter_prefix = "compute"

    @property
    def periodic(self, *args, **kwargs):
        return self.frame.is_periodic(*args, **kwargs)

    @property
    def orthorhombic(self):
        return self.frame.orthorhombic()

    @classmethod
    def from_cclib(cls, ccobj):
        from exatomic.interfaces.cclib import universe_from_cclib
        return cls(**universe_from_cclib(ccobj))

    # Note that compute_* function may be called automatically by typed
    # properties defined in UniverseMeta
    def compute_frame(self):
        """
        Compute a minmal frame table.
        """
        self.frame = compute_frame_from_atom(self.atom)

    def compute_unit_atom(self):
        """Compute minimal image for periodic systems."""
        self.unit_atom = UnitAtom.from_universe(self)

    def compute_visual_atom(self):
        self.visual_atom = VisualAtom.from_universe(self)
        self.compute_molecule_com()

    def compute_atom_two(self, *args, **kwargs):
        """
        Compute interatomic two body properties (e.g. bonds).

        Args:
            mapper (dict): Custom radii to use when determining bonds
            bond_extra (float): Extra additive factor to use when determining bonds
        """
        self.atom_two = compute_atom_two(self, *args, **kwargs)

    def compute_bonds(self, *args, **kwargs):
        """
        Updates bonds (and molecules).

        See Also:
            :func:`~exatomic.two.AtomTwo.compute_bonds`
        """
        _compute_bonds(self.atom, self.atom_two, *args, **kwargs)

    def compute_bond_count(self):
        """
        Compute bond counts and attach them to the :class:`~exatomic.atom.Atom` table.
        """
        _compute_bond_count(self)

    def compute_molecule(self):
        """Compute the :class:`~exatomic.molecule.Molecule` table."""
        self.molecule = compute_molecule(self)
        self.compute_molecule_count()

    def compute_molecule_com(self):
        cx, cy, cz = compute_molecule_com(self)
        self.molecule['cx'] = cx
        self.molecule['cy'] = cy
        self.molecule['cz'] = cz

    def compute_atom_count(self):
        """Compute number of atoms per frame."""
        self.frame['atom_count'] = self.atom.cardinal_groupby().size()

    def compute_molecule_count(self):
        """Compute number of molecules per frame."""
        self.frame['molecule_count'] = compute_molecule_count(self)

    def compute_basis_dims(self):
        """Compute basis dimensions."""
        bset = self.basis_set
        mapr = self.atom.set.map
        self.basis_dims = {
            'npc': mapr(bset.primitives(False).groupby('set').sum()).sum(),
            'nps': mapr(bset.primitives(True).groupby('set').sum()).sum(),
            'ncc': mapr(bset.functions(False).groupby('set').sum()).sum(),
            'ncs': mapr(bset.functions(True).groupby('set').sum()).sum(),
            'sets': bset.functions_by_shell()}

    def compute_basis_functions(self, **kwargs):
        self.basis_functions = Basis(self)

    def enumerate_shells(self, frame=0):
        atom = self.atom[self.atom.frame == frame]
        shls = self.basis_set.shells()
        grps = shls.groupby('set')
        # Pointers into (xyzs, shls) arrays
        ptrs = np.array([(c, idx) for c, seht in enumerate(atom.set)
                                  for idx in grps.get_group(seht).index])
        return ptrs, atom[['x', 'y', 'z']].values, shls[0].values

    def add_field(self, field):
        """Adds a field object to the universe."""
        self._traits_need_update = True
        if isinstance(field, AtomicField):
            if not hasattr(self, 'field'):
                self.field = field
            else:
                self.field._revert_categories()
                new_field_values = self.field.field_values + field.field_values
                newdx = range(len(self.field), len(self.field) + len(field))
                field.index = newdx
                new_field = pd.concat([self.field, field])
                self.field = AtomicField(new_field, field_values=new_field_values)
        elif isinstance(field, list):
            if not hasattr(self, 'field'):
                fields = pd.concat(field)
                fields.index = range(len(fields))
                fields_values = [j for i in field for j in i.field_values]
                self.field = AtomicField(fields, field_values=fields_values)
            else:
                new_field_values = self.field.field_values + [j for i in field for j in i.field_values]
                newdx = range(len(self.field), len(self.field) + sum([len(i.field_values) for i in field]))
                for i, idx in enumerate(newdx):
                    field[i].index = [idx]
                new_field = pd.concat([self.field] + field)
                self.field = AtomicField(new_field, field_values=new_field_values)
        else:
            raise TypeError('field must be an instance of exatomic.field.AtomicField or a list of them')

    def add_molecular_orbitals(self, field_params=None, mocoefs=None,
                               vector=None, frame=0, replace=True):
        """Add molecular orbitals to universe.

        Args
            field_params (dict, pd.Series): see `:meth:exatomic.algorithms.orbital_util.make_fps`
            mocoefs (str): column in the :class:`~exatomic.core.orbital.MOMatrix`
            vector (iter): indices of orbitals to evaluate (0-based)
            frame (int): frame of atomic positions for the orbitals
            replace (bool): if False, do not remove previous fields
        """
        assert hasattr(self, 'momatrix')
        assert hasattr(self, 'basis_set')
        assert hasattr(self, 'basis_set_order')
        add_molecular_orbitals(self, field_params=field_params,
                               mocoefs=mocoefs, vector=vector,
                               frame=frame, replace=replace)

    def __len__(self):
        return len(self.frame)

    def __init__(self, **kwargs):
        super(Universe, self).__init__(**kwargs)


def concat(name=None, description=None, meta=None, *universes):
    """
    Warning:
        This function is not fully featured or tested yet!
    """
    raise NotImplementedError()


def basis_function_contributions(universe, mo, mocoefs='coef',
                                 tol=0.01, ao=None, frame=0):
    """
    Provided a universe with momatrix and basis_set_order attributes,
    return the major basis function contributions of a particular
    molecular orbital.

    Args
        universe (exatomic.container.Universe): a universe
        mo (int): molecular orbital index
        mocoefs (str): column of interest in universe.momatrix
        tol (float): minimum value of coefficient by which to filter
        frame (int): frame of the universe (default is zero)

    Returns
        together (pd.DataFrame): a join of momatrix and basis_set_order
    """
    small = universe.momatrix.contributions(mo, tol=tol, mocoefs=mocoefs, frame=frame)
    chis = small['chi'].values
    coefs = small[mocoefs]
    coefs.index = chis
    together = pd.concat([universe.basis_set_order.ix[chis], coefs], axis=1)
    if ao is None:
        return together
    else:
        raise NotImplementedError("not clever enough for that.")
