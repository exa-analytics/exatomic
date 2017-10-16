# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
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
import numpy as np
import pandas as pd
from exa import Field, DataFrame, Container, TypedMeta
from .error import BasisSetNotFoundError
from .frame import Frame, compute_frame_from_atom
from .atom import Atom, UnitAtom, ProjectedAtom, VisualAtom, Frequency
from .two import (AtomTwo, MoleculeTwo, compute_atom_two,
                  _compute_bond_count, compute_molecule_two, _compute_bonds)
from .molecule import (Molecule, compute_molecule, compute_molecule_com,
                       compute_molecule_count)
from .field import AtomicField
from .orbital import Orbital, Excitation, MOMatrix, DensityMatrix
from .basis import Overlap, BasisSet, BasisSetOrder
from exatomic.algorithms.orbital import add_molecular_orbitals
#from exatomic.interfaces.cclib import universe_from_cclib
from exatomic.widget import UniverseWidget


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
    overlap = Overlap
    multipole = DataFrame
    momatrix = MOMatrix
    excitation = Excitation
    density = DensityMatrix
    contribution = DataFrame
    basis_set_order = BasisSetOrder
    basis_set = BasisSet


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

    def compute_density(self):
        """Compute density from momatrix and occupation vector."""
        if not hasattr(self, 'occupation_vector'):
            raise Exception('Universe must have momatrix and occupation_vector attributes')
        self.density = DensityMatrix.from_momatrix(self.momatrix, self.occupation_vector)

    def add_field(self, field):
        """Adds a field object to the universe."""
        self._traits_need_update = True
        if isinstance(field, AtomicField):
            if not hasattr(self, 'field'):
                self.field = field
            else:
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
                               vector=None, frame=None):
        """
        Adds molecular orbitals to universe. field_params define the numerical
        field and may be a tuple of (min, max, nsteps) or a series containing
        all of the columns specified in the exatomic.field.AtomicField table.

        Warning:
            Removes any existing field attribute of the universe.
        """
        for attr in ['momatrix', 'basis_set', 'basis_set_order']:
            if not hasattr(self, attr):
                raise AttributeError("universe must have {} attribute.".format(attr))
        add_molecular_orbitals(self, field_params=field_params,
                               mocoefs=mocoefs, vector=vector, frame=frame)

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
