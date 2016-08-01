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
import pandas as pd
import numpy as np
from exa.numerical import Field
from exa.container import TypedMeta, Container
from exatomic.error import BasisSetNotFoundError
from exatomic.widget import UniverseWidget
from exatomic.frame import Frame, compute_frame_from_atom
from exatomic.atom import Atom, UnitAtom, ProjectedAtom, VisualAtom
from exatomic.two import (AtomTwo, MoleculeTwo, compute_atom_two,
                          compute_bond_count, compute_molecule_two)
from exatomic.molecule import Molecule, compute_molecule, compute_com
from exatomic.widget import UniverseWidget
from exatomic.field import AtomicField
from exatomic.orbital import Orbital, MOMatrix, DensityMatrix
from exatomic.basis import (SphericalGTFOrder, CartesianGTFOrder, Overlap,
                            BasisSetSummary, GaussianBasisSet, BasisSetOrder,
                            Primitive)


class Meta(TypedMeta):
    """
    Defines strongly typed attributes of the :class:`~exatomic.universe.Universe`
    and :class:`~exatomic.editor.AtomicEditor` objects. All "aliases" below are
    in fact type definitions that get dynamically generated on package load
    for :class:`~exatomic.container.Universe` and :class:`~exatomic.editor.Editor`.
    """
    atom = Atom
    frame = Frame
    atom_two = AtomTwo
    unit_atom = UnitAtom
    projected_atom = ProjectedAtom
    visual_atom = VisualAtom
    molecule = Molecule
    molecule_two = MoleculeTwo
    field = AtomicField
    orbital = Orbital
    overlap = Overlap
    momatrix = MOMatrix
    primitive = Primitive
    density = DensityMatrix
    basis_set_order = BasisSetOrder
    basis_set_summary = BasisSetSummary
    gaussian_basis_set = GaussianBasisSet
    spherical_gtf_order = SphericalGTFOrder
    cartesian_gtf_order = CartesianGTFOrder


class Universe(Container, metaclass=Meta):
    """
    The atomic container is called a universe because it represents everything
    known about the atomistic simulation (whether quantum or classical). This
    includes data such as atomic coordinates, molecular orbital energies, as
    well as (classical phenomena) such as two body distances, etc.

    Attributes:
        atom (:class:`~exatomic.atom.Atom`): Atomic coordinates, symbols, forces, etc.
    """
    _widget_class = UniverseWidget
    _cardinal = 'frame'

    @property
    def basis_set(self):
        """
        Attempts to find the correct basis set table for the universe.
        """
        if hasattr(self, '_gaussian_basis_set'):
            return self.gaussian_basis_set
        else:
            raise BasisSetNotFoundError()

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
        """"""
        self.visual_atom = VisualAtom.from_universe(self)

    def compute_atom_two(self, mapper=None, bond_extra=0.45):
        """
        Compute interatomic two body properties (e.g. bonds).

        Args:
            mapper (dict): Custom radii to use when determining bonds
            bond_extra (float): Extra additive factor to use when determining bonds
        """
        if self.frame.is_periodic():
            atom_two, projected_atom = compute_atom_two(self, mapper, bond_extra)
            self.atom_two = atom_two
            self.projected_atom = projected_atom
        else:
            self.atom_two = compute_atom_two(self, mapper, bond_extra)

    def compute_bonds(self, mapper=None, bond_extra=0.45):
        """
        Updates bonds (and molecules).

        See Also:
            :func:`~exatomic.two.AtomTwo.compute_bonds`
        """
        self.atom_two.compute_bonds(self.atom['symbol'], mapper=mapper, bond_extra=bond_extra)
        self.compute_molecule()

    def compute_bond_count(self):
        """
        Compute bond counts and attach them to the :class:`~exatomic.atom.Atom` table.
        """
        self.atom['bond_count'] = compute_bond_count(self)

    def compute_molecule(self):
        """Compute the :class:`~exatomic.molecule.Molecule` table."""
        self.molecule = compute_molecule(self)

    def compute_molecule_com(self):
        cx, cy, cz = compute_com(self)
        self.molecule['cx'] = cx
        self.molecule['cy'] = cy
        self.molecule['cz'] = cz

    def compute_atom_count(self):
        """Compute the atom count for each frame."""
        self.frame['atom_count'] = self.atom.grouped().size()

    def _custom_traits(self):
        """
        Build traits depending on multiple dataframes.
        """
        traits = {}
        # Hack for now...
        if hasattr(self, '_atom_two') or len(self)*100 > self.frame['atom_count'].sum():
            mapper = self.atom.get_atom_labels().astype(np.int64)
            traits.update(self.atom_two._bond_traits(mapper))
        return traits

    @classmethod
    def from_small_molecule_data(cls, center=None, ligand=None, distance=None, geometry=None,
                                 offset=None, plane=None, axis=None, domains=None, unit='A'):
        '''
        Build a universe from small molecule data

        See
            exatomic.algorithms.geometry.make_small_molecule
        '''
        return cls(atom=Atom.from_small_molecule_data(center=center, ligand=ligand,
                                                      distance=distance, geometry=geometry,
                                                      offset=offset, plane=plane, axis=axis,
                                                      domains=domains, unit=unit))

    def __len__(self):
        return len(self.frame)


def concat(*universes, name=None, description=None, meta=None):
    """
    Warning:
        This function is not fully featured or tested yet!
    """
    raise NotImplementedError()
    kwargs = {'name': name, 'description': description, 'meta': meta}
    names = []
    for universe in universes:
        for key, data in universe._data().items():
            name = key[1:] if key.startswith('_') else key
            names.append(name)
            if name in kwargs:
                kwargs[name].append(data)
            else:
                kwargs[name] = [data]
    for name in set(names):
        cls = kwargs[name][0].__class__
        if isinstance(kwargs[name][0], Field):
            data = pd.concat(kwargs[name])
            values = [v for field in kwargs[name] for v in field.field_values]
            kwargs[name] = cls(data, field_values=values)
        else:
            kwargs[name] = cls(pd.concat(kwargs[name]))
    return Universe(**kwargs)
