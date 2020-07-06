# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
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
from exatomic.algorithms.basis import BasisFunctions, compute_uncontracted_basis_set_order
from .tensor import Tensor

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
    cart_momatrix = MOMatrix
    sphr_momatrix = MOMatrix
    excitation = Excitation
    overlap = Overlap
    density = DensityMatrix
    basis_set_order = BasisSetOrder
    cart_basis_set_order = BasisSetOrder
    sphr_basis_set_order = BasisSetOrder
    uncontracted_basis_set_order = BasisSetOrder
    basis_set = BasisSet
    basis_dims = dict
    basis_functions = BasisFunctions
    contribution = DataFrame
    multipole = DataFrame
    tensor = Tensor


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
        frequency (:class:`~exatomic.core.atom.Frequency`): Vibrational modes and atom displacements
        excitation (:class:`~exatomic.core.orbital.Excitation`): Electronic excitation information
        basis_set (:class:`~exatomic.core.basis.BasisSet`): Basis set specification
        overlap (:class:`~exatomic.core.basis.Overlap`): The overlap matrix
        basis_functions (:class:`~exatomic.algorithms.basis.BasisFunctions`): Basis function evaluation
        field (:class:`~exatomic.core.field.AtomicField`): Scalar fields (MOs, densities, etc.)
    """
    _cardinal = "frame"
    _getter_prefix = "compute"

    @property
    def current_momatrix(self):
        if self.meta['spherical']:
            try: return self.sphr_momatrix
            except AttributeError: return self.momatrix
        try: return self.cart_momatrix
        except AttributeError: return self.momatrix

    @property
    def current_basis_set_order(self):
        if 'uncontracted' in self.meta:
            return self.uncontracted_basis_set_order
        if self.meta['spherical']:
            try: return self.sphr_basis_set_order
            except AttributeError: return self.basis_set_order
        try: return self.cart_basis_set_order
        except AttributeError: return self.basis_set_order

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
        """Compute a minmal frame table."""
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
        bset = self.basis_set.copy()
        bset['set'] = bset['set'].astype(np.int64)
        mapr = self.atom.set.map
        self.basis_dims = {
            'npc': mapr(bset.primitives(False).groupby('set').sum()).astype(int).sum(),
            'nps': mapr(bset.primitives(True).groupby('set').sum()).astype(int).sum(),
            'ncc': mapr(bset.functions(False).groupby('set').sum()).astype(int).sum(),
            'ncs': mapr(bset.functions(True).groupby('set').sum()).astype(int).sum(),
            'sets': bset.functions_by_shell()}

    def compute_basis_functions(self, **kwargs):
        if self.meta['program'] in ['nwchem']:
            self.basis_functions = BasisFunctions(self, cartp=False)
        else:
            self.basis_functions = BasisFunctions(self)

    def compute_uncontracted_basis_set_order(self):
        """Compute an uncontracted basis set order."""
        self.uncontracted_basis_set_order = compute_uncontracted_basis_set_order(self)

    def enumerate_shells(self, frame=0):
        """Extract minimal information from the universe to be used in
        numba-compiled numerical procedures.

        .. code-block:: python

            pointers, atoms, shells = uni.enumerate_shells()

        Args:
            frame (int): state of the universe (default 0)
        """
        atom = self.atom.groupby('frame').get_group(frame)
        if self.meta['program'] not in ['molcas', 'adf', 'nwchem', 'gaussian']:
            print('Warning: Check spherical shell parameter for {} '
                  'molecular orbital generation'.format(self.meta['program']))
        shls = self.basis_set.shells(self.meta['program'],
                                     self.meta['spherical'],
                                     self.meta['gaussian'])
        grps = shls.groupby('set')
        # Pointers into (xyzs, shls) arrays
        ptrs = np.array([(c, idx) for c, seht in enumerate(atom.set)
                                  for idx in grps.get_group(seht).index])
        return ptrs, atom[['x', 'y', 'z']].values, shls[0].values

    def add_field(self, field):
        """Adds a field object to the universe.

        .. code-block:: python

            # Assuming field[n] is of type AtomicField
            uni.add_field(field)
            uni.add_field([field1, field2])

        Args:
            field (iter, :class:`exatomic.core.field.AtomicField`): field(s) to add

        Warning:
            Adding a large number of (high resolution) fields may impact performance.
        """
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
                               vector=None, frame=0, replace=False,
                               inplace=True, verbose=True, irrep=None):
        """Add molecular orbitals to universe.

        .. code-block:: python

            uni.add_molecular_orbitals()                  # Default around (HOMO-5, LUMO+7)
            uni.add_molecular_orbitals(vector=range(5))   # Specifies the first 5 MOs
            uni.add_molecular_orbitals(                   # Higher resolution fields
                field_params={'rmin': -10,                # smallest value in 'x', 'y', 'z'
                              'rmax': 10,                 # largest value in 'x', 'y', 'z'
                              'nr': 100})                 # number of points between rmin and rmax
            uni.field                                     # The field parameters
            uni.field.field_values                        # The generated scalar fields

        Args:
            field_params (dict, pd.Series): see :func:`exatomic.algorithms.orbital_util.make_fps`
            mocoefs (str): column in :class:`~exatomic.core.orbital.MOMatrix`
            vector (iter): indices of orbitals to evaluate (0-based)
            frame (int): frame of atomic positions for the orbitals
            replace (bool): remove previous fields (default False)
            inplace (bool): add directly to uni or return :class:`~exatomic.core.field.AtomicField` (default True)
            verbose (bool): print timing statistics (default True)
            irrep (int): irreducible representation

        Warning:
            Default behavior just continually adds fields to the universe.  This can
            affect performance if adding many fields. `replace` modifies this behavior.

        Warning:
            Specifying very high resolution field parameters, e.g. 'nr' > 100
            may slow things down and/or crash the kernel.  Use with caution.
        """
        if not hasattr(self, 'momatrix'):
            raise AttributeError('uni must have momatrix attribute.')
        if not hasattr(self, 'basis_set'):
            raise AttributeError('uni must have basis_set attribute.')
        return add_molecular_orbitals(self, field_params=field_params,
                                      mocoefs=mocoefs, vector=vector,
                                      frame=frame, replace=replace,
                                      inplace=inplace, verbose=verbose,
                                      irrep=irrep)

    def write_cube(self, file_name='output', field_number=0):
        """
        Write to a file in cube format for a single 3D scalar field in universe object.

        .. code-block:: python

            uni.add_molecular_orbitals()                  # Default around (HOMO-5, LUMO+7)
            uni.write_cube('cubefile', 0)                    # write to cubefile.cube for HOMO-5

        Args:
            file_name (str): name of the output file without file extension
            field_number (int): number of the single field starting with 0

        Returns:
            None
        """
        import os
        from exatomic.interfaces.cube import Cube
        if os.path.isfile(file_name+'.cube'):
            raise FileExistsError('File '+file_name+'.cube '+'exists.')
        cube_edi = Cube.from_universe(self,field_number)
        cube_edi.write(file_name+'.cube')

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

    .. code-block:: python

        # display the 16th orbital coefficients > abs(0.15)
        basis_function_contributions(uni, 15, tol=0.15) # 0-based indexing!

    Args:
        universe (class:`exatomic.core.universe.Universe`): a universe
        mo (int): molecular orbital index
        mocoefs (str): column of interest in universe.momatrix
        tol (float): minimum value of coefficient by which to filter
        frame (int): frame of the universe (default is zero)

    Returns:
        joined (pd.DataFrame): a join of momatrix and basis_set_order
    """
    small = universe.momatrix.contributions(mo, tol=tol, mocoefs=mocoefs, frame=frame)
    chis = small['chi'].values
    coefs = small[mocoefs]
    coefs.index = chis
    joined = pd.concat([universe.basis_set_order.ix[chis], coefs], axis=1)
    if ao is None:
        return joined
    else:
        raise NotImplementedError("not clever enough for that.")
