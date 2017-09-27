#<<<<<<< HEAD
#<<<<<<< HEAD
#=======
#>>>>>>> 27d553b3b4e163d985a0bdc43198c525ebdd64c5
## -*- coding: utf-8 -*-
## Copyright (c) 2015-2017, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#The Universe
###############################
#The :class:`~exatomic.container.Universe` is a
#:class:`~exa.core.container.Container` object that stores all information about
#the atomic system under investigation.
#"""
#<<<<<<< HEAD
#from exa import Container
#
#
#class Universe(Container):
#=======
#import six
#import numpy as np
#import pandas as pd
#try:
#    from exa.core.base import DataObject
#    from exa.core.numerical import Field, DataFrame
#    from exa.core.container import Container
#except ImportError:
#    from exa.container import TypedMeta as DataObject
#    from exa.numerical import Field, DataFrame
#    from exa.container import Container
#from exatomic.error import BasisSetNotFoundError
#from exatomic.frame import Frame, compute_frame_from_atom
#from exatomic.atom import Atom, UnitAtom, ProjectedAtom, VisualAtom, Frequency
#from exatomic.two import (AtomTwo, MoleculeTwo, compute_atom_two,
#                          compute_bond_count, compute_molecule_two)
#from exatomic.molecule import (Molecule, compute_molecule, compute_molecule_com,
#                               compute_molecule_count)
#from exatomic.field import AtomicField
#from exatomic.orbital import Orbital, Excitation, MOMatrix, DensityMatrix
#from exatomic.basis import Overlap, BasisSet, BasisSetOrder
#from exatomic.algorithms.orbital import add_molecular_orbitals
#from exatomic.interfaces.cclib import universe_from_cclib
#from exatomic.widget import UniverseWidget
#
#
#class Meta(DataObject):
#>>>>>>> 27d553b3b4e163d985a0bdc43198c525ebdd64c5
#    """
#    A system of atoms and/or molecules being studied.
#    """
#<<<<<<< HEAD
#    pass
#
#
#
#
#
##"""
##The Atomic Universe
###########################
##The :class:`~exatomic.container.Universe` object is a subclass of
##:class:`~exa.container.Container` that stores data coming from computational
##chemistry experiments in a unified and systematic way. Data is organized into
##"frames". A frame is an axis that can represent time (e.g. molecular dynamics
##simulations), step number (e.g. geometry optimization), or an arbitrary index
##(e.g. density functional theory exchange correlation functional).
##"""
###import six
##import numpy as np
##import pandas as pd
##try:
##    from exa.core.base import DataObject
##    from exa.core.numerical import Field
##    from exa.core.container import Container
##except ImportError:
##    from exa.container import TypedMeta as DataObject
##    from exa.numerical import Field
##    from exa.container import Container
###from exatomic.error import BasisSetNotFoundError
##from exatomic.frame import Frame, compute_frame_from_atom
##from exatomic.atom import Atom, UnitAtom, ProjectedAtom, VisualAtom, Frequency
##from exatomic.two import (AtomTwo, MoleculeTwo, compute_atom_two,
##                          compute_bond_count) #, compute_molecule_two)
##from exatomic.molecule import (Molecule, compute_molecule, compute_molecule_com,
##                               compute_molecule_count)
##from exatomic.widget import Universe as UniverseWidget
##from exatomic.widget import TestUniverse as TestUniverseWidget
##from exatomic.field import AtomicField
##from exatomic.orbital import Orbital, Excitation, MOMatrix, DensityMatrix
##from exatomic.basis import Overlap, BasisSet, BasisSetOrder
##from exatomic.algorithms.orbital import add_molecular_orbitals
##from exatomic.interfaces.cclib import universe_from_cclib
##
##
##class Meta(DataObject):
##    """
##    Defines strongly typed attributes of the :class:`~exatomic.universe.Universe`
##    and :class:`~exatomic.editor.AtomicEditor` objects. All "aliases" below are
##    in fact type definitions that get dynamically generated on package load
##    for :class:`~exatomic.container.Universe` and :class:`~exatomic.editor.Editor`.
##    """
##    atom = Atom
##    frame = Frame
##    atom_two = AtomTwo
##    unit_atom = UnitAtom
##    projected_atom = ProjectedAtom
##    visual_atom = VisualAtom
##    frequency = Frequency
##    molecule = Molecule
##    molecule_two = MoleculeTwo
##    field = AtomicField
##    orbital = Orbital
##    overlap = Overlap
##    momatrix = MOMatrix
##    excitation = Excitation
##    density = DensityMatrix
##    basis_set_order = BasisSetOrder
##    basis_set = BasisSet
##
##class Universe(Container, metaclass=Meta):
##    """
##    The atomic container is called a universe because it represents everything
##    known about the atomistic simulation (whether quantum or classical). This
##    includes data such as atomic coordinates, molecular orbital energies, as
##    well as (classical phenomena) such as two body distances, etc.
##
##    Attributes:
##        atom (:class:`~exatomic.atom.Atom`): Atomic coordinates, symbols, forces, etc.
##    """
##    _cardinal = 'frame'
##
##    @classmethod
##    def from_cclib(cls, ccobj):
##        return cls(**universe_from_cclib(ccobj))
##
##    # Note that compute_* function may be called automatically by typed
##    # properties defined in UniverseMeta
##    def compute_frame(self):
##        """
##        Compute a minmal frame table.
##        """
##        self.frame = compute_frame_from_atom(self.atom)
##
##    def compute_unit_atom(self):
##        """Compute minimal image for periodic systems."""
##        self.unit_atom = UnitAtom.from_universe(self)
##
##    def compute_visual_atom(self):
##        """"""
##        self.visual_atom = VisualAtom.from_universe(self)
##        self.compute_molecule_com()
##
##    def compute_atom_two(self, mapper=None, bond_extra=0.45):
##        """
##        Compute interatomic two body properties (e.g. bonds).
##
##        Args:
##            mapper (dict): Custom radii to use when determining bonds
##            bond_extra (float): Extra additive factor to use when determining bonds
##        """
##        if self.frame.is_periodic():
##            atom_two, projected_atom = compute_atom_two(self, mapper, bond_extra)
##            self.atom_two = atom_two
##            self.projected_atom = projected_atom
##        else:
##            self.atom_two = compute_atom_two(self, mapper, bond_extra)
##        self._traits_need_update = True
##
##    def compute_bonds(self, mapper=None, bond_extra=0.45):
##        """
##        Updates bonds (and molecules).
##
##        See Also:
##            :func:`~exatomic.two.AtomTwo.compute_bonds`
##        """
##        self.atom_two.compute_bonds(self.atom['symbol'], mapper=mapper, bond_extra=bond_extra)
##        self.compute_molecule()
##        self._traits_need_update = True
##
##    def compute_bond_count(self):
##        """
##        Compute bond counts and attach them to the :class:`~exatomic.atom.Atom` table.
##        """
##        self.atom['bond_count'] = compute_bond_count(self)
##
##    def compute_molecule(self):
##        """Compute the :class:`~exatomic.molecule.Molecule` table."""
##        self.molecule = compute_molecule(self)
##        self.compute_molecule_count()
##
##    def compute_molecule_com(self):
##        cx, cy, cz = compute_molecule_com(self)
##        self.molecule['cx'] = cx
##        self.molecule['cy'] = cy
##        self.molecule['cz'] = cz
##
##    def compute_atom_count(self):
##        """Compute number of atoms per frame."""
##        self.frame['atom_count'] = self.atom.cardinal_groupby().size()
##
##    def compute_molecule_count(self):
##        """Compute number of molecules per frame."""
##        self.frame['molecule_count'] = compute_molecule_count(self)
##
##    def compute_density(self):
##        """Compute density from momatrix and occupation vector."""
##        if not hasattr(self, 'occupation_vector'):
##            raise Exception('Universe must have momatrix and occupation_vector attributes')
##        self.density = DensityMatrix.from_momatrix(self.momatrix, self.occupation_vector)
##
##    def add_field(self, field):
##        """Adds a field object to the universe."""
##        self._traits_need_update = True
##        if isinstance(field, AtomicField):
##            if not hasattr(self, 'field'):
##                self.field = field
##            else:
##                new_field_values = self.field.field_values + field.field_values
##                newdx = range(len(self.field), len(self.field) + len(field))
##                field.index = newdx
##                new_field = pd.concat([self.field, field])
##                self.field = AtomicField(new_field, field_values=new_field_values)
##        elif isinstance(field, list):
##            if not hasattr(self, 'field'):
##                fields = pd.concat(field)
##                fields.index = range(len(fields))
##                fields_values = [j for i in field for j in i.field_values]
##                self.field = AtomicField(fields, field_values=fields_values)
##            else:
##                new_field_values = self.field.field_values + [j for i in field for j in i.field_values]
##                newdx = range(len(self.field), len(self.field) + sum([len(i.field_values) for i in field]))
##                for i, idx in enumerate(newdx):
##                    field[i].index = [idx]
##                new_field = pd.concat([self.field] + field)
##                self.field = AtomicField(new_field, field_values=new_field_values)
##        else:
##            raise TypeError('field must be an instance of exatomic.field.AtomicField or a list of them')
##        self._traits_need_update = True
##
##    def add_molecular_orbitals(self, field_params=None, mocoefs=None,
##                               vector=None, frame=None):
##        """
##        Adds molecular orbitals to universe. field_params define the numerical
##        field and may be a tuple of (min, max, nsteps) or a series containing
##        all of the columns specified in the exatomic.field.AtomicField table.
##
##        Warning:
##            Removes any existing field attribute of the universe.
##        """
##        for attr in ['momatrix', 'basis_set', 'basis_set_order']:
##            if not hasattr(self, attr):
##                raise AttributeError("universe must have {} attribute.".format(attr))
##        add_molecular_orbitals(self, field_params=field_params,
##                               mocoefs=mocoefs, vector=vector, frame=frame)
##        self._traits_need_update = True
##
##
##    def _custom_traits(self):
##        """
##        Build traits depending on multiple dataframes.
##        """
##        traits = {}
##        # Hack for now...
##        if hasattr(self, '_atom_two') or len(self)*100 > self.frame['atom_count'].sum():
##            mapper = self.atom.get_atom_labels().astype(np.int64)
##            traits.update(self.atom_two._bond_traits(mapper))
##        return traits
##
##    def __len__(self):
##        return len(self.frame)
##
##    def __init__(self, **kwargs):
##        super().__init__(**kwargs)
##        self._traits_need_update = True
##        if hasattr(self, '_atom'):
##            self._widget = UniverseWidget(container=self)
##        else:
##            self._widget = TestUniverseWidget()
##
##    def _repr_html_(self):
##        return self._widget._ipython_display_()
##
##
##
##def concat(name=None, description=None, meta=None, *universes):
##    """
##    Warning:
##        This function is not fully featured or tested yet!
##    """
##    raise NotImplementedError()
##    kwargs = {'name': name, 'description': description, 'meta': meta}
##    names = []
##    for universe in universes:
##        for key, data in universe._data().items():
##            name = key[1:] if key.startswith('_') else key
##            names.append(name)
##            if name in kwargs:
##                kwargs[name].append(data)
##            else:
##                kwargs[name] = [data]
##    for name in set(names):
##        cls = kwargs[name][0].__class__
##        if isinstance(kwargs[name][0], Field):
##            data = pd.concat(kwargs[name])
##            values = [v for field in kwargs[name] for v in field.field_values]
##            kwargs[name] = cls(data, field_values=values)
##        else:
##            kwargs[name] = cls(pd.concat(kwargs[name]))
##    return Universe(**kwargs)
##
##
##def basis_function_contributions(universe, mo, mocoefs='coef',
##                                 tol=0.01, ao=None, frame=0):
##    """
##    Provided a universe with momatrix and basis_set_order attributes,
##    return the major basis function contributions of a particular
##    molecular orbital.
##
##    Args
##        universe (exatomic.container.Universe): a universe
##        mo (int): molecular orbital index
##        mocoefs (str): column of interest in universe.momatrix
##        tol (float): minimum value of coefficient by which to filter
##        frame (int): frame of the universe (default is zero)
##
##    Returns
##        together (pd.DataFrame): a join of momatrix and basis_set_order
##    """
##    small = universe.momatrix.contributions(mo, tol=tol, frame=frame)
##    chis = small['chi'].values
##    coefs = small[mocoefs]
##    coefs.index = chis
##    together = pd.concat([universe.basis_set_order.ix[chis], coefs], axis=1)
##    if ao is None:
##        return together
##    else:
##        raise NotImplementedError("not clever enough for that.")
#=======
#    atom = Atom
#    frame = Frame
#    atom_two = AtomTwo
#    unit_atom = UnitAtom
#    projected_atom = ProjectedAtom
#    visual_atom = VisualAtom
#    frequency = Frequency
#    molecule = Molecule
#    molecule_two = MoleculeTwo
#    field = AtomicField
#    orbital = Orbital
#    overlap = Overlap
#    multipole = DataFrame
#    momatrix = MOMatrix
#    excitation = Excitation
#    density = DensityMatrix
#    contribution = DataFrame
#    basis_set_order = BasisSetOrder
#    basis_set = BasisSet
#
#class Universe(Container, metaclass=Meta):
#    """
#    The atomic container is called a universe because it represents everything
#    known about the atomistic simulation (whether quantum or classical). This
#    includes data such as atomic coordinates, molecular orbital energies, as
#    well as (classical phenomena) such as two body distances, etc.
#
#    Attributes:
#        atom (:class:`~exatomic.atom.Atom`): Atomic coordinates, symbols, forces, etc.
#    """
#    _widget_class = UniverseWidget
#    _cardinal = 'frame'
#
#    @classmethod
#    def from_cclib(cls, ccobj):
#        return cls(**universe_from_cclib(ccobj))
#
#    # Note that compute_* function may be called automatically by typed
#    # properties defined in UniverseMeta
#    def compute_frame(self):
#        """
#        Compute a minmal frame table.
#        """
#        self.frame = compute_frame_from_atom(self.atom)
#
#    def compute_unit_atom(self):
#        """Compute minimal image for periodic systems."""
#        self.unit_atom = UnitAtom.from_universe(self)
#
#    def compute_visual_atom(self):
#        """"""
#        self.visual_atom = VisualAtom.from_universe(self)
#        self.compute_molecule_com()
#
#    def compute_atom_two(self, mapper=None, bond_extra=0.45):
#        """
#        Compute interatomic two body properties (e.g. bonds).
#
#        Args:
#            mapper (dict): Custom radii to use when determining bonds
#            bond_extra (float): Extra additive factor to use when determining bonds
#        """
#        if self.frame.is_periodic():
#            atom_two, projected_atom = compute_atom_two(self, mapper, bond_extra)
#            self.atom_two = atom_two
#            self.projected_atom = projected_atom
#        else:
#            self.atom_two = compute_atom_two(self, mapper, bond_extra)
#        self._traits_need_update = True
#
#    def compute_bonds(self, mapper=None, bond_extra=0.45):
#        """
#        Updates bonds (and molecules).
#
#        See Also:
#            :func:`~exatomic.two.AtomTwo.compute_bonds`
#        """
#        self.atom_two.compute_bonds(self.atom['symbol'], mapper=mapper, bond_extra=bond_extra)
#        self.compute_molecule()
#        self._traits_need_update = True
#
#    def compute_bond_count(self):
#        """
#        Compute bond counts and attach them to the :class:`~exatomic.atom.Atom` table.
#        """
#        self.atom['bond_count'] = compute_bond_count(self)
#
#    def compute_molecule(self):
#        """Compute the :class:`~exatomic.molecule.Molecule` table."""
#        self.molecule = compute_molecule(self)
#        self.compute_molecule_count()
#
#    def compute_molecule_com(self):
#        cx, cy, cz = compute_molecule_com(self)
#        self.molecule['cx'] = cx
#        self.molecule['cy'] = cy
#        self.molecule['cz'] = cz
#
#    def compute_atom_count(self):
#        """Compute number of atoms per frame."""
#        self.frame['atom_count'] = self.atom.cardinal_groupby().size()
#
#    def compute_molecule_count(self):
#        """Compute number of molecules per frame."""
#        self.frame['molecule_count'] = compute_molecule_count(self)
#
#    def compute_density(self):
#        """Compute density from momatrix and occupation vector."""
#        if not hasattr(self, 'occupation_vector'):
#            raise Exception('Universe must have momatrix and occupation_vector attributes')
#        self.density = DensityMatrix.from_momatrix(self.momatrix, self.occupation_vector)
#
#    def add_field(self, field):
#        """Adds a field object to the universe."""
#        self._traits_need_update = True
#        if isinstance(field, AtomicField):
#            if not hasattr(self, 'field'):
#                self.field = field
#            else:
#                new_field_values = self.field.field_values + field.field_values
#                newdx = range(len(self.field), len(self.field) + len(field))
#                field.index = newdx
#                new_field = pd.concat([self.field, field])
#                self.field = AtomicField(new_field, field_values=new_field_values)
#        elif isinstance(field, list):
#            if not hasattr(self, 'field'):
#                fields = pd.concat(field)
#                fields.index = range(len(fields))
#                fields_values = [j for i in field for j in i.field_values]
#                self.field = AtomicField(fields, field_values=fields_values)
#            else:
#                new_field_values = self.field.field_values + [j for i in field for j in i.field_values]
#                newdx = range(len(self.field), len(self.field) + sum([len(i.field_values) for i in field]))
#                for i, idx in enumerate(newdx):
#                    field[i].index = [idx]
#                new_field = pd.concat([self.field] + field)
#                self.field = AtomicField(new_field, field_values=new_field_values)
#        else:
#            raise TypeError('field must be an instance of exatomic.field.AtomicField or a list of them')
#        self._traits_need_update = True
#
#    def add_molecular_orbitals(self, field_params=None, mocoefs=None,
#                               vector=None, frame=None):
#        """
#        Adds molecular orbitals to universe. field_params define the numerical
#        field and may be a tuple of (min, max, nsteps) or a series containing
#        all of the columns specified in the exatomic.field.AtomicField table.
#
#        Warning:
#            Removes any existing field attribute of the universe.
#        """
#        for attr in ['momatrix', 'basis_set', 'basis_set_order']:
#            if not hasattr(self, attr):
#                raise AttributeError("universe must have {} attribute.".format(attr))
#        add_molecular_orbitals(self, field_params=field_params,
#                               mocoefs=mocoefs, vector=vector, frame=frame)
#        self._traits_need_update = True
#
#
#    def _custom_traits(self):
#        """
#        Build traits depending on multiple dataframes.
#        """
#        traits = {}
#        # Hack for now...
#        if hasattr(self, '_atom_two') or len(self)*100 > self.frame['atom_count'].sum():
#            mapper = self.atom.get_atom_labels().astype(np.int64)
#            traits.update(self.atom_two._bond_traits(mapper))
#        return traits
#
#    def __len__(self):
#        return len(self.frame)
#
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)
#        self._widget = self._widget_class(self)
#
#
#
#def concat(name=None, description=None, meta=None, *universes):
#    """
#    Warning:
#        This function is not fully featured or tested yet!
#    """
#    raise NotImplementedError()
#    kwargs = {'name': name, 'description': description, 'meta': meta}
#    names = []
#    for universe in universes:
#        for key, data in universe._data().items():
#            name = key[1:] if key.startswith('_') else key
#            names.append(name)
#            if name in kwargs:
#                kwargs[name].append(data)
#            else:
#                kwargs[name] = [data]
#    for name in set(names):
#        cls = kwargs[name][0].__class__
#        if isinstance(kwargs[name][0], Field):
#            data = pd.concat(kwargs[name])
#            values = [v for field in kwargs[name] for v in field.field_values]
#            kwargs[name] = cls(data, field_values=values)
#        else:
#            kwargs[name] = cls(pd.concat(kwargs[name]))
#    return Universe(**kwargs)
#
#
#def basis_function_contributions(universe, mo, mocoefs='coef',
#                                 tol=0.01, ao=None, frame=0):
#    """
#    Provided a universe with momatrix and basis_set_order attributes,
#    return the major basis function contributions of a particular
#    molecular orbital.
#
#    Args
#        universe (exatomic.container.Universe): a universe
#        mo (int): molecular orbital index
#        mocoefs (str): column of interest in universe.momatrix
#        tol (float): minimum value of coefficient by which to filter
#        frame (int): frame of the universe (default is zero)
#
#    Returns
#        together (pd.DataFrame): a join of momatrix and basis_set_order
#    """
#    small = universe.momatrix.contributions(mo, tol=tol, mocoefs=mocoefs, frame=frame)
#    chis = small['chi'].values
#    coefs = small[mocoefs]
#    coefs.index = chis
#    together = pd.concat([universe.basis_set_order.ix[chis], coefs], axis=1)
#    if ao is None:
#        return together
#    else:
#        raise NotImplementedError("not clever enough for that.")
#<<<<<<< HEAD
#>>>>>>> 811f6aaae1e1aef968c27a34842d5ad9e7267217
#=======
## -*- coding: utf-8 -*-
## Copyright (c) 2015-2016, Exa Analytics Development Team
## Distributed under the terms of the Apache License 2.0
#"""
#The Atomic Universe
##########################
#The :class:`~exatomic.container.Universe` object is a subclass of
#:class:`~exa.container.Container` that stores data coming from computational
#chemistry experiments in a unified and systematic way. Data is organized into
#"frames". A frame is an axis that can represent time (e.g. molecular dynamics
#simulations), step number (e.g. geometry optimization), or an arbitrary index
#(e.g. density functional theory exchange correlation functional).
#"""
#import six
#import numpy as np
#import pandas as pd
#try:
#    from exa.core.base import DataObject
#    from exa.core.numerical import Field, DataFrame
#    from exa.core.container import Container
#except ImportError:
#    from exa.container import TypedMeta as DataObject
#    from exa.numerical import Field, DataFrame
#    from exa.container import Container
#from exatomic.error import BasisSetNotFoundError
#from exatomic.frame import Frame, compute_frame_from_atom
#from exatomic.atom import Atom, UnitAtom, ProjectedAtom, VisualAtom, Frequency
#from exatomic.two import (AtomTwo, MoleculeTwo, compute_atom_two,
#                          compute_bond_count, compute_molecule_two)
#from exatomic.molecule import (Molecule, compute_molecule, compute_molecule_com,
#                               compute_molecule_count)
##from exatomic.widget import TestUniverse, UniverseWidget
#from exatomic.field import AtomicField
#from exatomic.orbital import Orbital, Excitation, MOMatrix, DensityMatrix
#from exatomic.basis import Overlap, BasisSet, BasisSetOrder
#from exatomic.algorithms.orbital import add_molecular_orbitals
#from exatomic.interfaces.cclib import universe_from_cclib
#
#
#class Meta(DataObject):
#    """
#    Defines strongly typed attributes of the :class:`~exatomic.universe.Universe`
#    and :class:`~exatomic.editor.AtomicEditor` objects. All "aliases" below are
#    in fact type definitions that get dynamically generated on package load
#    for :class:`~exatomic.container.Universe` and :class:`~exatomic.editor.Editor`.
#    """
#    atom = Atom
#    frame = Frame
#    atom_two = AtomTwo
#    unit_atom = UnitAtom
#    projected_atom = ProjectedAtom
#    visual_atom = VisualAtom
#    frequency = Frequency
#    molecule = Molecule
#    molecule_two = MoleculeTwo
#    field = AtomicField
#    orbital = Orbital
#    overlap = Overlap
#    multipole = DataFrame
#    momatrix = MOMatrix
#    excitation = Excitation
#    density = DensityMatrix
#    contribution = DataFrame
#    basis_set_order = BasisSetOrder
#    basis_set = BasisSet
#
#class Universe(Container, metaclass=Meta):
#    """
#    The atomic container is called a universe because it represents everything
#    known about the atomistic simulation (whether quantum or classical). This
#    includes data such as atomic coordinates, molecular orbital energies, as
#    well as (classical phenomena) such as two body distances, etc.
#
#    Attributes:
#        atom (:class:`~exatomic.atom.Atom`): Atomic coordinates, symbols, forces, etc.
#    """
#    _cardinal = 'frame'
#
#    @classmethod
#    def from_cclib(cls, ccobj):
#        return cls(**universe_from_cclib(ccobj))
#
#    # Note that compute_* function may be called automatically by typed
#    # properties defined in UniverseMeta
#    def compute_frame(self):
#        """
#        Compute a minmal frame table.
#        """
#        self.frame = compute_frame_from_atom(self.atom)
#
#    def compute_unit_atom(self):
#        """Compute minimal image for periodic systems."""
#        self.unit_atom = UnitAtom.from_universe(self)
#
#    def compute_visual_atom(self):
#        """"""
#        self.visual_atom = VisualAtom.from_universe(self)
#        self.compute_molecule_com()
#
#    def compute_atom_two(self, mapper=None, bond_extra=0.45):
#        """
#        Compute interatomic two body properties (e.g. bonds).
#
#        Args:
#            mapper (dict): Custom radii to use when determining bonds
#            bond_extra (float): Extra additive factor to use when determining bonds
#        """
#        if len(self.atom.last_frame.index) > 200: return
#        if self.frame.is_periodic():
#            atom_two, projected_atom = compute_atom_two(self, mapper, bond_extra)
#            self.atom_two = atom_two
#            self.projected_atom = projected_atom
#        else:
#            self.atom_two = compute_atom_two(self, mapper, bond_extra)
#        self._traits_need_update = True
#
#    def compute_bonds(self, mapper=None, bond_extra=0.45):
#        """
#        Updates bonds (and molecules).
#
#        See Also:
#            :func:`~exatomic.two.AtomTwo.compute_bonds`
#        """
#        self.atom_two.compute_bonds(self.atom['symbol'], mapper=mapper, bond_extra=bond_extra)
#        self.compute_molecule()
#        self._traits_need_update = True
#
#    def compute_bond_count(self):
#        """
#        Compute bond counts and attach them to the :class:`~exatomic.atom.Atom` table.
#        """
#        self.atom['bond_count'] = compute_bond_count(self)
#
#    def compute_molecule(self):
#        """Compute the :class:`~exatomic.molecule.Molecule` table."""
#        self.molecule = compute_molecule(self)
#        self.compute_molecule_count()
#
#    def compute_molecule_com(self):
#        cx, cy, cz = compute_molecule_com(self)
#        self.molecule['cx'] = cx
#        self.molecule['cy'] = cy
#        self.molecule['cz'] = cz
#
#    def compute_atom_count(self):
#        """Compute number of atoms per frame."""
#        self.frame['atom_count'] = self.atom.cardinal_groupby().size()
#
#    def compute_molecule_count(self):
#        """Compute number of molecules per frame."""
#        self.frame['molecule_count'] = compute_molecule_count(self)
#
#    def compute_density(self):
#        """Compute density from momatrix and occupation vector."""
#        if not hasattr(self, 'occupation_vector'):
#            raise Exception('Universe must have momatrix and occupation_vector attributes')
#        self.density = DensityMatrix.from_momatrix(self.momatrix, self.occupation_vector)
#
#    def add_field(self, field):
#        """Adds a field object to the universe."""
#        self._traits_need_update = True
#        if isinstance(field, AtomicField):
#            if not hasattr(self, 'field'):
#                self.field = field
#            else:
#                new_field_values = self.field.field_values + field.field_values
#                newdx = range(len(self.field), len(self.field) + len(field))
#                field.index = newdx
#                new_field = pd.concat([self.field, field])
#                self.field = AtomicField(new_field, field_values=new_field_values)
#        elif isinstance(field, list):
#            if not hasattr(self, 'field'):
#                fields = pd.concat(field)
#                fields.index = range(len(fields))
#                fields_values = [j for i in field for j in i.field_values]
#                self.field = AtomicField(fields, field_values=fields_values)
#            else:
#                new_field_values = self.field.field_values + [j for i in field for j in i.field_values]
#                newdx = range(len(self.field), len(self.field) + sum([len(i.field_values) for i in field]))
#                for i, idx in enumerate(newdx):
#                    field[i].index = [idx]
#                new_field = pd.concat([self.field] + field)
#                self.field = AtomicField(new_field, field_values=new_field_values)
#        else:
#            raise TypeError('field must be an instance of exatomic.field.AtomicField or a list of them')
#        self._traits_need_update = True
#
#    def add_molecular_orbitals(self, field_params=None, mocoefs=None,
#                               vector=None, frame=None):
#        """
#        Adds molecular orbitals to universe. field_params define the numerical
#        field and may be a tuple of (min, max, nsteps) or a series containing
#        all of the columns specified in the exatomic.field.AtomicField table.
#
#        Warning:
#            Removes any existing field attribute of the universe.
#        """
#        for attr in ['momatrix', 'basis_set', 'basis_set_order']:
#            if not hasattr(self, attr):
#                raise AttributeError("universe must have {} attribute.".format(attr))
#        add_molecular_orbitals(self, field_params=field_params,
#                               mocoefs=mocoefs, vector=vector, frame=frame)
#        self._traits_need_update = True
#
#
#    def get_atom_labels(self):
#        return self.atom.get_atom_labels().astype(np.int64)
#
##    def _custom_traits(self):
##        """
##        Build traits depending on multiple dataframes.
##        """
##        traits = {}
##        # Hack for now...
##        if hasattr(self, '_atom_two') or len(self)*100 > self.frame['atom_count'].sum():
##            mapper = self.atom.get_atom_labels().astype(np.int64)
##            traits.update(self.atom_two._bond_traits(mapper))
##        return traits
#
#    def __len__(self):
#        return len(self.frame)
#
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)
#        self._traits_need_update = True
#        #if hasattr(self, '_atom'):
#        #    self._widget = UniverseWidget(self)
#        #else:
#        #    self._widget = TestUniverse()
#
#    def _repr_html_(self):
#        return self._widget._ipython_display_()
#
#
#
#def concat(name=None, description=None, meta=None, *universes):
#    """
#    Warning:
#        This function is not fully featured or tested yet!
#    """
#    raise NotImplementedError()
#    kwargs = {'name': name, 'description': description, 'meta': meta}
#    names = []
#    for universe in universes:
#        for key, data in universe._data().items():
#            name = key[1:] if key.startswith('_') else key
#            names.append(name)
#            if name in kwargs:
#                kwargs[name].append(data)
#            else:
#                kwargs[name] = [data]
#    for name in set(names):
#        cls = kwargs[name][0].__class__
#        if isinstance(kwargs[name][0], Field):
#            data = pd.concat(kwargs[name])
#            values = [v for field in kwargs[name] for v in field.field_values]
#            kwargs[name] = cls(data, field_values=values)
#        else:
#            kwargs[name] = cls(pd.concat(kwargs[name]))
#    return Universe(**kwargs)
#
#
#def basis_function_contributions(universe, mo, mocoefs='coef',
#                                 tol=0.01, ao=None, frame=0):
#    """
#    Provided a universe with momatrix and basis_set_order attributes,
#    return the major basis function contributions of a particular
#    molecular orbital.
#
#    Args
#        universe (exatomic.container.Universe): a universe
#        mo (int): molecular orbital index
#        mocoefs (str): column of interest in universe.momatrix
#        tol (float): minimum value of coefficient by which to filter
#        frame (int): frame of the universe (default is zero)
#
#    Returns
#        together (pd.DataFrame): a join of momatrix and basis_set_order
#    """
#    small = universe.momatrix.contributions(mo, tol=tol, frame=frame)
#    chis = small['chi'].values
#    coefs = small[mocoefs]
#    coefs.index = chis
#    together = pd.concat([universe.basis_set_order.ix[chis], coefs], axis=1)
#    if ao is None:
#        return together
#    else:
#        raise NotImplementedError("not clever enough for that.")
#>>>>>>> 454ebaa9677a776a535abb28f528efefabda52c5
#=======
#
#>>>>>>> 27d553b3b4e163d985a0bdc43198c525ebdd64c5
