# -*- coding: utf-8 -*-
'''
Atom DataFrame
==========================
'''
from scipy.spatial import cKDTree
from atomic import _np as np
from atomic import _pd as pd
from exa import DataFrame, Config
from exa.errors import MissingColumns
from atomic.errors import PeriodicError
from atomic.twobody import TwoBody
if Config.numba:
    from exa.jitted.iteration import periodic_supercell, repeat_int
    from exa.jitted.broadcasting import mod
else:
    from exa.algorithms.iteration import periodic_supercell
    import numpy.repeat as repeat_int
    import numpy.mod as mod


class Atom(DataFrame):
    '''
    Absolute positions of atoms and their symbol.

    Required indexes: frame, atom

    Required columns: symbol, x, y, z
    '''
    __indices__ = ['frame', 'atom']
    __columns__ = ['symbol', 'x', 'y', 'z']


class VisualAtom(DataFrame):
    '''
    Special positions for atoms used to generate coherent animations.
    '''
    __indices__ = ['frame', 'vis_atom']
    __columns__ = ['x', 'y', 'z', 'atom']


class PrimitiveAtom(DataFrame):
    '''
    Primitive (or in unit cell) coordinates.
    '''
    __indices__ = ['frame', 'prim_atom']
    __columns__ = ['x', 'y', 'z', 'atom']


class SuperCellAtom(DataFrame):
    '''
    A 3 x 3 x 3 super cell generate using the primitive cell positions.

    See Also:
        :class:`~atomic.atom.PrimitiveAtom`
    '''
    __indices__ = ['frame', 'sc_atom']
    __columns__ = ['x', 'y', 'z', 'atom']



rfc = ['rx', 'ry', 'rz', 'ox', 'oy', 'oz']    # Required columns in the Frame table for periodic calcs


def compute_primitive(universe):
    '''
    Compute the primitive cell positions for each frame in the universe.

    Args:
        universe (:class:`~atomic.universe.Universe`): Universe containing the atoms table

    Returns:
        prim_atoms (:class:`~atomic.atom.PrimitiveAtom`): Primitive positions table
    '''
    if 'periodic' not in universe.frames.columns:
        raise PeriodicError()
    elif any(universe.frames['periodic'] == True):
        if any((col not in universe.frames.columns for col in rfc)):
            raise MissingColumns()


def _compute_primitive_frame(xs, ys, zs, rx, ry, rz, ox, oy, oz):
    '''
    For a given frame (see :class:`~atomic.atom.Atom`), compute the primitive
    cell.

    Args:
        xs (:class:`~numpy.ndarray`): Array of x positions
        ys (:class:`~numpy.ndarray`): Array of y positions
        zs (:class:`~numpy.ndarray`): Array of z positions
        rx (float): Cell magnitude in x
        ry (float): Cell magnitude in y
        rz (float): Cell magnitude in z
        ox (float): Cell origin in x
        oy (float): Cell origin in y
        oz (float): Cell origin in z

    Returns
    '''



#
#def compute_twobody(universe, k=None, bond_extra=0.45, dmax=13.0, dmin=0.3):
#    '''
#    Compute two body information given a :class:`~atomic.atom.Atom` dataframe.
#
#    For non-periodic systems the only required argument is the table of atom
#    positions and symbols. For periodic systems, at a minimum, the atoms and
#    frames dataframes must be provided.
#
#    Bonds are computed semi-empirically and exist if:
#
#    .. math::
#
#        distance(A, B) < covalent\_radius(A) + covalent\_radius(B) + bond\_extra
#
#    Args:
#        atoms (:class:`~atomic.atom.Atom`): Table of nuclear positions and symbols
#        frames (:class:`~pandas.DataFrame`): DataFrame of periodic cell dimensions (or False if )
#        k (int): Number of distances (per atom) to compute
#        bond_extra (float): Extra distance to include when determining bonds (see above)
#        dmax (float): Max distance of interest (larger distances are ignored)
#        dmin (float): Min distance of interest (smaller distances are ignored)
#
#    Returns:
#        df (:class:`~atomic.twobody.TwoBody`): Two body property table
#    '''
#    if 'periodic' in universe.frames.columns:    # Figure out what type of two body properties to compute
#        if any(universe.frames['periodic'] == True):
#            req = ['xr', 'yr', 'zr']
#            if any((mag not in universe.frames.columns for mag in req)): # Check the requirements are met
#                raise ColumnError(req, universe.frames)
#        return _compute_periodic_twobody(universe, k, bond_extra, dmax)
#    else:
#        raise NotImplementedError()
#
#
#
#def compute_pbc_atom(universe):
#    '''
#    Create a periodic super cell from original positions.
#
#    Args:
#        universe (:class:`~atomic.universe.Universe`): Atomic universe
#
#    Returns:
#        pbcatoms (:class:`~atomic.atom.PBCAtom`): Periodic super cell
#    '''
#    # Perform checks before computing
#    raise NotImplementedError()
#
#
#def _compute_pbc_atoms(xyzdf, xrs, yrs, zrs, ois, ojs, oks):
#    '''
#    Args:
#        xyz (:class:`~atomic.atom.Atom`): Dataframe containing only x, y, z values
#        xrs (:class:`~pandas.Series`): Unit cell magnitude in x
#        yrs (:class:`~pandas.Series`): Unit cell magnitude in y
#        zrs (:class:`~pandas.Series`): Unit cell magnitude in z
#        ois (:class:`~pandas.Series`): Unit cell origin in x
#        ojs (:class:`~pandas.Series`): Unit cell origin in y
#        oks (:class:`~pandas.Series`): Unit cell origin in z
#
#    Returns:
#        pbc_atoms (:class:`~atomic.atom.PBCAtom`): Periodic atom positions dataframe
#    '''
#    groups = xyzdf.groupby(level='frame')
#    pbc_xyz_list = []          # These lists contain per frame data to be
#    pbc_atom_list = []         # concatenated. Performing operations on
#    pbc_index_list = []        # pandas groups in this manner is more
#    pbc_frame_list = []        # efficient than attempting to reconstruct
#    for fdx, xyz in groups:    # a DataFrame or Series on the fly.
#        xr = xrs[fdx]
#        yr = yrs[fdx]
#        zr = zrs[fdx]          # Build the original atom index for pbc_xyz
#        atom = np.tile(xyz.index.get_level_values('atom'), 27)
#        n = len(atom)
#        unit_xyz = np.mod(xyz, [xr, yr, zr]) - []
#        pbc_xyz_list.append(periodic_supercell(unit_xyz.values, xr, yr, zr))
#        pbc_atom_list.append(atom)
#        pbc_frame_list.append(repeat_int([fdx], n))
#        pbc_index_list.append(range(n))
#    df = pd.DataFrame(np.concatenate(pbc_xyz_list), columns=['x', 'y', 'z'])
#    df['pbc_atom'] = np.concatenate(pbc_index_list)
#    df['atom'] = np.concatenate(pbc_atom_list)
#    df['frame'] = np.concatenate(pbc_frame_list)
#    df.set_index(['frame', 'pbc_atom'], inplace=True)
#    return PBCAtom(df)
#
#
#def _compute_periodic_twobody(universe, k, bond_extra, dmax):
#    '''
#    Compute the periodic two body properties for a given set of atoms and frames.
#
#    The algorithm first iterates over each frame in the universe and computes
#    the data needed for the dataframes to be built, appending to a storage
#    container. After completition of this loop, the dataframe objects are
#    compiled and processed.
#    '''
#    groups = universe.atoms[['x', 'y', 'z']].groupby(level='frame')
#    pbc_xyz_list = []             # XYZ data for PBCAtom
#    pbc_atom_frame = []           # Original frame index for PBCAtom
#    pbc_atom_indices = []         # Original (expanded) atom indices (for PBCAtom)
#    pbc_twobody_distances = []    # Two body distances
#    pbc_twobody_atom1 = []        # Index corresponds to the (future) pbc_xyz df
#    pbc_twobody_atom2 = []        # Ditto
#    pbc_twobody_frame = []        # Original frame index for PBCTwoBody
#    for i, (fdx, xyz) in enumerate(groups):
#        frame = universe.frames.loc[fdx]
#        nat = frame.atom_count
#        k = nat if k is None else k
#        xr = frame.xr
#        yr = frame.yr
#        zr = frame.zr                                                   # Relative (relevant) timings
#        atom_index = np.tile(xyz.index.get_level_values('atom'), 27)    # are given in (...)
#        frame_index = repeat_int([fdx], len(atom_index))
#        unit_xyz = np.mod(xyz, [xr, yr, zr])                            # Compute unit [cell] xyz  (1.0x)
#        pbc_xyz = periodic_supercell(unit_xyz.values, xr, yr, zr)       # Compute pbc supercell (2.5x)
#        distances, indices = cKDTree(pbc_xyz).query(unit_xyz, k=k, distance_upper_bound=dmax)    # Compute distances (10x)
#        pbc_atom1 = repeat_int(indices[:, 0], k)
#        pbc_atom2 = indices.ravel()
#        pbc_frame = repeat_int([fdx], len(pbc_atom2))
#        pbc_xyz_list.append(pbc_xyz)                       # Now begin appending
#        pbc_atom_indices.append(atom_index)                # data to be concatenated
#        pbc_atom_frame.append(frame_index)                 # later.
#        pbc_twobody_distances.append(distances.ravel())
#        pbc_twobody_atom1.append(pbc_atom1)
#        pbc_twobody_atom2.append(pbc_atom2)
#        pbc_twobody_frame.append(pbc_frame)
#
#    # Begin building DataFrames
#    pbc_xyz = pd.DataFrame(np.concatenate(pbc_xyz_list), columns=['x', 'y', 'z'])
#    return (pbc_xyz_list, pbc_atom_indices, pbc_atom_frame, pbc_twobody_distances, pbc_twobody_atom1, pbc_twobody_atom2, pbc_twobody_frame)
#
