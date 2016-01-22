# -*- coding: utf-8 -*-
'''
Two Body Properties DataFrame
===============================
Two body properties are interatomic distances.
'''
import gc
from itertools import combinations
from scipy.spatial import cKDTree
from exa import _np as np
from exa import _pd as pd
from exa import DataFrame, Config
if Config.numba:
    from exa.jitted.iteration import periodic_supercell, repeat_i8, repeat_i8_array, tile_i8
    from exa.jitted.iteration import pdist2d as pdist
else:
    from exa.algorithms.iteration import periodic_supercell
    import numpy.repeat as repeat_i8
    import numpy.tile as tile_i8
    from scipy.spatial.distance import pdist
from atomic import Isotope
from atomic.atom import SuperAtom
from atomic.tools import check


bond_extra = 0.45
dmin = 0.3
dmax = 12.3


class TwoBody(DataFrame):
    '''
    '''
    __indices__ = ['frame', 'index']
    __columns__ = ['atom1', 'atom2', 'symbols', 'distance']


class PeriodicTwoBody(DataFrame):
    '''
    Two body properties corresponding to the super cell atoms dataframe.

    See Also:
        :class:`~atomic.atom.SuperAtom`
    '''
    __indices__ = ['frame', 'index']
    __columns__ = ['super_atom1', 'super_atom2', 'symbols', 'distance']


def compute_twobody(universe, k=None, bond_extra=bond_extra, dmax=dmax, dmin=dmin):
    '''
    Compute two body information given a universe.

    For non-periodic systems the only required argument is the table of atom
    positions and symbols. For periodic systems, at a minimum, the atoms and
    frames dataframes must be provided.

    Bonds are computed semi-empirically and exist if:

    .. math::

        distance(A, B) < covalent\_radius(A) + covalent\_radius(B) + bond\_extra

    Args:
        atoms (:class:`~atomic.atom.Atom`): Table of nuclear positions and symbols
        frames (:class:`~pandas.DataFrame`): DataFrame of periodic cell dimensions (or False if )
        k (int): Number of distances (per atom) to compute
        bond_extra (float): Extra distance to include when determining bonds (see above)
        dmax (float): Max distance of interest (larger distances are ignored)
        dmin (float): Min distance of interest (smaller distances are ignored)

    Returns:
        df (:class:`~atomic.twobody.TwoBody`): Two body property table
    '''
    if check(universe):
        if len(universe._super_atoms) == 0:
            if len(universe._primitive_atoms) == 0:
                return _periodic_from_atoms(universe, k=k, bond_extra=bond_extra, dmax=dmax, dmin=dmin)
            return _periodic_from_primitive(universe, k=k, bond_extra=bond_extra, dmax=dmax, dmin=dmin)
        return _periodic_from_super(universe, k=k, bond_extra=bond_extra, dmax=dmax, dmin=dmin)
    else:
        return _free_in_mem(universe, dmax=dmax, dmin=dmin, bond_extra=bond_extra)


def _free_in_mem(universe, dmax=12.3, dmin=0.3, bond_extra=bond_extra):
    '''
    Free boundary condition two body properties computed in memory.

    Args:
        universe (:class:`~atomic.universe.Universe`): The atomic universe
        dmax (float): Max distance of interest
        dmin (float): Minimum distance of interest
        bond_extra (float): Extra distance to add when determining bonds

    Returns:
        twobody (:class:`~atomic.twobody.TwoBody`)
    '''
    xyz = universe.atoms.groupby(level='frame')[['x', 'y', 'z']]
    syms = universe.atoms.groupby(level='frame')['symbol']
    n = xyz.ngroups
    distances = np.empty((n, ), dtype='O')
    symbols1 = np.empty((n, ), dtype='O')
    symbols2 = np.empty((n, ), dtype='O')
    indices = np.empty((n, ), dtype='O')
    frames = np.empty((n, ), dtype='O')
    atom1 = np.empty((n, ), dtype='O')
    atom2 = np.empty((n, ), dtype='O')
    for i, (fdx, xyz) in enumerate(xyz):   # Process each frame separately
        sym = syms.get_group(fdx)
        atom = xyz.index.get_level_values('atom')
        atom1[i], atom2[i] = list(zip(*combinations(atom, 2)))
        symbols1[i], symbols2[i] = list(zip(*combinations(sym, 2)))
        distances[i] = pdist(xyz.values)
        m = len(atom1[i])
        indices[i] = range(m)
        frames[i] = repeat_i8(fdx, m)
    distances = np.concatenate(distances)
    atom1 = np.concatenate(atom1)
    atom2 = np.concatenate(atom2)
    symbols1 = np.concatenate(symbols1)
    symbols2 = np.concatenate(symbols2)
    indices = np.concatenate(indices)
    frames = np.concatenate(frames)        # Build the dataframe all at once
    df = pd.DataFrame.from_dict({'distance': distances, 'symbol1': symbols1, 'symbol2': symbols2,
                                 'atom1': atom1, 'atom2': atom2, 'index': indices, 'frame': frames})
    df = df[(df['distance'] > dmin) & (df['distance'] < dmax)]
    df.set_index(['frame', 'index'], inplace=True)   # Prune and set indices
    df['symbols'] = df['symbol1'] + df['symbol2']
    df['r1'] = df['symbol1'].map(Isotope.symbol_radius)
    df['r2'] = df['symbol2'].map(Isotope.symbol_radius)
    df['mbl'] = df['r1'] + df['r2'] + bond_extra
    df['bond'] = df['distance'] < df['mbl']
    del df['symbol1']             # Remove duplicated/unnecessary data
    del df['symbol2']
    del df['r1']
    del df['r2']
    del df['mbl']
    return TwoBody(df)


def _free_memmap():    # Same as _free_in_mem but using numpy.memmap
    raise NotImplementedError()


def _periodic_from_atoms(universe, k=None, dmax=dmax, dmin=dmin, bond_extra=bond_extra):    # Compute primitive cell, super cell, and periodic two body
    '''
    Compute periodic two body properties given only the absolute positions and
    the cell dimensions.

    Args:
        universe (:class:`~atomic.universe.Universe`): The atomic universe
        k (int): Number of distances (per atom) to compute
        dmax (float): Max distance of interest
        dmin (float): Minimum distance of interest
        bond_extra (float): Extra distance to add when determining bonds

    Returns:
    '''
    raise NotImplementedError()


def _periodic_from_primitive(universe, k=None, dmax=dmax, dmin=dmin, bond_extra=bond_extra):
    '''
    Compute periodic two body properties given only the absolute positions and
    the cell dimensions.

    Args:
        universe (:class:`~atomic.universe.Universe`): The atomic universe
        k (int): Number of distances (per atom) to compute
        dmax (float): Max distance of interest
        dmin (float): Minimum distance of interest
        bond_extra (float): Extra distance to add when determining bonds

    Returns:
        periodic_twobody (:class:`~atomic.twobody.PeriodicTwoBody`): Periodic two body properties
    '''
    groups = universe.primitive_atoms.groupby(level='frame')
    pxyzs = groups[['x', 'y', 'z']]
    ng = groups.ngroups
    super_xyz = np.empty((ng, ), dtype='O')
    super_index = np.empty((ng, ), dtype='O')
    super_frame = np.empty((ng, ), dtype='O')
    super_atom = np.empty((ng, ), dtype='O')
    symbol_list = np.empty((ng, ), dtype='O')
    tb_super_atom1 = np.empty((ng, ), dtype='O')
    tb_super_atom2 = np.empty((ng, ), dtype='O')
    tb_dists = np.empty((ng, ), dtype='O')
    tb_frame = np.empty((ng, ), dtype='O')
    tb_index = np.empty((ng, ), dtype='O')
    for i, (fdx, pxyz) in enumerate(pxyzs):
        if np.mod(i, Config.gc) == 0:
            gc.collect()
        rx = universe.frames.ix[fdx, 'rx']
        ry = universe.frames.ix[fdx, 'ry']
        rz = universe.frames.ix[fdx, 'rz']
        atom = pxyz.index.get_level_values('atom').values
        nat = len(atom)
        k = nat if k is None else k
        super_xyz[i] = periodic_supercell(pxyz.values, rx, ry, rz)
        nsuper = len(super_xyz[i])
        super_frame[i] = repeat_i8(fdx, nsuper)
        super_index[i] = range(nsuper)
        super_atom[i] = tile_i8(atom, 27)
        symbol_list[i] = universe.atoms.ix[fdx, 'symbol'].tolist() * 27
        dists, indices = cKDTree(super_xyz[i]).query(pxyz, k=k, distance_upper_bound=dmax)
        tb_dists[i] = dists.ravel()
        ntb = len(tb_dists[i])
        tb_super_atom1[i] = repeat_i8_array(indices[:, 0], k)
        tb_super_atom2[i] = indices.ravel()
        tb_frame[i] = repeat_i8(fdx, ntb)
        tb_index[i] = range(ntb)

    # SuperAtom
    super_xyz = np.concatenate(super_xyz)
    super_frame = np.concatenate(super_frame)
    super_index = np.concatenate(super_index)
    super_atom = np.concatenate(super_atom)
    super_atomsdf = pd.DataFrame(super_xyz, columns=['x', 'y', 'z'])
    super_atomsdf['frame'] = super_frame
    super_atomsdf['super_atom'] = super_index
    super_atomsdf['atom'] = super_atom
    super_atomsdf['symbol'] = np.concatenate(symbol_list)
    super_atomsdf.set_index(['frame', 'super_atom'], inplace=True)

    # PeriodicTwoBody
    tb_dists = np.concatenate(tb_dists)
    tb_super_atom1 = np.concatenate(tb_super_atom1)
    tb_super_atom2 = np.concatenate(tb_super_atom2)
    tb_frame = np.concatenate(tb_frame)
    tb_index = np.concatenate(tb_index)
    tbdf = pd.DataFrame.from_dict({'distance': tb_dists, 'super_atom1': tb_super_atom1,
                                   'super_atom2': tb_super_atom2, 'frame': tb_frame,
                                   'index': tb_index})
    tbdf.set_index(['frame', 'index'], inplace=True)
    tbdf = tbdf[(tbdf['distance'] > dmin) & (tbdf['distance'] < dmax)]
    mapper = super_atomsdf['atom'].to_dict()
    def map_mapper(value):
        return mapper[value]
    df = tbdf.reset_index('index').set_index('super_atom1', append=True)
    df.index.names = ['frame', 'super_atom']
    tbdf['atom1'] = df.index.map(map_mapper)
    df = df.reset_index('super_atom', drop=True).set_index('super_atom2', append=True)
    df.index.names = ['frame', 'super_atom']
    tbdf['atom2'] = df.index.map(map_mapper)
    mapper = universe.atoms['symbol'].to_dict()
    df = tbdf.reset_index('index').set_index('atom1', append=True)
    df.index.names = ['frame', 'atom']
    tbdf['symbol1'] = df.index.map(map_mapper)
    df = df.reset_index('atom', drop=True).set_index('atom2', append=True)
    tbdf.index.names = ['frame', 'atom']
    tbdf['symbol2'] = df.index.map(map_mapper)
    tbdf['symbols'] = tbdf['symbol1'] + tbdf['symbol2']
    tbdf['r1'] = tbdf['symbol1'].map(Isotope.symbol_radius)
    tbdf['r2'] = tbdf['symbol2'].map(Isotope.symbol_radius)
    tbdf['mbl'] = tbdf['r1'] + tbdf['r2'] + bond_extra
    tbdf['bond'] = tbdf['distance'] < tbdf['mbl']
    del tbdf['r1']
    del tbdf['r2']
    del tbdf['mbl']
    tbdf.index.names = ['frame', 'index']
    return SuperAtom(super_atomsdf), PeriodicTwoBody(tbdf)


def _periodic_from_super(universe, k=None, dmax=dmax, dmin=dmin, bond_extra=bond_extra):
    '''
    '''
    raise NotImplementedError('Bug here; use _periodic_from_primitive')
    xyz_groups = universe.super_atoms.groupby(level='frame')[['x', 'y', 'z']]
    n = xyz_groups.ngroups
    distances = np.empty((n, ), dtype='O')
    super_atom1 = np.empty((n, ), dtype='O')
    super_atom2 = np.empty((n, ), dtype='O')
    symbol1 = np.empty((n, ), dtype='O')
    symbol2 = np.empty((n, ), dtype='O')
    distances = np.empty((n, ), dtype='O')
    indices = np.empty((n, ), dtype='O')
    frames = np.empty((n, ), dtype='O')
    for i, (fdx, xyz) in enumerate(xyz_groups):    # Deal with each frame
        nat = universe.frames.ix[fdx, 'atom_count']
        k = nat if k is None else k
        xyzv = xyz.values
        central = xyzv[13*nat:14*nat]
        dists, idxs = cKDTree(xyzv).query(central, k=k, distance_upper_bound=dmax)
        super_atom1[i] = repeat_i8_array(idxs[:, 0], k)
        super_atom2[i] = idxs.ravel()
        distances[i] = dists.ravel()
        n = len(distances[i])
        indices[i] = range(n)
        frames[i] = repeat_i8(fdx, n)
    distances = np.concatenate(distances)
    super_atom1 = np.concatenate(super_atom1)
    super_atom2 = np.concatenate(super_atom2)
    indices = np.concatenate(indices)
    frames = np.concatenate(frames)    # Build the dataframe once
    tbdf = pd.DataFrame.from_dict({'distance': distances, 'super_atom1': super_atom1,
                                 'super_atom2': super_atom2, 'index': indices,
                                 'frame': frames})
    tbdf = tbdf[(tbdf['distance'] > dmin) & (tbdf['distance'] < dmax)] # Slow from here down
    tbdf.set_index(['frame', 'index'], inplace=True)   # Prune and index the dataframe
    mapper = universe.super_atoms['atom'].to_dict()
    def map_mapper(value):
        return mapper[value]
    df = tbdf.reset_index('index').set_index('super_atom1', append=True)
    df.index.names = ['frame', 'super_atom']
    tbdf['atom1'] = df.index.map(map_mapper)
    df = df.reset_index('super_atom', drop=True).set_index('super_atom2', append=True)
    df.index.names = ['frame', 'super_atom']
    tbdf['atom2'] = df.index.map(map_mapper)
    mapper = universe.atoms['symbol'].to_dict()
    df = tbdf.reset_index('index').set_index('atom1', append=True)
    df.index.names = ['frame', 'atom']
    tbdf['symbol1'] = df.index.map(map_mapper)
    df = df.reset_index('atom', drop=True).set_index('atom2', append=True)
    tbdf.index.names = ['frame', 'atom']
    tbdf['symbol2'] = df.index.map(map_mapper)
    tbdf['symbols'] = tbdf['symbol1'] + tbdf['symbol2']
    tbdf['r1'] = tbdf['symbol1'].map(Isotope.symbol_radius)
    tbdf['r2'] = tbdf['symbol2'].map(Isotope.symbol_radius)
    tbdf['mbl'] = tbdf['r1'] + tbdf['r2'] + bond_extra
    tbdf['bond'] = tbdf['distance'] < tbdf['mbl']
    del tbdf['r1']
    del tbdf['r2']
    del tbdf['mbl']
    tbdf.index.names = ['frame', 'index']
    return PeriodicTwoBody(tbdf)


def compute_bond_counts(universe):
    '''
    '''
    grouped_twobody = universe.twobody.groupby(level='frame')
    counts = np.empty((grouped_twobody.ngroups, ), dtype='O')
    for i, (fdx, group) in enumerate(grouped_twobody):
        atom1_counts = group.groupby('atom1')['bond'].sum()
        atom2_counts = group.groupby('atom2')['bond'].sum()
        counts[i] = atom1_counts.add(atom2_counts, fill_value=0) // 2
    return pd.concat(counts).values.astype(np.int64)
