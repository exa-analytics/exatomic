# -*- coding: utf-8 -*-
'''
Universe Slicing Algorithms
===============================================
These are methods and algorithms for slicing the atomic specific data
container. Included here are algorithms for complex operations such as
slicing by selection of nearest neighbors.
'''
import numpy as np
import pandas as pd


def nearest_molecules(universe, n, sources, others=None, symbols=None,
                              by='atom', exact=False, convert=True):
    '''
    Get nearest n neighbor molecules/atoms to (each - if multiple source molecules/atoms
    exist per frame) other molecule/atom.

    .. code-block:: Python

        # Get 5 nearest anything to solute
        compute_nearest_molecules(universe, 5, 'solute')
        # Get 5 nearest anything to solute's Na atom(s)
        compute_nearest_molecules(universe, 5, 'solute', symbols='Na')
        # Get 5 nearest solvent to solute
        compute_nearest_molecules(universe, 5, 'solute', 'solvent')
        # Get 5 nearest solvent/hydroxide to solute
        compute_nearest_molecules(universe, 5, 'solute', ['solvent', 'hydroxide'])
        # Get 5 nearest classified molecules to solute
        compute_nearest_molecules(universe, 5, 'solute', 'solvent', exact=True)

    Args:
        universe (:class:`~atomic.universe.Universe): The atomic Universe
        n (int): Number of neighbors to find
        sources (str or list): Molecules/atoms to search from
        others (str or list): Molecules/atoms to select
        symbols (str or list): Specific atoms of the source(s) to look from
        by (str): Search criteria, default 'atom': atom to atom distance, 'com': center of mass
        exact (bool): If false (default), include unclassified in partialverse, w/o counting
        convert (bool): If periodic universe, convert to free boundary conditions (default)

    Returns:
        partialverse (:class:`~atomic.universe.Universe): Partial universe with only nearest neighbors

    '''
    # Check arguments
    if 'classification' not in universe.molecule:
        raise TypeError('Please classify the molecules.')
    if isinstance(sources, str):
        sources = [sources]
    elif not isinstance(sources, list):
        raise TypeError('sources must be str or list')
    if isinstance(others, str):
        others = [others] if exact else [others, np.nan]
    elif not isinstance(sources, list) and others is not None:
        raise TypeError('others must be str, list, or None')
    elif others is None:
        others = list(set(universe.molecule['classification'].cat.categories).difference(sources))
        if not exact:
            others += [np.nan]
    if isinstance(symbols, str):
        symbols = [symbols]
    elif not isinstance(symbols, list) and symbols is not None:
        raise TypeError('symbols must be str, list, or None')
    # Determine grouping algorithm
    if by == 'atom' and universe.is_periodic:
        return _periodic_byatom(universe, n, sources, others, symbols, convert)
    elif by == 'atom' and not universe.is_periodic:
        raise NotImplementedError()
    elif by == 'com':
        raise NotImplementedError()


def _periodic_byatom(uni, n, sources, others, symbols, convert):
    '''
    Args:
        uni (:class:`~atomic.universe.Universe): Atomic universe
        n (int): Number of nearest molecules
        sources (list): List of source identifiers
        others (list): List of other identifiers
        symbols (list): List of source symbols to look from
        convert (bool): Whether to convert to free boundary conditions or not
    '''
    # Select molecules using the classifications provided
    smolecules = uni.molecule[uni.molecule['classification'].isin(sources)]
    if len(smolecules) == 0:
        raise KeyError('No source molecules using classification(s) {}.'.format(sources))
    if n == 0:
        return smolecules.index.values.astype(np.int64)
    omolecules = uni.molecule[uni.molecule['classification'].isin(others)]
    if len(omolecules) == 0:
        raise KeyError('No other molecules using classification(s) {}.'.format(others))

    # Converting molecule indices into atom indices to search two body data
    satoms = uni.atom[uni.atom['molecule'].isin(smolecules.index)]
    if isinstance(symbols, list):
        satoms = satoms[satoms['symbol'].isin(symbols)]
    oatoms = uni.atom[uni.atom['molecule'].isin(omolecules.index)]
    spa_idx = uni.projected_atom[uni.projected_atom['atom'].isin(satoms.index)].index
    opa_idx = uni.projected_atom[uni.projected_atom['atom'].isin(oatoms.index)].index

    # Select the relevant distances
    dists = uni.two[(uni.two['prjd_atom0'].isin(spa_idx) & uni.two['prjd_atom1'].isin(opa_idx)) |
                    (uni.two['prjd_atom1'].isin(spa_idx) & uni.two['prjd_atom0'].isin(opa_idx))]

    # The dists object has every frame in it, so we need to groupb by frame and sort
    # the distances of that specific frame. On the backend we only need the (prjd)
    # atom indices and the corresponding frame.
    fullstack = dists.groupby('frame').apply(lambda g: g.sort_values('distance'))
    fullstack = fullstack[['prjd_atom0', 'prjd_atom1']].stack().to_frame().reset_index(level=1, drop=True)
    fullstack.columns = ['prjd_atom']
    fullstack['atom'] = fullstack['prjd_atom'].map(uni.projected_atom['atom'])
    fullstack['molecule'] = fullstack['atom'].map(uni.atom['molecule'])
    fullstack['frame'] = fullstack['molecule'].map(uni.molecule['frame'])
    fullstack['classification'] = fullstack['molecule'].map(uni.molecule['classification'])
    stack = fullstack[~fullstack['molecule'].isin(smolecules.index)]


    mapper = {key: 0 if key is np.nan else 1 for key in others}
    unique = stack.drop_duplicates('molecule', keep='first')

    search = unique.groupby('frame').apply(lambda g: g['classification'].astype(object).map(
                                           lambda x: mapper[x]).cumsum().values)
    search.index = search.index.astype(np.int64)
    search = pd.Series({frame: np.where(search[frame] == n)[0][0] for frame in search.index}) + 1

    sm = smolecules.groupby('frame').apply(lambda g: list(g.index))
    sm.index = sm.index.astype(np.int64)

    grps = unique.groupby('frame')
    mids = np.empty((grps.ngroups, ), dtype='O')
    for i, (frame, grp) in enumerate(grps):
        mids[i] = sm[frame] + grp.iloc[:search[frame], 2].tolist()
    mids = np.concatenate(mids).astype(np.int64)

    # Build free boundary universe
    # By virtue of the selection process, this returns a system
    # in free boundary conditions; this is typically what the user
    # because it is hard to visually inspect periodic nearest neighbors.
    return mids, fullstack
    updater = uni.projected_atom[uni.projected_atom.index.isin(fullstack['prjd_atom'])]
    updater = updater.set_index('atom')[['x', 'y', 'z']]

    atom = uni.visual_atom[uni.visual_atom['molecule'].isin(mids)].copy()
    print(atom.shape)
    print(updater.shape)
    atom.update(updater)
    atom = atomic.Atom(atom)
    atom.reset_label()
    u = atomic.Universe(atom=atom)
    return u


#def slice_by_molecules(universe, mids):
#    '''
#    '''
#    kwargs = {}
#    molecule = universe.molecule[universe.molecule.index.isin(mids)]
#    molecule = atomic.Molecule(molecule.copy())
#    kwargs['molecule'] = molecule
#    atom = universe.atom[universe.atom['molecule'].isin(mids)]
#    atom = atomic.Atom(atom.copy())
#    atom.reset_label()
#    kwargs['atom'] = atom
#    frame = universe.frame[universe.frame.index.isin(atom['frame'])]
#    frame = atomic.Frame(frame.copy())
#    kwargs['frame'] = frame
#    if universe.field:
#        kwargs['field'] = atomic.field.AtomicField(universe.field_values, universe.field.copy())
#    if universe.is_periodic:
#        unit_atom = universe._unit_atom[universe._unit_atom.index.isin(atom.index)]
#        unit_atom = atomic.atom.UnitAtom(unit_atom.copy())
#        kwargs['unit_atom'] = unit_atom
#        projected_atom = universe.projected_atom[universe.projected_atom['atom'].isin(atom.index)]
#        projected_atom = atomic.atom.ProjectedAtom(projected_atom.copy())
#        kwargs['projected_atom'] = projected_atom
#        two = universe.two[(universe.two['prjd_atom0'].isin(projected_atom.index) &
#                            universe.two['prjd_atom1'].isin(projected_atom.index)) |
#                           (universe.two['prjd_atom1'].isin(projected_atom.index) &
#                            universe.two['prjd_atom0'].isin(projected_atom.index))]
#        two = atomic.two.PeriodicTwo(two.copy())
#        kwargs['periodic_two'] = two
#    else:
#        two = universe.two[universe.two['atom0'].isin(atom.index) &
#                           universe.two['atom1'].isin(atom.index)]
#        two = atomic.two.Two(two.copy())
#        kwargs['two'] = two
#    return atomic.Universe(**kwargs)
