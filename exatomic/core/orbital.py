# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Orbital DataFrame
####################
Orbital information. All of the dataframe structures and functions associated
with the results of a quantum chemical calculation. The Orbital table itself
summarizes information such as centers and energies. The Excitation table
collects information about orbital excitations from time-dependent calculations.
The convolve() bound method can be used to generate photoelectron spectroscopy
and absorbance spectra.

The MOMatrix table contains a C matrix as it is presented in quantum textbooks,
stored in a columnar format. The bound method square() returns the
matrix as one would write it out. This table should have dimensions
N_basis_functions * N_basis_functions. The DensityMatrix table stores
a triangular matrix in columnar format and contains a similar square()
method to return the matrix as we see it on a piece of paper.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
from exa import DataFrame
from exa.util.units import Energy
from exatomic.algorithms.numerical import (density_from_momatrix,
                                           density_as_square)
                                           #momatrix_as_square)
from exatomic.core.field import AtomicField


class _Convolve(DataFrame):
    @property
    def _constructor(self):
        return _Convolve

    @staticmethod
    def _gauss(sigma, en, en0):
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
               np.exp(-(en - en0) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def _lorentz(gamma, en, en0):
        return gamma / (2 * np.pi * (en - en0) ** 2 + (gamma / 2) ** 2)

    @property
    def last_frame(self):
        return self['frame'].cat.as_ordered().max()

    @property
    def last_group(self):
        return self[self.frame == self.last_frame].group.cat.as_ordered().max()

    def convolve(self, func='gauss', units='eV', ewin=None, broaden=0.13,
                 padding=5, npoints=1001, group=None, frame=None, name=None,
                 normalize=True):
        """
        Compute a spectrum based on excitation energies and oscillator strengths.

        Args:
            func (str): either 'gauss' or 'lorentz'
            units (str): units of resulting spectrum
            ewin (iter): (emin, emax) in same units as units (default in eV)
            broaden (float): how broad to make the peaks (FWHM, default in eV)
            npoints (int): "resolution" of the spectrum
            name (str): optional - name the column of returned data
            normalize (bool): set the largest value of signal equal to 1.0

        Returns:
            df (pd.DataFrame): contains x and y values of a spectrum
                               (signal and energy)
        """
        frame = self.last_frame if frame is None else frame
        group = self.last_group if group is None else group
        if func not in ['gauss', 'lorentz']:
            raise NotImplementedError('Convolution must be one of "gauss" or "lorentz".')
        choices = {'gauss': self._gauss, 'lorentz': self._lorentz}
        if units == 'Ha': units = 'energy'
        if units not in self.columns:
            self[units] = self['energy'] * Energy['Ha', units]
        sm = self[units].min() if ewin is None else ewin[0]
        lg = self[units].max() if ewin is None else ewin[1]
        mine = sm - padding * broaden
        maxe = lg + padding * broaden
        enrg = np.linspace(mine, maxe, npoints)
        spec = np.zeros(npoints)
        if self.__class__.__name__ == 'Excitation':
            smdf = self[(self[units] > sm) & (self[units] < lg) &
                        (self['frame'] == frame) & self['group'] == group]
            for osc, en0 in zip(smdf['osc'], smdf[units]):
                spec += osc * choices[func](broaden, enrg, en0)
        else:
            smdf = self[(self[units] > sm) & (self[units] < lg) &
                        (self['occupation'] > 0)]
            for en0 in smdf[units]:
                spec += choices[func](broaden, enrg, en0)
        if np.isclose(spec.max(), 0):
            print('Spectrum is all zeros, check energy window.')
        else:
            if normalize:
                spec /= spec.max()
        name = 'signal' if name is None else name
        return pd.DataFrame.from_dict({units: enrg, name: spec})


class Orbital(_Convolve):
    """
    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | frame             | category | non-unique integer (req.)                 |
    +-------------------+----------+-------------------------------------------+
    | group             | category | like frame but for same geometry          |
    +-------------------+----------+-------------------------------------------+
    | orbital           | int      | vector of MO coefficient matrix           |
    +-------------------+----------+-------------------------------------------+
    | label             | int      | label of orbital                          |
    +-------------------+----------+-------------------------------------------+
    | occupation        | float    | population of orbital                     |
    +-------------------+----------+-------------------------------------------+
    | energy            | float    | eigenvalue of orbital eigenvector         |
    +-------------------+----------+-------------------------------------------+
    | symmetry          | str      | symmetry designation (if applicable)      |
    +-------------------+----------+-------------------------------------------+
    | x                 | float    | orbital center in x                       |
    +-------------------+----------+-------------------------------------------+
    | y                 | float    | orbital center in y                       |
    +-------------------+----------+-------------------------------------------+
    | z                 | float    | orbital center in z                       |
    +-------------------+----------+-------------------------------------------+

    Note:
        Spin zero means alpha spin or unknown and spin one means beta spin.
    """
    _columns = ['frame', 'group', 'energy', 'occupation', 'vector', 'spin']
    _index = 'orbital'
    _cardinal = ('frame', np.int64)
    _categories = {'spin': np.int64, 'frame': np.int64, 'group': np.int64}

    #@property
    #def _constructor(self):
    #    return Orbital

    def get_orbital(self, orb=-1, spin=0, orbocc='occupation',
                    index=None, group=None, frame=None):
        """
        Returns a specific orbital.

        Args:
            orb (int): See note below (default HOMO)
            spin (int): 0, no spin or alpha (default); 1, beta
            index (int): Orbital dataframe index (default None)
            frame (int): The frame of the universe (default max(frame))
            group (int): The group of orbitals within a given frame
            orbocc (str): column of occupations (default 'occupation')

        Returns:
            slc (pd.Series): row of the Orbital table

        Note:
            If the index is not known (usually), but a criterion
            such as (HOMO or LUMO) is desired, use the *orb* and
            *spin* criteria. Negative *orb* values are occupied,
            positive are unoccupied. So -1 returns the HOMO, -2
            returns the HOMO-1; 0 returns the LUMO, 1 returns the
            LUMO+1, etc.
        """
        frame = self.last_frame if frame is None else frame
        group = self.last_group if group is None else group
        if index is None:
            if orb > -1:
                return self[(self['frame'] == frame) &
                            (self['group'] == group) &
                            (self[orbocc] == 0) &
                            (self['spin'] == spin)].iloc[orb]
            else:
                return self[(self['frame'] == frame) &
                            (self['group'] == group) &
                            (self[orbocc] > 0) &
                            (self['spin'] == spin)].iloc[orb]
        else:
            return self.iloc[index]

    def active_orbitals(self, orbocc='occupation', maxocc=1.985, minocc=0.015,
                        irrep=None):
        """
        Returns a slice of the orbital table containing so-called
        active orbitals (defined here as orbitals with occupations
        between minocc and maxocc).

        Args:
            orbocc (str): column name of occupation numbers in uni.orbital
            maxocc (float): maximum occupation to be considered active
            minocc (float): minimum occupation to be considered active
            irrep (int): irreducible representation

        Returns:
            slc (pd.DataFrame): active orbitals in the Orbital table
        """
        if irrep is not None:
            return self[(self[orbocc] < maxocc) & (self[orbocc] > minocc) &
                        (self['irrep'] == irrep)]
        return self[(self[orbocc] < maxocc) & (self[orbocc] > minocc)]


    @classmethod
    def from_energies(cls, energies, alphae, betae, os=False):
        """
        Convenience method to generate the Orbital table from an array
        of orbital energies and the number of alpha and beta electrons.

        Args:
            energies (np.ndarray): orbital energies
            alphae (int): number of alpha electrons
            betae (int): number of beta electrons
            os (bool): open shell

        Returns:
            orb (~:class:`exatomic.core.orbital.Orbital`): the Orbital table
        """
        try: ae, be = int(alphae), int(betae)
        except: raise NotImplementedError('Only integer occupation. '
                                          'Low hanging fruit to implement.')
        nmos = energies.shape[0] if not os else energies.shape[0] // 2
        nmos = energies.shape[0] // 2
        if os:
            nmos = energies.shape[0] // 2
            spin = np.concatenate((np.zeros(nmos), np.ones(nmos)))
            vector = np.concatenate((range(nmos), range(nmos)))
            occs = np.concatenate((np.ones(ae), np.zeros(nmos - ae),
                                   np.ones(be), np.zeros(nmos - be)))
            frame = group = np.zeros(nmos * 2, dtype=np.int64)
        else:
            nmos = energies.shape[0]
            spin = np.zeros(nmos, dtype=np.int64)
            vector = range(nmos)
            occs = np.concatenate((np.repeat(2, ae),
                                   np.zeros(nmos - ae)))
            frame = group = np.zeros(nmos, dtype=np.int64)
        return cls.from_dict({'frame': frame, 'group': group,
                              'energy': energies, 'spin': spin,
                              'occupation': occs, 'vector': vector})

    @classmethod
    def from_occupation_vector(cls, occvec, os=False):
        """
        Convenience method to generate the Orbital table from an array
        of occupation numbers.

        Args:
            energies (np.ndarray): orbital energies
            os (bool): open shell

        Returns:
            orb (~:class:`exatomic.core.orbital.Orbital`): the Orbital table
        """
        if not os: return cls.from_dict({'frame': 0, 'group': 0,
                                         'energy': 0, 'spin': 0,
                                         'occupation': occvec, 'vector': range(len(occvec))})
        else:
            raise NotImplementedError('Implement open shell lazy')


class Excitation(_Convolve):
    """
    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | energy            | float    | excitation energy in Ha                   |
    +-------------------+----------+-------------------------------------------+
    | irrep             | str      | irreducible representation of excitation  |
    +-------------------+----------+-------------------------------------------+
    | osc               | float    | oscillator strength (length repr.)        |
    +-------------------+----------+-------------------------------------------+
    | occ               | int      | occupied orbital of excitation            |
    +-------------------+----------+-------------------------------------------+
    | virt              | int      | virtual orbital of excitation             |
    +-------------------+----------+-------------------------------------------+
    | occsym            | str      | occupied orbital symmetry                 |
    +-------------------+----------+-------------------------------------------+
    | virtsym           | str      | virtual orbital symmetry                  |
    +-------------------+----------+-------------------------------------------+
    | frame             | int      | non-unique integer (req.)                 |
    +-------------------+----------+-------------------------------------------+
    | group             | int      | like frame but for same geometry          |
    +-------------------+----------+-------------------------------------------+
    """
    _columns = ['energy', 'osc', 'frame', 'group']
    _index = 'excitation'
    _cardinal = ('frame', np.int64)
    _categories = {'frame': np.int64, 'group': np.int64}

    #@property
    #def _constructor(self):
    #    return Excitation

    @classmethod
    def from_universe(cls, uni, initial=None, final=None, spin=0):
        """
        Generate the zeroth order approximation to excitation energies
        via the transition dipole method (provided a universe contains
        an MOMatrix and dipole moment integrals already).
        """
        if not hasattr(uni, 'multipole'):
            print('Universe must have dipole integrals.')
            return
        dim = len(uni.basis_set_order.index)
        fix = (np.ones((dim, dim)) - np.eye(dim, dim) / 2)
        rx = ((uni.multipole.pivot('chi0', 'chi1', 'ix1').fillna(0.0)
             + uni.multipole.pivot('chi0', 'chi1', 'ix1').T.fillna(0.0)) * fix).values
        ry = ((uni.multipole.pivot('chi0', 'chi1', 'ix2').fillna(0.0)
             + uni.multipole.pivot('chi0', 'chi1', 'ix2').T.fillna(0.0)) * fix).values
        rz = ((uni.multipole.pivot('chi0', 'chi1', 'ix3').fillna(0.0)
             + uni.multipole.pivot('chi0', 'chi1', 'ix3').T.fillna(0.0)) * fix).values
        mo = uni.momatrix.square().values
        ens = pd.concat([uni.orbital[uni.orbital.spin == spin].energy] * dim, axis=1).values
        tdm = pd.DataFrame.from_dict({
            'energy': pd.DataFrame(ens.T - ens).stack(),
            'mux': pd.DataFrame(np.dot(mo.T, np.dot(rx, mo))).stack(),
            'muy': pd.DataFrame(np.dot(mo.T, np.dot(ry, mo))).stack(),
            'muz': pd.DataFrame(np.dot(mo.T, np.dot(rz, mo))).stack()})
        tdm['osc'] = tdm['energy'] ** 3 * (tdm['mux'] + tdm['muy'] + tdm['muz']) ** 2
        tdm['frame'] = tdm['group'] = 0
        tdm.index.rename(['occ', 'virt'], inplace=True)
        return cls(tdm.reset_index())


class MOMatrix(DataFrame):
    """
    The MOMatrix is the result of solving a quantum mechanical eigenvalue
    problem in a finite basis set. Individual columns are eigenfunctions
    of the Fock matrix with eigenvalues corresponding to orbital energies.

    .. math::

        C^{*}SC = 1

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | chi               | int      | row of MO coefficient matrix              |
    +-------------------+----------+-------------------------------------------+
    | orbital           | int      | vector of MO coefficient matrix           |
    +-------------------+----------+-------------------------------------------+
    | coef              | float    | weight of basis_function in MO            |
    +-------------------+----------+-------------------------------------------+
    | frame             | category | non-unique integer (req.)                 |
    +-------------------+----------+-------------------------------------------+
    """
    # TODO :: add an irrep column for groupby?
    #_traits = ['orbital']
    _columns = ['chi', 'orbital']
    _cardinal = ('frame', np.int64)
    _index = 'index'

    #@property
    #def _constructor(self):
    #    return MOMatrix

    def contributions(self, orbital, mocoefs='coef', tol=0.01, frame=0):
        """
        Returns a slice containing all non-negligible basis function
        contributions to a specific orbital.

        Args
            orbital (int): orbital index
        """
        tmp = self[self['frame'] == frame].groupby('orbital').get_group(orbital)
        return tmp[np.abs(tmp[mocoefs]) > tol]


    def square(self, frame=0, column='coef', mocoefs=None, irrep=None):
        """
        Returns a square dataframe corresponding to the canonical C matrix
        representation.
        """
        if mocoefs is None: mocoefs = column
        if 'irrep' in self.columns:
            if irrep is None:
                irreps, i, j = self.groupby('irrep'), 0, 0
                norb = (irreps.orbital.max() + 1).sum()
                nchi = (irreps.chi.max() + 1).sum()
                cmat = np.zeros((nchi, norb))
                for irrep, grp in irreps:
                    piv = grp.pivot('chi', 'orbital', mocoefs)
                    ii, jj = piv.shape
                    cmat[i : i + ii, j : j + jj] = piv.values
                    i += ii
                    j += jj
                return pd.DataFrame(cmat, index=pd.Index(range(nchi), name='chi'),
                                    columns=pd.Index(range(norb), name='orbital'))
            return self.groupby('irrep').get_group(irrep
                        ).pivot('chi', 'orbital', mocoefs)
        return self.pivot('chi', 'orbital', mocoefs)


class DensityMatrix(DataFrame):
    """
    The density matrix in a contracted basis set. As it is
    square symmetric, only n_basis_functions * (n_basis_functions + 1) / 2
    rows are stored.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | chi0              | int      | first index in matrix                     |
    +-------------------+----------+-------------------------------------------+
    | chi1              | int      | second index in matrix                    |
    +-------------------+----------+-------------------------------------------+
    | coef              | float    | overlap matrix element                    |
    +-------------------+----------+-------------------------------------------+
    | frame             | category | non-unique integer (req.)                 |
    +-------------------+----------+-------------------------------------------+
    """
    _columns = ['chi0', 'chi1', 'coef']
    _cardinal = ('frame', np.int64)
    _index = 'index'

    #@property
    #def _constructor(self):
    #    return DensityMatrix

    def square(self, frame=0):
        """Returns a square dataframe of the density matrix."""
        denvec = self[self['frame'] == frame]['coef'].values
        square = pd.DataFrame(density_as_square(denvec))
        square.index.name = 'chi0'
        square.columns.name = 'chi1'
        return square

    @classmethod
    def from_momatrix(cls, momatrix, occvec, mocoefs='coef'):
        """
        A density matrix can be constructed from an MOMatrix by:
        .. math::

            D_{uv} = \sum_{i}^{N} C_{ui} C_{vi} n_{i}

        Args:
            momatrix (:class:`~exatomic.orbital.MOMatrix`): a C matrix
            occvec (:class:`~np.array` or similar): vector of len(C.shape[0])
                containing the occupations of each molecular orbital.

        Returns:
            ret (:class:`~exatomic.orbital.DensityMatrix`): The density matrix
        """
        cmat = momatrix.square(column=mocoefs).values
        chi0, chi1, dens, frame = density_from_momatrix(cmat, occvec)
        return cls.from_dict({'chi0': chi0, 'chi1': chi1,
                              'coef': dens, 'frame': frame})

    @classmethod
    def from_universe(cls, uni, mocoefs, orbocc):
        """
        The density matrix is defined as:
        .. math::

            D_{uv} = \sum_{i}^{N} C_{ui} C_{vi} n_{i}

        Args:
            uni (:class:`~exatomic.core.universe.Universe`): a universe containing momatrix and orbital
            mocoefs (str): column name of C matrix in uni.momatrix
            orbocc (str): column name of occupation vector in uni.orbital

        Returns:
            ret (:class:`~exatomic.orbital.DensityMatrix`): The density matrix
        """
        cmat = uni.momatrix.square(mocoefs=mocoefs).values
        occvec = uni.orbital[orbocc].values
        chi0, chi1, dens, frame = density_from_momatrix(cmat, occvec)
        return cls.from_dict({'chi0': chi0, 'chi1': chi1,
                              'coef': dens, 'frame': frame})
