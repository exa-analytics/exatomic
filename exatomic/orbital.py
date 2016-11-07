# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
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
'''
import numpy as np
import pandas as pd
from exa import DataFrame
from exatomic.algorithms.orbital import (density_from_momatrix,
                                         density_as_square,
                                         momatrix_as_square)
from exatomic.field import AtomicField

class _Convolve(DataFrame):

    def _gauss(sigma, en, en0):
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
               np.exp(-(en - en0) ** 2 / (2 * sigma ** 2))

    def _lorentz(gamma, en, en0):
        return gamma / (2 * np.pi * (en - en0) ** 2 + (gamma / 2) ** 2)

    def convolve(self, func='gauss', units='eV', ewin=None, broaden=0.13,
                 padding=5, npoints=1001, name=None):
        """
        Compute a spectrum based on excitation energies and oscillator strengths.

        Args
            func (str): either 'gauss' or 'lorentz'
            units (str): units of resulting spectrum
            ewin (iter): (emin, emax) in same units as units (default in eV)
            broaden (float): how broad to make the peaks (FWHM, default in eV)
            npoints (int): "resolution" of the spectrum
            name (str): optional - name the column of returned data

        Returns
            df (pd.DataFrame): contains x and y values of a spectrum
                               (signal and energy)
        """
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
        smdf = self[(self[units] > sm) & (self[units] < lg)]
        if self.__name__ == 'Excitation':
            for osc, en0 in zip(smdf['osc'], smdf[units]):
                spec += osc * choices[func](broaden, enrg, en0)
        else:
            for en0 in smdf[units]:
                spec += choices[func](broaden, enrg, en0)
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
    _columns = ['frame', 'energy', 'occupation', 'vector', 'spin']
    _index = 'orbital'
    _cardinal = ('frame', np.int64)
    _categories = {'spin': np.int64}

    def get_orbital(self, frame=0, orb=-1, spin=0, index=None):
        """
        Returns a specific orbital.

        Args
            orb (int): See note below (default HOMO)
            spin (int): 0, no spin or alpha (default); 1, beta
            index (int): Orbital index (default None)
            frame (int): The frame of the universe (default 0)

        Returns
            orbital (exatomic.orbital.Orbital): Orbital row

        Note
            If the index is not known (usually), but a criterion
            such as (HOMO or LUMO) is desired, use the *orb* and
            *spin* criteria. Negative *orb* values are occupied,
            positive are unoccupied. So -1 returns the HOMO, -2
            returns the HOMO-1; 0 returns the LUMO, 1 returns the
            LUMO+1, etc.
        """
        if index is None:
            if orb > -1:
                return self[(self['frame'] == frame) &
                            (self['occupation'] == 0) &
                            (self['spin'] == spin)].iloc[orb]
            else:
                return self[(self['frame'] == frame) &
                            (self['occupation'] > 0) &
                            (self['spin'] == spin)].iloc[orb]
        else:
            return self.iloc[index]

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
    """
    _columns = ['energy', 'osc', 'frame']
    _index = 'excitation'
    _cardinal = ('frame', np.int64)
    _categories = {}


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
    # TODO :: add spin as a column and make it the first groupby?
    #_traits = ['orbital']
    _columns = ['chi', 'orbital']
    _cardinal = ('frame', np.int64)
    _index = 'index'

    def contributions(self, orbital, tol=0.01, frame=0):
        """
        Returns a slice containing all non-negligible basis function
        contributions to a specific orbital.

        Args
            orbital (int): orbital index
        """
        tmp = self[self['frame'] == frame].groupby('orbital').get_group(orbital)
        return tmp[abs(tmp['coefficient']) > tol]


    def square(self, frame=0, column='coefficient'):
        """
        Returns a square dataframe corresponding to the canonical C matrix
        representation.
        """
        movec = self[self['frame'] == frame][column].values
        square = pd.DataFrame(momatrix_as_square(movec))
        square.index.name = 'chi'
        square.columns.name = 'orbital'
        return square


class DensityMatrix(DataFrame):
    """
    The density matrix in a contracted basis set. As it is
    square symmetric, only n_basis_functions * (n_basis_functions + 1) / 2
    rows are stored.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | chi1              | int      | first basis function                      |
    +-------------------+----------+-------------------------------------------+
    | chi2              | int      | second basis function                     |
    +-------------------+----------+-------------------------------------------+
    | coefficient       | float    | overlap matrix element                    |
    +-------------------+----------+-------------------------------------------+
    | frame             | category | non-unique integer (req.)                 |
    +-------------------+----------+-------------------------------------------+
    """
    _columns = ['chi1', 'chi2', 'coefficient']
    _cardinal = ('frame', np.int64)
    _index = 'index'

    def square(self, frame=0):
        """Returns a square dataframe of the density matrix."""
        denvec = self[self['frame'] == frame]['coefficient'].values
        square = pd.DataFrame(density_as_square(denvec))
        square.index.name = 'chi1'
        square.columns.name = 'chi2'
        return square

    @classmethod
    def from_momatrix(cls, momatrix, occvec):
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
        cmat = momatrix.square().values
        chi1, chi2, dens, frame = density_from_momatrix(cmat, occvec)
        return cls.from_dict({'chi1': chi1, 'chi2': chi2,
                              'coefficient': dens, 'frame': frame})
