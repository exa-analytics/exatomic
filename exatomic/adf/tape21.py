# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
ADF TAPE21 ASCII converted output
######################################
this module provides the primary (user facing) parser for an ASCII
converted TAPE21 file from ADF
"""

from exatomic.exa.core.container import TypedMeta
from exatomic.exa.core.editor import Editor
from exatomic.core.atom import Atom, Frequency
from exatomic.core.gradient import Gradient
from exatomic.core.tensor import JCoupling, NMRShielding
from exatomic.base import z2sym, sym2isomass
import numpy as np
import pandas as pd
import six

class MissingSection(Exception):
    pass

def _get_isomass(symbol):
    mapper = sym2isomass(symbol)
    mass = list(map(mapper.get, symbol))
    mass = np.repeat(mass, 3).astype(float)
    return mass

class Tape21Meta(TypedMeta):
    atom = Atom
    frequency = Frequency
    gradient = Gradient
    j_coupling = JCoupling
    nmr_shielding = NMRShielding

class Tape21(six.with_metaclass(Tape21Meta, Editor)):
    '''
    Parser for ADF Tape21 that have been converted to an ASCII file with
    their dmpkf utility.

    **All properties are parsed based on the input order.**

    Note:
        This is not yet tested for ADF versions newer than 2017.
    '''

    @staticmethod
    def rmass_mwc(data, symbol):
        '''
        Calculate the reduced masses from the mass-weighted normal modes. With
        the equation,

        .. math::
            \\mu_i = \\left(\\sum_k^{3N} \\left(\\frac{l_{MWCk,i}}
                                                {\\sqrt{m_k}}\\right)^2\\right)^{-1}

        Note:
            This assumes that the normal modes have already been placed in the
            :code:`['dx', 'dy', 'dz']` columns.

        Args:
            data (:class:`pandas.DataFrame`): Data frame the has the mass-weighted
                                              normal modes.
            symbol (:obj:`list`): List-like object that has the atomic symbols.

        Returns:
            r_mass (:class:`numpy.ndarray`): Array containing the calculated reduced
                                             masses.
        '''
        cols = ['dx', 'dy', 'dz']
        mass = _get_isomass(symbol)
        mass = mass.reshape(data[cols].shape)
        disps = data[cols].values
        r_mass = np.sum(np.square(disps)/mass)
        r_mass = 1/r_mass
        return r_mass

    @staticmethod
    def rmass_cart(data, symbol):
        '''
        Calculate the reduced masses from the normalized non-mass-weighted cartesian
        normal modes. With the equation,

        .. math::
            \\mu_i = \\left(\\sum_k^{3N} l_{CARTk,i}^2\\right)^{-1}

        Note:
            This assumes that the normal modes have already been placed in the
            :code:`['dx', 'dy', 'dz']` columns.

        Args:
            data (:class:`pandas.DataFrame`): Data frame the has the non-mass-weighted
                                              cartesian normal modes.
            symbol (:obj:`list`): List-like object that has the atomic symbols.

        Returns:
            r_mass (:class:`numpy.ndarray`): Array containing the calculated reduced
                                             masses.
        '''
        cols = ['dx', 'dy', 'dz']
        # get the isotopic masses of the unique atoms
        mass = _get_isomass(symbol)
        mass = mass.reshape(data[cols].shape)
        disps = data[cols].values
        norms = np.linalg.norm(disps*np.sqrt(mass))
        norms = 1/norms
        disps *= norms
        r_mass = np.sum(np.square(disps))
        r_mass = 1/r_mass
        return r_mass

    def _intme(self, fitem, idx=0):
        return int(self[fitem[idx]+1].split()[0])

    def _dfme(self, fitem, dim, idx=0):
        start = fitem[idx] + 2
        col = min(len(self[start].split()), dim)
        stop = np.ceil(start + dim / col).astype(np.int64)
        return self.pandas_dataframe(start, stop, col).stack().values

    def parse_frequency(self, cart=True):
        '''
        ADF frequency parser.

        Note:
            This will toss a warning if it cannot find the mass-weighted normal modes
            which must be used to generate the displaced structures for vibrational
            averaging. Also, it will be unable to calculate the reduced masses as it will
            have normalized cartesian coordinates where it expects normalized
            mass-weighted cartesian normal modes.

        Args:
            cart (:obj:`bool`, optional): Parse the normalized cartesian coordinates or
                                          the mass-weighted normal modes. Defaults to
                                          :code:`True`.
        '''
        # search flags
        _renorm = "NormalModes_RAW"
        _recartnorm = "Normalmodes"
        _refreq = "Frequencies"
        _refreqexc = r"\bFrequencies\b"
        _rekey = r"\bFreq\b"
        found = self.find(_refreq, _renorm, _recartnorm, keys_only=True)
        key = self.regex(_rekey, _refreqexc, keys_only=True)
        # need to do this to ensure that we only match the data in the Freq block
        found_freq = []
        for k in key[_rekey]:
            for f in found[_refreq]:
                if f-1 == k:
                    found_freq.append(f)
                    break
        if not found_freq:
            return
        found[_refreq] = found_freq
        if not found[_refreq]:
            return
        if not hasattr(self, 'atom'):
            self.parse_atom()
        # get the number of atoms
        nat = self.atom.last_frame.shape[0]
        # get the frequencies
        freq = self._dfme(found[_refreq], nat*3)
        # find where the frequencies are zero
        # these should be the ones that ADF determines to be translations and rotations
        # TODO: need a test case with one imaginary frequency
        low = freq != 0
        # get only the ones that are non-zero
        freq = freq[low]
        nmodes = freq.shape[0]
        freq = np.repeat(freq, nat)
        if found[_renorm] and not cart:
            # get the mass-weighted normal modes
            ndisps = int(self[found[_renorm][0]+1].split()[0])
            normalmodes = self._dfme(np.array(found[_renorm]), ndisps, idx=0)
        elif found[_recartnorm] and cart:
            # get the non-mass-weighted normal modes and toss warning
            ndisps = int(self[found[_recartnorm][0]+1].split()[0])
            normalmodes = self._dfme(np.array(found[_recartnorm]), ndisps, idx=0)
        else:
            raise MissingSection("There was an issue reading the file. Could " \
                                 +"not find the secions 'NormalModes_RAW', " \
                                 +"or 'Normalmodes'. Contents of what was " \
                                 +"found in the file {}".format(found))
        # get the vibrational modes in the three cartesian directions
        # the loop is neede in case there are any negative modes
        # because then the normal mode displacements for the negative mode
        # are listed first and we need those
        dx = []
        dy = []
        dz = []
        for idx in np.where(low)[0]:
            dx.append(normalmodes[idx*nat*3+0:(idx+1)*nat*3+0:3])
            dy.append(normalmodes[idx*nat*3+1:(idx+1)*nat*3+1:3])
            dz.append(normalmodes[idx*nat*3+2:(idx+1)*nat*3+2:3])
        # flatten arrays to vectors
        dx = np.array(dx).flatten()
        dy = np.array(dy).flatten()
        dz = np.array(dz).flatten()
        freqdx = np.repeat(range(nmodes), nat)
        label = np.tile(self.atom['label'], nmodes)
        symbol = np.tile(self.atom['symbol'], nmodes)
        # put the data together
        df = pd.DataFrame({'dx': dx, 'dy': dy, 'dz': dz, 'frequency': freq,
                           'freqdx': freqdx})
        # calculate the reduced masses
        if not cart:
            r_mass = df.groupby(['freqdx']).apply(self.rmass_mwc,
                                                  self.atom['symbol']).values
        else:
            r_mass = df.groupby(['freqdx']).apply(self.rmass_cart,
                                                  self.atom['symbol']).values
        df['r_mass'] = np.repeat(r_mass, nat)
        df['symbol'] = symbol
        df['label'] = label
        # TODO: find out if this is stored in the file anywhere
        df['ir_int'] = 0
        df['frame'] = 0
        self.frequency = df

    def parse_atom(self, input_order=False):
        '''
        Parse the atom table.

        Args:
            input_order (:obj:`bool`, optional): Parse the atom table
                in the input order format. Defaults to :code:`False`.
        '''
        # search flags
        _reinpatom = "xyz InputOrder"
        _reordatom = "xyz"
        #_regeom = "Geometry"
        _reqtch = "qtch"
        _rentyp = "ntyp"
        _renqptr = "nqptr"
        _reinporder = "atom order index"
        _remass = "mass"
        found = self.find(_reinpatom, _reordatom, _reqtch, _rentyp,
                          _renqptr, _reinporder, _remass, keys_only=True)
        if input_order:
            _reatom = found[_reinpatom]
        else:
            idx = 0
            tmp = found[_reordatom]
            while self[tmp[idx]-1].strip() != 'Geometry': idx += 1
            _reatom = [tmp[idx]]
        ncoords = self._intme(_reatom)
        coords = self._dfme(_reatom, ncoords)
        x = coords[::3]
        y = coords[1::3]
        z = coords[2::3]
        # get the number of atom types
        ntyp = int(self[found[_rentyp][1]+2].split()[0])
        # get the charges for each atom type
        qtch = self._dfme(found[_reqtch], ntyp)
        # get the span of each atom type
        nqptr = self._dfme(found[_renqptr], ntyp+1) - 1
        nat = nqptr.max()
        # get the znum vector from the ordered atom table
        zordered = np.zeros(nat)
        for n in range(ntyp):
            for idx in range(nqptr[n], nqptr[n+1]):
                zordered[idx] = qtch[n]
        if input_order:
            # convert to the input structure
            zinput = np.zeros(nat)
            input_order = self._dfme(found[_reinporder],
                                     nat*2).reshape(2, nat).astype(int) - 1
            # iterate over the input order array as this gives the
            # location of each atom type after
            # the re-ordering done in adf
            for od, inp in zip(input_order[0], range(nat)):
                zinput[inp] = zordered[od]
            Z = zinput.astype(int)
        else:
            Z = zordered
        set = np.array(list(range(nat)))
        symbol = pd.Series(Z).map(z2sym)
        # put it all together
        df = pd.DataFrame.from_dict({'symbol': symbol, 'set': set,
                                     'label': set, 'x': x, 'y': y,
                                     'z': z, 'Z': Z, 'frame': 0})
        self.atom = df

    def parse_gradient(self, input_order=False):
        ''' Parse the gradients in the input order. '''
        # search flags
        _reinpgrad = "Gradients_InputOrder"
        _refrggrad = "Gradients_CART"
        _reinporder = "atom order index"
        found = self.find(_reinpgrad, _refrggrad, keys_only=True)
        if not found[_refrggrad]:
            return
        if input_order:
            if found[_reinpgrad]:
                _regrad = _reinpgrad
            else:
                msg = "Could not find the 'Gradients_InputOrder'" \
                      +"section."
                raise MissingSection(msg)
        else:
            _regrad = _refrggrad
        # get the atom frame with the selected input_order flag
        # will overwrite what was previously parsed
        self.parse_atom(input_order=input_order)
        symbol = self.atom.last_frame['symbol'].values
        Z = self.atom.last_frame['Z'].values.astype(int)
        # get the gradients
        ngrad = self._intme(np.array(found[_regrad]))
        grad = self._dfme(np.array(found[_regrad]), ngrad)
        x = grad[::3]
        y = grad[1::3]
        z = grad[2::3]
        atom = list(range(len(x)))
        df = pd.DataFrame.from_dict({'Z': Z, 'atom': atom, 'fx': x,
                                     'fy': y, 'fz': z, 'symbol': symbol,
                                     'frame': 0})
        df = df[['atom', 'Z', 'fx', 'fy', 'fz', 'symbol', 'frame']]
        self.gradient = df

    def parse_nmr_shielding(self):
        ''' Parse the NMR shielding tensors in the input order. '''
        _reiso = "NMR Shieldings InputOrder"
        _retensor = "NMR Shielding Tensor InputOrder"
        found = self.find(_reiso, _retensor, keys_only=True)
        if not found[_reiso]:
            return
        if not hasattr(self, 'atom'):
            self.parse_atom(input_order=True)
        nshield = self._intme(found[_reiso])
        shielding = self._dfme(found[_reiso], nshield)
        ntens = self._intme(found[_retensor])
        tensor = self._dfme(found[_retensor], ntens)
        tensor = tensor.reshape(nshield, 9)
        zeros = list(map(lambda x: all(x != 0), tensor))
        requested = np.where(zeros)[0]
        tensor = tensor[requested]
        shielding = shielding[requested]
        #requested = np.where(shielding != 0)[0]
        cols = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        df = pd.DataFrame(tensor, columns=cols)
        df['isotropic'] = shielding
        df['atom'] = requested
        df['symbol'] = self.atom.last_frame.iloc[requested]['symbol'].values
        df['label'] = 'nmr_shielding'
        df['frame'] = 0
        self.nmr_shielding = df

    def parse_j_coupling(self):
        ''' Parse the J Coupling in the Cartesian representation. '''
        _reiso = "NMR Coupling J const InputOrder"
        _retensor = "NMR Coupling J tens InputOrder"
        found = self.find(_reiso, _retensor, keys_only=True)
        if not found[_reiso]:
            return
        if not hasattr(self, 'atom'):
            self.parse_atom(input_order=True)
        ncoupl = self._intme(found[_reiso])
        natom = np.sqrt(ncoupl)
        coupling = self._dfme(found[_reiso], ncoupl)
        ntens = self._intme(found[_retensor])
        tensor = self._dfme(found[_retensor], ntens)
        requested = np.where(coupling != 0)[0]
        tensor = tensor.reshape(ncoupl, 9)[requested]
        cols = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        df = pd.DataFrame(tensor, columns=cols)
        atoms = np.transpose(list(map(lambda x: divmod(x, natom), requested)))
        df['isotropic'] = coupling[coupling != 0]
        df['atom'] = atoms[0].astype(int)
        symbols = self.atom.last_frame['symbol'].values
        if len(symbols) > natom:
            raise NotImplementedError("Cannot deal with more than one atom frame.")
        df['symbol'] = list(map(lambda x: symbols[x], df['atom'].values))
        df['pt_atom'] = atoms[1].astype(int)
        df['pt_symbol'] = list(map(lambda x: symbols[x], df['pt_atom'].values))
        df['label'] = 'j coupling'
        df['frame'] = 0
        self.j_coupling = df

