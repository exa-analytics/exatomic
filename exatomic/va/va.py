# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Vibrational Averaging
#########################
Collection of classes for VA program
"""
import numpy as np
import pandas as pd
import csv
import os
from exa.util.units import Length
from exa.util.utility import mkp

_gauss_template='''{link0}
{route}

{title}

{charge} {mult}
'''
float_format = '%    .8f'
class GenInput:
    """
    Supporting class for Vibrational Averaging that will generate input files
    for a selected program under a certain displacement parameter.

    Computes displaced coordinates for all available normal modes from the equilibrium
    position by using the displacement vector components contained in the
    :class:`~exatomic.atom.Frequency` dataframe. It will scale these displacements to a
    desired type defined by the user with the delta_type keyword. For more information
    on this keyword see the documentation on the
    :class:`~exatomic.va.va.GenInputs.gen_delta` function.

    Args:
        uni (:class:`~exatomic.Universe`): Universe object containg pertinent data
        delta_type (int): Integer value to define the type of delta parameter to use
    """

    def _gen_delta(self, freq, delta_type):
        """
        Function to compute the delta parameter to be used for the maximum distortion
        of the molecule along the normal mode.

        When delta_type = 0 we normalize the displacments to have a maximum of 0.04 Bohr
        on each normal mode.

        When delta_type = 1 we normalize all atomic displacements along all normal modes
        to have a global average displacement of 0.04 Bohr.

        When delta_type = 2 we normalize each displacement so every atom has a maximum
        displacement of 0.04 Bohr on every normal mode.

        Args:
            freq (:class:`exatomic.atom.Frequency`): Frequency dataframe
            delta_type (int): Integer value to define the type of delta parameter to use
        """
        # average displacement of 0.04 bohr for each normal mode
        nat = len(freq['label'].drop_duplicates())
        nmode = len(freq['freqdx'].drop_duplicates())
        freqdx = freq['freqdx'].values
        if delta_type == 0:
            # Code using for loops
            # a = np.linalg.norm(freq[['dx', 'dy', 'dz']].
            #                                    values, axis=1, ord=2)
            # vec_sum = []
            # for i in range(0,len(a),3):
            #     vec_sum.append([0])
            #     for j in range(i,i+3):
            #         vec_sum[-1] += a[j]
            # print(a)
            # print(vec_sum)
            d = freq.groupby(['freqdx', 'frame']).apply(
                lambda x: np.sum(np.linalg.norm(
                    x[['dx', 'dy', 'dz']].values, axis=1))).values
            delta = 0.04 * nat / d
            delta = np.repeat(delta, nat)
    
        # global avrage displacement of 0.04 bohr for all atom displacements
        elif delta_type == 1:
            d = np.sum(np.linalg.norm(
                freq[['dx', 'dy', 'dz']].values, axis=1))
            delta = 0.04 * nat * nmode / (np.sqrt(3) * d)
            delta = np.repeat(delta, nat*nmode)
    
        # maximum displacement of 0.04 bohr for any atom in each normal mode
        elif delta_type == 2:
            d = freq.groupby(['freqdx', 'frame']).apply(lambda x:
                np.amax(abs(np.linalg.norm(x[['dx', 'dy', 'dz']].values, axis=1)))).values
            delta = 0.04 / d
            delta = np.repeat(delta, nat)
        self.delta = pd.DataFrame.from_dict({'delta': delta, 'freqdx': freqdx})
        #self.delta['delta'] *= Length['Angstrom', 'au']

    def _gen_displaced(self, freq, atom, fdx):
        # get needed data from dataframes
        eqcoord = atom[['x', 'y', 'z']].values
        symbols = atom['symbol'].values
        znums = atom['Zeff'].values
        if fdx == -1:
            freq_g = freq.copy()
        else:
            freq_g = freq.groupby('freqdx').filter(lambda x: fdx in
                                                    x['freqdx'].drop_duplicates().values).copy()
        disp = freq_g[['dx','dy','dz']].values
        modes = freq_g['frequency'].drop_duplicates().values
        nat = len(eqcoord)
        freqdx = freq_g['freqdx'].drop_duplicates().values
        tnmodes = len(freq['freqdx'].drop_duplicates())
        nmodes = len(freqdx)
        # chop all values less than 1e-6
        eqcoord[abs(eqcoord) < 1e-6] = 0.0
        # get delta values for wanted frequencies
        if fdx == -1:
            delta = self.delta['delta'].values
        else:
            delta = self.delta.groupby('freqdx').filter(lambda x:
                                      fdx in x['freqdx'].drop_duplicates().values)['delta'].values
        # calculate displaced coordinates in positive and negative directions
        disp_pos = np.tile(np.transpose(eqcoord), nmodes) + np.multiply(np.transpose(disp), delta)
        disp_neg = np.tile(np.transpose(eqcoord), nmodes) - np.multiply(np.transpose(disp), delta)
        # for now we comment this out so that we can just generate the necessary files in the 
        # format of the original program
#        full = np.concatenate((np.transpose(disp_neg), eqcoord, np.transpose(disp_pos)), axis=0)
#        # generate frequency indexes
#        # negative values are for displacement in negative direction
#        freqdx = [i for i in range(-nmodes, nmodes+1, 1)]
        full = np.concatenate((eqcoord, np.transpose(disp_pos), np.transpose(disp_neg)), axis=0)
        freqdx = [i+1+tnmodes*j for j in range(0,2,1) for i in freqdx]
        freqdx = np.concatenate(([0],freqdx))
        freqdx = np.repeat(freqdx, nat)
        modes = np.repeat(np.concatenate(([0],modes,modes)), nat)
        symbols = np.tile(symbols, 2*nmodes+1)
        znums = np.tile(znums, 2*nmodes+1)
        # create dataframe
        df = pd.DataFrame(full, columns=['x', 'y', 'z'])
        df['freqdx'] = freqdx
        df['Z'] = znums
        df['symbols'] = symbols
        df['modes'] = modes
        self.displacements = df

    def gen_gauss_inputs(self, path, routeg, routep, charge=0, mult=1, link0=''):
        """
        Method to write the displacements given in the displacements class variable to a
        gaussian input file. This writes a gradient (confg*.inp) and property (confp*.inp)
        files. As such, routeg and routep must be defined separately.

        Args:
            path (str): path to where the files will be written
            routeg (str): gaussian route input for gradient calculation
            routep (str): gaussian route input for property calculation
            charge (int): charge of molecular system
            mult (int): spin multiplicity of molecular system
            link0 (str): link0 commands for gaussian
        """
        grouped = self.displacements.groupby('freqdx')
        freqdx = self.displacements['freqdx'].drop_duplicates().values
        n = len(str(max(freqdx)))
        for fdx in freqdx:
            grad_file = 'confg'+str(fdx).zfill(n)+'.inp'
            prop_file = 'confo'+str(fdx).zfill(n)+'.inp'
            with open(mkp(path, grad_file), 'w') as g:
                xyz = grouped.get_group(fdx)[['symbols', 'x', 'y', 'z']]
                g.write(_gauss_template.format(link0=link0, route=routeg, 
                        title=str(fdx)+' gradient', charge=charge, mult=mult))
                xyz['x'] *= Length['au', 'Angstrom']
                xyz['y'] *= Length['au', 'Angstrom']
                xyz['z'] *= Length['au', 'Angstrom']
                xyz.to_csv(g, header=False, index=False, sep=' ', float_format=float_format,
                            quoting=csv.QUOTE_NONE, escapechar=' ')
                g.write('\n')
            with open(mkp(path, prop_file), 'w') as p:
                xyz = grouped.get_group(fdx)[['symbols', 'x', 'y', 'z']]
                p.write(_gauss_template.format(link0=link0, route=routep,
                        title=str(fdx)+' property', charge=charge, mult=mult))
                xyz['x'] *= Length['au', 'Angstrom']
                xyz['y'] *= Length['au', 'Angstrom']
                xyz['z'] *= Length['au', 'Angstrom']
                xyz.to_csv(p, header=False, index=False, sep=' ', float_format=float_format,
                            quoting=csv.QUOTE_NONE, escapechar=' ')
                p.write('\n')

    def __init__(self, uni, delta_type=0, fdx=-1, *args, **kwargs):
        if "_frequency" not in vars(uni):
            raise AttributeError("Frequency dataframe cannot be found in universe")
        freq = uni.frequency.copy()
        atom = uni.atom.copy()
        self._gen_delta(freq, delta_type)
        self._gen_displaced(freq, atom, fdx)
        
