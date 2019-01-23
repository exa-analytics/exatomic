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
import glob
import re
from exa.util.constants import speed_of_light_in_vacuum as C, Planck_constant as H, \
                               Boltzmann_constant as KB
from exa.util.units import Length, Energy, Mass, Time
from exa.util.utility import mkp
from exatomic.core import Atom, Gradient
from exa import TypedMeta
import warnings
warnings.simplefilter("default")

def get_data(path, attr, soft, f_end='', f_start='', sort_index=['']):
    # TODO: Make something so that we do not have to set the type of output parser by default
    #       allow the user to specify which it is based on the file.
    #       Consider just using soft as an input of a class
    if not isinstance(sort_index, list):
        raise TypeError("Variable sort_index must be of type list")
    if not hasattr(soft, attr):
        raise NotImplementedError("parse_{} is not a method of {}".format(attr, soft))
    files = glob.glob(path)
    array = []
    for file in files:
        if file.split('/')[-1].endswith(f_end) and file.split('/')[-1].startswith(f_start):
            ed = soft(file)
            try:
                df = getattr(ed, attr)
            except AttributeError:
                raise AttributeError("The property {} cannot be found in output {}".format(
                                                                                        attr, file))
            fdx = list(map(int, re.findall('\d+', file.split('/')[-1].replace(
                                                                   f_start, '').replace(f_end, ''))))
            df['file'] = np.tile(fdx, len(df))
        else:
            continue
        array.append(df)
    cdf = pd.concat([arr for arr in array])
    if sort_index[0] == '':
        if 'file' in cdf.columns.values:
            if 'label' in cdf.columns.values or 'atom' in cdf.columns.values:
                try:
                    cdf.sort_values(by=['file', 'label'], inplace=True)
                except KeyError:
                    cdf.sort_values(by=['file', 'atom'], inplace=True)
            else:
                warnings.warn("Sorting only by file label on DataFrame. Be careful if there is "+ \
                              "some order dependent function that is being used later based off"+ \
                              " this output.", Warning)
                cdf.sort_values(by=['file'], inplace=True)
    else:
        try:
            cdf.sort_values(by=sort_index, inplace=True)
        except KeyError:
            raise KeyError("Please make sure that the keys {} exist in the dataframe "+ \
                                        "created by {}.parse_{}.".format(sort_values, soft, attr))
    cdf.reset_index(drop=True, inplace=True)
    return cdf

def gen_delta(freq, delta_type, disp=None):
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
    freqdx = freq['freqdx'].drop_duplicates().values
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
    #    delta = np.repeat(delta, nat)

    # global avrage displacement of 0.04 bohr for all atom displacements
    elif delta_type == 1:
        d = np.sum(np.linalg.norm(
            freq[['dx', 'dy', 'dz']].values, axis=1))
        delta = 0.04 * nat * nmode / (np.sqrt(3) * d)
        delta = np.repeat(delta, nmode)
    #    delta = np.repeat(delta, nat*nmode)

    # maximum displacement of 0.04 bohr for any atom in each normal mode
    elif delta_type == 2:
        d = freq.groupby(['freqdx', 'frame']).apply(lambda x:
            np.amax(abs(np.linalg.norm(x[['dx', 'dy', 'dz']].values, axis=1)))).values
        delta = 0.04 / d
    #    delta = np.repeat(delta, nat)
    elif delta_type == 3:
        if disp is not None:
            delta = np.repeat(disp, nmode)
        else:
            raise ValueError("Must provide a displacement value through the disp variable for delta_type = 3")
    return pd.DataFrame.from_dict({'delta': delta, 'freqdx': freqdx})

class GenMeta(TypedMeta):
    disp = Atom
    delta = pd.DataFrame
    atom = Atom

class GenInput(metaclass = GenMeta):
    """
    Supporting class for Vibrational Averaging that will generate input files
    for a selected program under a certain displacement parameter.

    Computes displaced coordinates for all available normal modes from the equilibrium
    position by using the displacement vector components contained in the
    :class:`~exatomic.atom.Frequency` dataframe. It will scale these displacements to a
    desired type defined by the user with the delta_type keyword. For more information
    on this keyword see the documentation on the
    :class:`~exatomic.va.gen_delta` function.

    We can also define a specific normal mode or a list of normal modes that are of
    interest and generate displaced coordinates along those specific modes rather
    than all of the normal modes. Note that we use python indexing so the first
    normal mode corresponds to the index of 0. This is highly recommended if applicable
    as it may reduce number of computations and memory usage significantly.

    Args:
        uni (:class:`~exatomic.Universe`): Universe object containg pertinent data
        delta_type (int): Integer value to define the type of delta parameter to use
        fdx (int or list): Integer or list parameter to only displace along the
                           selected normal modes
        disp (float): Floating point value to set a specific displacement delta
                      parameter. Must be used with delta_type=3
    """

    _tol = 1e-6

    def _gen_displaced(self, freq, atom, fdx):
        """
        Function to generate displaced coordinates for each selected normal mode.
        We scale the displacements by the selected delta value in the positive and negative
        directions. We generate an array of coordinates that are put into a dataframe to
        write them to a file input for later evaluation.

        Note:
            The index 0 is reserved for the optimized coordinates, the equilibrium geometry.
            The displaced coordinates in the positive direction are given an index from
            1 to tnmodes (total number of normal modes).
            The displaced coordinates in the negative direction are given an index from
            tnmodes to 2*tnmodes.
            In an example with 39 normal modes 0 is the equilibrium geometry, 1-39 are the
            positive displacements and 40-78 are the negative displacements.
            nmodes are the number of normal modes that are selected. tnmodes are the total
            number of normal modes for the system.

        Args:
            freq (:class:`exatomic.atom.Frequency`): Frequency dataframe
            atom (:class:`exatomic.atom.Atom`): Atom dataframe
            fdx (int or list): Integer or list parameter to only displace along the
                               selected normal modes
        """
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
        eqcoord[abs(eqcoord) < self._tol] = 0.0
        # get delta values for wanted frequencies
        try:
            if fdx == -1:
                delta = self.delta['delta'].values
            elif -1 not in fdx:
                delta = self.delta.groupby('freqdx').filter(lambda x:
                                      fdx in x['freqdx'].drop_duplicates().values)['delta'].values
            else:
                raise TypeError("fdx must be a list of integers or a single integer")
            #if len(delta) != tnmodes:
            #    raise ValueError("Inappropriate length of delta. Passed a length of {} "+
            #                     "when it should have a length of {}. One value for each "+
            #                     "normal mode.".format(len(delta), tnmodes))
            #else:
            #    delta = np.repeat(delta, nat)
            delta = np.repeat(delta, nat)
        except AttributeError:
            raise AttributeError("Please compute self.delta first")
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
        frame = np.zeros(len(znums)).astype(np.int64)
        # create dataframe
        df = pd.DataFrame(full, columns=['x', 'y', 'z'])
        df['freqdx'] = freqdx
        df['Z'] = znums
        df['symbol'] = symbols
        df['modes'] = modes
        df['frame'] = frame
        return df

#    def gen_gauss_inputs(self, path, routeg, routep, charge=0, mult=1, link0=''):
#        """
#        Method to write the displacements given in the displacements class variable to a
#        gaussian input file. This writes a gradient (confg*.inp) and property (confp*.inp)
#        files. As such, routeg and routep must be defined separately.
#
#        Args:
#            path (str): path to where the files will be written
#            routeg (str): gaussian route input for gradient calculation
#            routep (str): gaussian route input for property calculation
#            charge (int): charge of molecular system
#            mult (int): spin multiplicity of molecular system
#            link0 (str): link0 commands for gaussian
#        """
#        grouped = self.disp.groupby('freqdx')
#        freqdx = self.disp['freqdx'].drop_duplicates().values
#        n = len(str(max(freqdx)))
#        for fdx in freqdx:
#            grad_file = 'confg'+str(fdx).zfill(n)+'.inp'
#            prop_file = 'confo'+str(fdx).zfill(n)+'.inp'
#            with open(mkp(path, grad_file), 'w') as g:
#                xyz = grouped.get_group(fdx)[['symbol', 'x', 'y', 'z']]
#                g.write(_gauss_template.format(link0=link0, route=routeg,
#                        title=str(fdx)+' gradient', charge=charge, mult=mult))
#                xyz['x'] *= Length['au', 'Angstrom']
#                xyz['y'] *= Length['au', 'Angstrom']
#                xyz['z'] *= Length['au', 'Angstrom']
#                xyz.to_csv(g, header=False, index=False, sep=' ', float_format=float_format,
#                            quoting=csv.QUOTE_NONE, escapechar=' ')
#                g.write('\n')
#            with open(mkp(path, prop_file), 'w') as p:
#                xyz = grouped.get_group(fdx)[['symbol', 'x', 'y', 'z']]
#                p.write(_gauss_template.format(link0=link0, route=routep,
#                        title=str(fdx)+' property', charge=charge, mult=mult))
#                xyz['x'] *= Length['au', 'Angstrom']
#                xyz['y'] *= Length['au', 'Angstrom']
#                xyz['z'] *= Length['au', 'Angstrom']
#                xyz.to_csv(p, header=False, index=False, sep=' ', float_format=float_format,
#                            quoting=csv.QUOTE_NONE, escapechar=' ')
#                p.write('\n')

    def gen_inputs(self, comm, soft):
        """
        Method to write the displaced coordinates as an input for the quantum code program
        of choice. Currently only the following input generators have been tested with this
        generalized input generator:
            - :class:`exatomic.nwchem.Input.from_universe`
            - :class:`exatomic.gaussian.Input.from_universe`
        More to come as the need is met.
        This code will use the software input and iterate over all available frequency
        indexes sending the data to the specified input generator. We have designed the code
        to create the self.atom attribute as it gets called by input generators.

        Note:
            comm is currently supported as a single dictionary, i.e. the gradient and property
            claculation will happen within the same script. The hope is that we can extend this
            so a user can calculate the property and gradient separately. One case that this is
            applicable to is if the user must use a different functional/basis for one of the
            calculations.
            The format is:
                - {[keys of specified software]: [values]}
            As an example this would be the comm input for a SP calculation at the
            B3LYP/6-31G* level of theory with NProc=4 and Chk=test.chk for
            exatomic.gaussian.Input.from_universe
                - {'link0': {'NProc': 4, 'Chk': 'test.chk'}, 'route': '#P B3LYP/6-31G* SP',
                   'writedir': dir_path, 'name': 'filename'}
            For questions regarding the inputs needed for each input generator please refer
            to the docs of the specific input generator.

        Args:
            comm (dict): Dictionary containing all of the pertinent commands for the input
            soft (class instance): Software of choice for the input generation
        """
        grouped = self.disp.groupby('freqdx')
        freqdx = self.disp['freqdx'].drop_duplicates().values
        n = len(str(max(freqdx)))
        name = comm['name']
        for fdx in freqdx:
            comm['name'] = name+str(fdx).zfill(n)+'.inp'
            self.atom = grouped.get_group(fdx)
            soft(uni=self, **comm)

    def gen_slurm_inputs(self, path, sbatch, module, end_com=''):
        """
        Method to write slurm scripts to execute gradients and property calculations given
        the displaced coordinates.

        Method generates separate directories containing the slurm script for each calculation.
        Will need to submit with some external shell script.

        Need to define the module and sbatch variables to what is needed by each user. It was
        built like this to make it the most general and applicable to more than one type of
        quantum chemistry code.

        Args:
            path (str): path to where the directories will be generated and the inputs will
                        will be read from
            sbatch (dict): sbatch commands that are to be used for batch script execution
            module (str): multiline string that will contain the module loading and other
                          user specific variables
            end_com (str): commands to be placed at the end of the slurm script
        """
        _name = "{job}{int}.dir"
        _sbatch = "#SBATCH --{key}={value}"
        files = os.listdir(path)
        for file in files:
            if file.endswith(".inp") and file.startswith("confo"):
                fdx = file.replace("confo", "").replace(".inp", "")
                job = "jobo"
            elif file.endswith(".inp") and file.startswith("confg"):
                fdx = file.replace("confg", "").replace(".inp", "")
                job = "jobg"
            else:
                continue
            try:
                os.mkdir(path+_name.format(job=job, int=fdx))
                j_path = path+_name.format(job=job, int=fdx)
            except OSError:
                raise OSError("Failed to create directory {}".format(path+_name.format(
                                                                                job=job, int=fdx)))
            slurm = file.replace(".inp", ".slurm")
            with open(path+file, 'r') as f:
                with open(mkp(j_path, slurm), 'w') as j:
                    j.write("#!/bin/bash\n")
                    for key in sbatch.keys():
                        j.write(_sbatch.format(key=key, value=sbatch[key])+'\n')
                    j.write(_sbatch.format(key="job-name", value=file.replace(".inp", "")))
                    j.write(_sbatch.format(key="output", value=file.replace(".inp", ".out")))
                    j.write(module)
                    for line in f:
                        j.write(line)
                    j.write(end_com+'\n')

    @staticmethod
    def write_data_file(path, array, fn):
        with open(mkp(path, fn), 'w') as f:
            for item in array:
                f.write("{}\n".format(item))

    def to_va(self, uni, path):
        """
        Simple script to be able to use the vibaverage.exe program to calculate the needed
        parameters (temporary).

        Args:
            uni (:class:`~exatomic.Universe`): Universe object containg pertinent data
            path (str): path to where the *.dat files will be written to
        """
        freq = uni.frequency.copy()
        atom = uni.atom.copy()
        freq_ext = uni.frequency_ext.copy()
        # construct delta data file
        fn = "delta.dat"
        delta = self.delta['delta'].drop_duplicates().values
        self.write_data_file(path=path, array=delta, fn=fn)
        # construct smatrix data file
        fn = "smatrix.dat"
        smatrix = freq[['dx', 'dy', 'dz']].stack().values
        self.write_data_file(path=path, array=smatrix, fn=fn)
        # construct atom order data file
        fn = "atom_order.dat"
        atom_order = atom['symbol'].values
        self.write_data_file(path=path, array=atom_order, fn=fn)
        # construct reduced mass data file
        fn = "redmass.dat"
        redmass = freq_ext['r_mass'].values * Mass['au_mass', 'u']
        self.write_data_file(path=path, array=redmass, fn=fn)
        # construct eqcoord data file
        fn = "eqcoord.dat"
        eqcoord = atom[['x', 'y', 'z']].stack().values
        eqcoord *= Length['au', 'Angstrom']
        self.write_data_file(path=path, array=eqcoord, fn=fn)
        # construct frequency data file
        fn = "freq.dat"
        frequency = freq_ext['freq'].values * Energy['Ha', 'cm^-1']
        self.write_data_file(path=path, array=frequency, fn=fn)
        # construct actual displacement data file
        fn = "displac_a.dat"
        delta = np.repeat(self.delta['delta'].values, len(atom))
        disp = np.multiply(np.linalg.norm(np.transpose(freq[['dx','dy','dz']].values), axis=0),
                                                        delta)
        disp *= Length['au', 'Angstrom']
        freqdx = freq['freqdx'].drop_duplicates().values
        n = len(atom_order)
        with open(mkp(path, fn), 'w') as f:
            f.write("actual displacement in angstroms\n")
            f.write("atom normal_mode distance_atom_moves\n")
            for fdx in range(len(freqdx)):
                for idx in range(n):
                    f.write("{} {}\t{}\n".format(idx+1, fdx+1, disp[fdx*15+idx]))

    def write_grad_prop(self, path, grad, prop):
        """
        Simple function to write the gradient and property datafiles to the format needed for
        vibaverage.exe (temporary).

        The gradient and property dataframes must be from the single point calculations. We
        assume this by not grouping by the frame column.

        Args:
            path (str): path to where the *.dat files will be written
            grad (np.ndarray): 1D array of values from grad[['fx', 'fy', 'fz']].stack().values
            prop (np.ndarray): 1D array of values from prop[property].values
        """
        # construct gradient data file
        fn = "grad.dat"
        if isinstance(grad[0], np.ndarray):
            raise ValueError("grad array must be a 1D array")
        self.write_data_file(path=path, array=grad, fn=fn)
        # construct property data file
        fn = "prop.dat"
        if isinstance(prop[0], np.ndarray):
            raise ValueError("prop array must be a 1D array")
        self.write_data_file(path=path, array=prop, fn=fn)

    def __init__(self, uni, *args, **kwargs):
        if not hasattr(uni, 'frequency'):
            raise AttributeError("Frequency dataframe cannot be found in universe")
        delta_type = kwargs.pop("delta_type", 0)
        fdx = kwargs.pop("fdx", -1)
        disp = kwargs.pop("disp", None)
        freq = uni.frequency.copy()
        atom = uni.atom.copy()
        self.delta = gen_delta(freq, delta_type, disp)
        self.disp = self._gen_displaced(freq, atom, fdx)

class VAMeta(TypedMeta):
    grad_0 = Gradient
    grad_plus = Gradient
    grad_minus = Gradient
    gradient = Gradient

class VA(metaclass=VAMeta):
    """
    Administrator class for VA to perform all initial calculations of necessary variables to pass
    for calculations.
    """
    # TODO: look to speed up code in staticmethods with jit
    @staticmethod
    def _alpha_squared(alpha):
        n = len(alpha)
        alpha_squared = np.zeros(n).astype(np.complex128)
        #real_text = ''
        #imag_text = ''
        #cart = [0,1,2,4,5,8]
        for fdx in range(len(alpha)):
            # debug code
            #for idx, (real, imag) in enumerate(zip(np.real(alpha[fdx]), np.imag(alpha[fdx]))):
            #    if idx in cart:
            #        real_text = real_text+"\n{:.10f}".format(real)
            #        imag_text = imag_text+"\n{:.10f}".format(imag)
            sum = 0.0
            for al in range(3):
                for be in range(3):
                    sum += (1./9.)*(alpha[fdx][al*3+al]*np.conj(alpha[fdx][be*3+be]))
            alpha_squared[fdx] = sum
        #fn = open("/home/herbertl/jupyter-notebooks/vroa_data_files/dalpha_dq_real_file", 'w')
        #fn.write(real_text)
        #fn.close()
        #fn = open("/home/herbertl/jupyter-notebooks/vroa_data_files/dalpha_dq_imag_file", 'w')
        #fn.write(imag_text)
        #fn.close()
        return alpha_squared

    @staticmethod
    def _beta_alpha(alpha):
        beta_alpha = []
        for fdx in range(len(alpha)):
            sum = 0.0
            for al in range(3):
                for be in range(3):
                    sum += 0.5*(3*alpha[fdx][al*3+be]*np.conj(alpha[fdx][al*3+be])- \
                                alpha[fdx][al*3+al]*np.conj(alpha[fdx][be*3+be]))
            beta_alpha.append(sum)
        return beta_alpha

    @staticmethod
    def _beta_g_prime(alpha, g_prime):
        beta_g_prime = []
        #cart = [0,1,2,4,5,8]
        #real_text = ''
        #imag_text = ''
        for fdx in range(len(alpha)):
            # debug code
            #for idx, (real, imag) in enumerate(zip(np.real(g_prime[fdx]), np.imag(g_prime[fdx]))):
            #    real_text = real_text+"\n{:.10f}".format(real)
            #    imag_text = imag_text+"\n{:.10f}".format(imag)
            sum = 0.0
            for al in range(3):
                for be in range(3):
                    sum += 1j*0.5*(3*alpha[fdx][al*3+be]*np.conj(g_prime[fdx][al*3+be])- \
                                alpha[fdx][al*3+al]*np.conj(g_prime[fdx][be*3+be]))
            beta_g_prime.append(sum)
        #fn = open("/home/herbertl/jupyter-notebooks/vroa_data_files/dg_dq_real_file", 'w')
        #fn.write(real_text)
        #fn.close()
        #fn = open("/home/herbertl/jupyter-notebooks/vroa_data_files/dg_dq_imag_file", 'w')
        #fn.write(imag_text)
        #fn.close()
        return beta_g_prime

    @staticmethod
    def _beta_A(omega, alpha, A):
        beta_A = []
        epsilon = [[0,0,0,0,0,1,0,-1,0],[0,0,-1,0,0,0,1,0,0],[0,1,0,-1,0,0,0,0,0]]
        #cart = [8,17,26]
        real_text = ''
        imag_text = ''
        for fdx in range(len(alpha)):
            # debug code
            #for ndx in [0,4,8]:
            #    for idx in range(3):
            #        for jdx in range(idx*9+ndx,idx*9+ndx+4):
            #            real_text = real_text+"\n{:.10f}".format(np.real(A[fdx][jdx]))
            #            imag_text = imag_text+"\n{:.10f}".format(np.imag(A[fdx][jdx]))
            #            if jdx in cart:
            #                break
            sum = 0.0
            for al in range(3):
                for be in range(3):
                    for de in range(3):
                        for ga in range(3):
                            sum += 0.5*omega[fdx]*alpha[fdx][al*3+be]* \
                                        epsilon[al][de*3+ga]*np.conj(A[fdx][de*9+ga*3+be])
            beta_A.append(sum)
        #fn = open("/home/herbertl/jupyter-notebooks/vroa_data_files/dA_dq_real_file", 'w')
        #fn.write(real_text)
        #fn.close()
        #fn = open("/home/herbertl/jupyter-notebooks/vroa_data_files/dA_dq_imag_file", 'w')
        #fn.write(imag_text)
        #fn.close()
        return beta_A

    @staticmethod
    def _alpha_g_prime(alpha, g_prime):
        alpha_g_prime = []
        for fdx in range(len(alpha)):
            sum = 0.0
            for al in range(3):
                for be in range(3):
                    sum += alpha[fdx][al*3+al]*np.conj(g_prime[fdx][be*3+be])/9.
            alpha_g_prime.append(1j*sum)
        return alpha_g_prime

    @staticmethod
    def _calc_kp(lambda_0, lambda_p):
        '''
        Function to calculate the K_p value as given in equation 2 on J. Chem. Phys. 2007, 127, 134101.
        We assume the temperature to be 298.15 as a hard coded value. Must get rid of this in future
        iterations. The final units of the equation is in m^2.
        Input values lambda_0 and lambda_p must be in the units of m^-1
        '''
        # epsilon_0 = 1/(4*np.pi*1e-7*C**2)
        # another hard coded value
        temp = 298.15 # Kelvin
        boltz = 1.0/(1.0-np.exp(-H*C*lambda_p/(KB*temp)))
        constants = H * np.pi**2 / C
        variables = (lambda_0 - lambda_p)**4/lambda_p
        kp = 2 * variables * constants * boltz * (Length['au', 'm']**4 / Mass['u', 'kg'])
        return kp

    @staticmethod
    def _sum(df):
        # simple function to sum up the imaginary and real parts of the tensors in the roa dataframe
        # we use a np.complex128 data type to keep 64 bit precision on both the real and imaginary parts
        cols = np.array(['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'])
        # get the index values of the imaginary parts
        mask = df.groupby('type').get_group('imag').index.values
        # add the imaginary values
        value_complex = 1j*df.loc[mask, cols].astype(np.complex128).values
        # add the real values
        value_complex += df.loc[~df.index.isin(mask), cols].astype(np.complex128).values
        value_complex = value_complex.reshape(9,)
        return value_complex

    def vroa(self, uni, delta):
        if not hasattr(self, 'roa'):
            raise AttributeError("Please set roa attribute.")
        if not hasattr(uni, 'frequency_ext'):
            raise AttributeError("Please compute frequency_ext dataframe.")
        # we must remove the 0 index file as by default our displaced coordinate generator will
        # include these values and they have no significane in this code as of yet
        try:
            roa_0 = self.roa.groupby('file').get_group(0)
            idxs = roa_0.index.values
            roa = self.roa.loc[~self.roa.index.isin(idxs)]
        except KeyError:
            roa = self.roa.copy()
        grouped = roa.groupby(['label', 'file'])
        nmodes = len(uni.frequency_ext.index.values)

        # add the real and complex parts of the tensors
        complex_roa = grouped.apply(lambda x: self._sum(x))

        # get alpha G' and A tensors and divide by the reduced mass
        rmass = np.sqrt(uni.frequency_ext['r_mass'].values)
        A = complex_roa[['Ax','Ay','Az']]
        alpha = complex_roa['alpha']
        g_prime = complex_roa['g_prime']
        A = A.groupby('file').apply(lambda x: np.array([x.values[0],x.values[1],
                                                        x.values[2]]).flatten())
        # separate tensors into positive and negative displacements
        # highly dependent on the value of the index
        # we neglect the equilibrium coordinates
        # 0 corresponds to equilibrium coordinates
        # 1 - nmodes corresponds to positive displacements
        # nmodes+1 - 2*nmodes corresponds to negative displacements
        alpha_plus = np.divide(alpha.loc[range(1,nmodes+1)].values, rmass)
        alpha_minus = np.divide(alpha.loc[range(nmodes+1, 2*nmodes+1)].values, rmass)
        g_prime_plus = np.divide(g_prime.loc[range(1,nmodes+1)].values, rmass)
        g_prime_minus = np.divide(g_prime.loc[range(nmodes+1, 2*nmodes+1)].values, rmass)
        A_plus = np.divide(A.loc[range(1,nmodes+1)].values, rmass)
        A_minus = np.divide(A.loc[range(nmodes+1, 2*nmodes+1)].values, rmass)

        # generate derivatives by two point difference method
        # TODO: check all of these values to Movipac software
        dalpha_dq = np.divide((alpha_plus - alpha_minus), 2 * delta)
        dg_dq = np.divide((g_prime_plus - g_prime_minus), 2 * delta)
        dA_dq = [np.divide((A_plus[i] - A_minus[i]), 2 * delta[i]) for i in range(nmodes)]
        #self.dalpha_dq = pd.Series(dalpha_dq)
        #self.dg_dq = pd.Series(dg_dq)
        #self.dA_dq = pd.Series(dA_dq)

        # get frequencies
        frequencies = uni.frequency_ext['freq']

        # generate properties as shown on equations 5-9 in paper
        # J. Chem. Phys. 2007, 127, 134101
        alpha_squared = np.real(self._alpha_squared(dalpha_dq))
        beta_alpha = np.real(self._beta_alpha(dalpha_dq))
        beta_g_prime = np.imag(self._beta_g_prime(dalpha_dq, dg_dq))
        beta_A = np.real(self._beta_A(frequencies, dalpha_dq, dA_dq))
        alpha_g = np.imag(self._alpha_g_prime(dalpha_dq, dg_dq))
        self.alpha_squared = pd.Series(alpha_squared*Length['au', 'Angstrom']**4)
        self.beta_alpha = pd.Series(beta_alpha*Length['au', 'Angstrom']**4)
        self.beta_g_prime = pd.Series(beta_g_prime*Length['au', 'Angstrom']**4/
                                                            (C*Length['m', 'au']/Time['s','au']))
        self.beta_A = pd.Series(beta_A*Length['au', 'Angstrom']**4/
                                                            (C*Length['m', 'au']/Time['s','au']))
        self.alpha_g = pd.Series(alpha_g*Length['au', 'Angstrom']**4/
                                                            (C*Length['m', 'au']/Time['s','au']))
        ## hard coded value
        #lambda_0 = 1. / (514.5 * Length['nm', 'm']) #in wavenumbers (m^{-1})
        #warnings.warn("Hard coded value of lambda_0 in vroa. This is a value corresponding to an "+ \
        #             "Ar ion laser with wavelength of 514.5 nm. Must find a way to calculate this.",
        #             Warning)

        ## have to convert frequencies from Ha to m^-1 to match equations units
        #lambda_p = uni.frequency_ext['freq'].values * Energy['Ha', 'cm^-1'] / Length['cm', 'm']
        #kp = self._calc_kp(lambda_0, lambda_p)
        #backscat = kp * (45.0 * alpha_squared + 7.0 * beta_alpha) / 45.0

        # calculate VROA back scattering and forward scattering intensities
        backscat_vroa = 4./(C*Length['m', 'au']/Time['s','au'])*(24 * beta_g_prime + 8 * beta_A)
        forwscat_vroa = 4./(C*Length['m', 'au']/Time['s','au'])* \
                                                 (180 * alpha_g + 4 * beta_g_prime - 4 * beta_A)
        #self.backscat = pd.Series(backscat)
        self.backscat_vroa = pd.Series(backscat_vroa)
        self.forwscat_vroa = pd.Series(forwscat_vroa)

    def init_va(self, uni, delta=None):
        """
        This is a method to initialize all of the variables that will be needed to execute the VA
        program. As a sanity check we calculate the frequencies from the force constants. If we
        have any negative force constants the results may not be as reliable.

        Args:
            uni (:class:`~exatomic.Universe`): Universe object containg pertinent data from
                                               frequency calculation
        """
        if delta is None:
            delta_df = gen_delta(freq=uni.frequency.copy(), delta_type=2)
            delta = delta_df['delta'].values
        if not hasattr(self, "gradient"):
            raise AttributeError("Please set gradient attribute first")
        if not hasattr(uni, "frequency_ext"):
            raise AttributeError("Cannot find frequency extended dataframe in universe")
        if not hasattr(uni, "frequency"):
            raise AttributeError("Cannot find frequency dataframe in universe")
#        if not vroa:
#            if not hasattr(self, "property"):
#                raise AttributeError("Please set property attribute first")
#        else:
#            self.vroa(uni=uni, delta=delta)
        # check that all attributes to be used exist
        # group the gradients by file (normal mode)
        grouped = self.gradient.groupby('file')
        # get number of normal modes
        nmodes = len(uni.frequency_ext.index.values)
        # generate delta dataframe
        # TODO: make something so delta can be set
        #       possible issues are a user using a different type of delta
        # get gradient of the equilibrium coordinates
        grad_0 = grouped.get_group(0)
        # get gradients of the displaced coordinates in the positive direction
        grad_plus = grouped.filter(lambda x: x['file'].drop_duplicates().values in
                                                                        range(1,nmodes+1))
        # get number of selected normal modes
        # TODO: check stability of using this parameter
        snmodes = len(grad_plus['file'].drop_duplicates().values)
        # get gradients of the displaced coordinates in the negative direction
        grad_minus = grouped.filter(lambda x: x['file'].drop_duplicates().values in
                                                                        range(nmodes+1, 2*nmodes+1))
        # TODO: Check if we can make use of numba to speed up this code
        delfq_zero = uni.frequency.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda x:
                                    np.sum(np.multiply(grad_0[['fx', 'fy', 'fz']].values, x.values)))
        delfq_zero = np.tile(delfq_zero, nmodes).reshape(snmodes, nmodes)

        delfq_plus = grad_plus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                                uni.frequency.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                    np.sum(np.multiply(y.values, x.values))))
        delfq_plus.reset_index(drop=True, inplace=True)

        delfq_minus = grad_minus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                                uni.frequency.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                    np.sum(np.multiply(y.values, x.values))))
        delfq_minus.reset_index(drop=True, inplace=True)

        # get diagonal elements of respqective matrix
        diag_plus = np.diag(delfq_plus)
        diag_minus = np.diag(delfq_minus)
        diag_zero = np.diag(delfq_zero)

        # calculate force constants
        kqi   = np.divide(diag_plus - diag_minus, 2.0*delta)
        kqiii = np.divide(diag_plus - 2.0 * diag_zero + diag_minus, np.multiply(delta, delta))
        kqijj = np.divide(delfq_plus - 2.0 * delfq_zero + delfq_minus,
                                                    np.multiply(delta, delta).reshape(snmodes, 1))

        # convert force constants to reduced normal coordinate force constants
        redmass = uni.frequency_ext['r_mass'].values*Mass['u', 'au_mass']
        vqi = np.divide(kqi, redmass)
        vqijj = np.divide(kqijj, np.sqrt(np.power(redmass, 3)).reshape(snmodes,1))

        # TODO: Check if we want to exit the program if we get a negative force constant
        n_force_warn = vqi[vqi < 0.]

        # calculate frequencies
        calcfreq = np.sqrt(vqi)
        calcfreq *= Energy['Ha', 'cm^-1']
        self.calcfreq = pd.DataFrame.from_dict({'calc_freq': calcfreq,
                                      'real_freq': uni.frequency_ext['freq']*Energy['Ha', 'cm^-1']})


        # This is mainly for debug purposes
        # Will most likely eliminate most if not all of these class attributes
        #self.delfq_zero = pd.DataFrame(delfq_zero)
        #self.delfq_plus = pd.DataFrame(delfq_plus)
        #self.delfq_minus = pd.DataFrame(delfq_minus)
        #idx = uni.frequency['freqdx'].drop_duplicates().values
        #ndx = np.repeat(idx, nmodes)
        ##print(len(ndx))
        #jdx = np.tile(idx, nmodes)
        ##print(len(jdx))
        #self.kqi   = pd.DataFrame.from_dict({'idx': idx, 'kqi': kqi, 'calculated_vqi': uni.frequency_ext['f_const']})
        #self.kqiii = pd.DataFrame.from_dict({'idx': idx, 'kqiii': kqiii})
        #self.kqijj = pd.DataFrame(kqijj)
        #self.delta = delta_df
        #self.vqi = pd.DataFrame.from_dict({'idx': idx, 'vqi': vqi})
        #pd.options.display.float_format = '{:.6E}'.format
