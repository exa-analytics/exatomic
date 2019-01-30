# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
import pandas as pd
import numpy as np
from exa.util.utility import mkp
from exa.util.units import Length, Energy, Mass
from exatomic.core import Atom
from exa import TypedMeta
import warnings
warnings.simplefilter("default")

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

    When delta_typ = 3 the user can select a delta parameter to use with the disp
    keyword this will displace all normal modes by that delta parameter.

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

#    def gen_slurm_inputs(self, path, sbatch, module, end_com=''):
#        """
#        Method to write slurm scripts to execute gradients and property calculations given
#        the displaced coordinates.
#
#        Method generates separate directories containing the slurm script for each calculation.
#        Will need to submit with some external shell script.
#
#        Need to define the module and sbatch variables to what is needed by each user. It was
#        built like this to make it the most general and applicable to more than one type of
#        quantum chemistry code.
#
#        Args:
#            path (str): path to where the directories will be generated and the inputs will
#                        will be read from
#            sbatch (dict): sbatch commands that are to be used for batch script execution
#            module (str): multiline string that will contain the module loading and other
#                          user specific variables
#            end_com (str): commands to be placed at the end of the slurm script
#        """
#        _name = "{job}{int}.dir"
#        _sbatch = "#SBATCH --{key}={value}"
#        files = os.listdir(path)
#        for file in files:
#            if file.endswith(".inp") and file.startswith("confo"):
#                fdx = file.replace("confo", "").replace(".inp", "")
#                job = "jobo"
#            elif file.endswith(".inp") and file.startswith("confg"):
#                fdx = file.replace("confg", "").replace(".inp", "")
#                job = "jobg"
#            else:
#                continue
#            try:
#                os.mkdir(path+_name.format(job=job, int=fdx))
#                j_path = path+_name.format(job=job, int=fdx)
#            except OSError:
#                raise OSError("Failed to create directory {}".format(path+_name.format(
#                                                                                job=job, int=fdx)))
#            slurm = file.replace(".inp", ".slurm")
#            with open(path+file, 'r') as f:
#                with open(mkp(j_path, slurm), 'w') as j:
#                    j.write("#!/bin/bash\n")
#                    for key in sbatch.keys():
#                        j.write(_sbatch.format(key=key, value=sbatch[key])+'\n')
#                    j.write(_sbatch.format(key="job-name", value=file.replace(".inp", "")))
#                    j.write(_sbatch.format(key="output", value=file.replace(".inp", ".out")))
#                    j.write(module)
#                    for line in f:
#                        j.write(line)
#                    j.write(end_com+'\n')

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
