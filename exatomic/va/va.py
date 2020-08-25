# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Vibrational Averaging
#########################
Collection of classes for VA program
"""
import numpy as np
import pandas as pd
import glob
import re
import os
from exa.util.constants import (speed_of_light_in_vacuum as C, Planck_constant as H,
                               Boltzmann_constant as boltzmann)
from exa.util.units import Length, Energy, Mass, Time
from exatomic.core import Atom, Gradient, Polarizability
from exatomic.base import sym2z
from exa import TypedMeta
from .vroa_funcs import _sum, _make_derivatives, _forwscat, _backscat
import warnings
warnings.simplefilter("default")

def get_data(path, attr, soft, f_end='', f_start='', sort_index=None):
    '''
    This script is made to be able to extract data from many different files and
    compile them all into one dataframe. You can pass wildcards as an input in the
    path variable. We use the glob.glob package to get an array of the file that
    match the given f_start and f_end strings.

    Note:
        There is nothing built in to handle returning an empty dataframe at the
        moment.

    Args:
        path (str): String pointing to location of files
        attr (str): The attribute that you want to extract
        soft (class): Class that you want to use to extract the data
        f_end (str): String to match to the end of the filename
        f_start (str): String to match to the start of the filename
        sort_index (list): List of strings that are to be used to sort the compiled dataframe

    Returns:
        cdf (pandas.DataFrame): Dataframe that has all of the compiled data
    '''
    if sort_index is None: sort_index = ['']
    if not isinstance(sort_index, list):
        raise TypeError("Variable sort_index must be of type list")
    if not hasattr(soft, attr):
        raise NotImplementedError("parse_{} is not a method of {}".format(attr, soft))
    files = glob.glob(path)
    array = []
    for file in files:
        if file.split(os.sep)[-1].endswith(f_end) and file.split(os.sep)[-1].startswith(f_start):
            ed = soft(file)
            try:
                df = getattr(ed, attr)
            except AttributeError:
                print("The property {} cannot be found in output {}".format(attr, file))
                continue
            # We assume that the file identifier is an integer
            fdx = list(map(int, re.findall('\d+', file.split(os.sep)[-1].replace(
                                                                   f_start, '').replace(f_end, ''))))
            #fdx = float(file.split(os.sep)[-1].replace(f_start, '').replace(f_end, ''))
            df['file'] = fdx[0]
        else:
            continue
        array.append(df)
    cdf = pd.concat(array)
    # TODO: check if this just absolute overkill in error handling
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
            raise KeyError("Please make sure that the keys {} exist in the dataframe created by {}.parse_{}.".format(sort_values, soft, attr))
    cdf.reset_index(drop=True, inplace=True)
    return cdf

class VAMeta(TypedMeta):
    gradient = Gradient
    roa = Polarizability
    eff_coord = Atom

class VA(metaclass=VAMeta):
    """
    Class that contains all of the Vibrational Averaging methods. Currently we have implemented:
    Vibrational Raman Optical Activity (vroa), Zero-Point Vibrational Corrections (zpvc).

    Note:
        We do not have any code that will get the class attributes that are needed to execute
        the respective methods. We only check to make sure that the class attributes exist.
        The vroa method will look for the gradient and roa class attributes. The zpvc method will
        look for the gradient and property class attibutes. We recommend using the
        exatomic.va.va.get_data function to get the data as it will compress everything into
        one dataframe as is expected for the zpvc and vroa methods
    """
    # TODO: can probably use jit for this but it may not provide a significant speed up
    @staticmethod
    def raman_int_units(lambda_0, lambda_p, temp=None):
        '''
        Function to calculate the K_p value as given in equation 2 on J. Chem. Phys. 2007, 127, 134101.
        We assume the temperature to be 298.15 as a hard coded value. Must get rid of this in future
        iterations. The final units of the equation are in cm^2/sr which are said to be the units for
        the Raman intensities.

        Note:
            Input values lambda_0 and lambda_p must be in the units of m^-1

        Args:
            lambda_0 (float): Wavenumber value of the incident light
            lambda_1 (numpy.array): Wavenumber values of the vibrational modes
            temp (float): Value of the temperature of the experiment

        Returns:
            kp (numpy.array): Array with the values of the conversion units of length lambda_1.shape[0]
        '''
        if temp is None: temp=298.15
        boltz = 1.0/(1.0-np.exp(-H*C*lambda_p/(boltzmann*temp)))
        constants = H * np.pi**2 / C
        variables = (lambda_0 - lambda_p)**4/lambda_p
        kp = variables * constants * boltz * (Length['au', 'm']**4 / Mass['u', 'kg']) * 16 / 45. * Length['m', 'cm']**2
        print(kp, boltz, lambda_p)
        return kp

    @staticmethod
    def _check_file_continuity(df, prop, nmodes):
        files = df['file'].drop_duplicates()
        pos_file = files[files.isin(range(1,nmodes+1))]
        neg_file = files[files.isin(range(nmodes+1, 2*nmodes+1))]-nmodes
        intersect = np.intersect1d(pos_file.values, neg_file.values)
        diff = np.unique(np.concatenate((np.setdiff1d(pos_file.values, intersect),
                                         np.setdiff1d(neg_file.values, intersect)), axis=None))
        rdf = df.copy()
        if len(diff) > 0:
            print("Seems that we are missing one of the {} outputs for frequency {} ".format(prop, diff)+ \
                  "we will ignore the {} data for these frequencies.".format(prop))
            rdf = rdf[~rdf['file'].isin(diff)]
            rdf = rdf[~rdf['file'].isin(diff+nmodes)]
        return rdf

    @staticmethod
    def _get_temp_factor(temp, freq):
        if temp > 1e-6:
            try:
                factor = freq*Energy['Ha', 'J'] / (2 * boltzmann * temp)
                temp_fac = np.cosh(factor) / np.sinh(factor)
            # this should be taken care of by the conditional but always good to
            # take care of explicitly
            except ZeroDivisionError:
                raise ZeroDivisionError("Something seems to have gone wrong with the sinh function")
        else:
            temp_fac = 1.
        return temp_fac

    @staticmethod
    def get_pos_neg_gradients(grad, freq):
        '''
        Here we get the gradients of the equilibrium, positive and negative displaced structures.
        We extract them from the gradient dataframe and convert them into normal coordinates
        by multiplying them by the frequency normal mode displacement values.

        Args:
            grad (:class:`exatomic.gradient.Gradient`): DataFrame containing all of the gradient data
            freq (:class:`exatomic.atom.Frquency`): DataFrame containing all of the frequency data

        Returns:
            delfq_zero (pandas.DataFrame): Normal mode converted gradients of equilibrium structure
            delfq_plus (pandas.DataFrame): Normal mode converted gradients of positive displaced structure
            delfq_minus (pandas.DataFrame): Normal mode converted gradients of negative displaced structure
        '''
        grouped = grad.groupby('file')
        # generate delta dataframe
        # TODO: make something so delta can be set
        #       possible issues are a user using a different type of delta
        nmodes = len(freq['freqdx'].drop_duplicates().values)
        # get gradient of the equilibrium coordinates
        grad_0 = grouped.get_group(0)
        # get gradients of the displaced coordinates in the positive direction
        grad_plus = grouped.filter(lambda x: x['file'].drop_duplicates().values in
                                                                        range(1,nmodes+1))
        snmodes = len(grad_plus['file'].drop_duplicates().values)
        # get gradients of the displaced coordinates in the negative direction
        grad_minus = grouped.filter(lambda x: x['file'].drop_duplicates().values in
                                                                        range(nmodes+1, 2*nmodes+1))
        # TODO: Check if we can make use of numba to speed up this code
        delfq_zero = freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda x:
                                    np.sum(np.multiply(grad_0[['fx', 'fy', 'fz']].values, x.values))).values
        # we extend the size of this 1d array as we will perform some matrix summations with the
        # other outputs from this method
        delfq_zero = np.tile(delfq_zero, snmodes).reshape(snmodes, nmodes)

        delfq_plus = grad_plus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                                freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                    np.sum(np.multiply(y.values, x.values)))).values
        delfq_minus = grad_minus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                                freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                    np.sum(np.multiply(y.values, x.values)))).values
        return [delfq_zero, delfq_plus, delfq_minus]

    @staticmethod
    def calculate_frequencies(delfq_0, delfq_plus, delfq_minus, redmass, select_freq, delta=None):
        '''
        Here we calculated the frequencies from the gradients calculated for each of the
        displaced structures along the normal mode. In principle this should give the same or
        nearly the same frequency value as that from a frequency calculation.

        Args:
            delfq_0 (numpy.ndarray): Array that holds all of the information about the gradient
                                     derivative of the equlilibrium coordinates
            delfq_plus (numpy.ndarray): Array that holds all of the information about the gradient
                                        derivative of the positive displaced coordinates
            delfq_minus (numpy.ndarray): Array that holds all of the information about the gradient
                                         derivative of the negative displaced coordinates
            redmass (numpy.ndarray): Array that holds all of the reduced masses. We can handle both
                                     a subset of the entire values or all of the values
            select_freq (numpy.ndarray): Array that holds the selected frequency indexes
            delta (numpy.ndarray): Array that has the delta values used in the displaced structures

        Returns:
            frequencies (numpy.ndarray): Frequency array from the calculation
        '''
        if delta is None:
            print("No delta has been given. Assume delta_type to be 2.")
            delta = va.gen_delta(delta_type=2, freq=freq.copy())['delta'].values
        # get number of selected normal modes
        # TODO: check stability of using this parameter
        snmodes = len(select_freq)
        #print("select_freq.shape: {}".format(select_freq.shape))
        if len(redmass) > snmodes:
            redmass_sel = redmass[select_freq]
        else:
            redmass_sel = redmass
        if len(delta) > snmodes:
            delta_sel = delta[select_freq]
        else:
            delta_sel = delta
        # calculate force constants
        kqi = np.zeros(len(select_freq))
        #print(redmass_sel.shape)
        for fdx, sval in enumerate(select_freq):
            kqi[fdx] = (delfq_plus[fdx][sval] - delfq_minus[fdx][sval]) / (2.0*delta_sel[fdx])

        vqi = np.divide(kqi, redmass_sel.reshape(snmodes,))
        # TODO: Check if we want to exit the program if we get a negative force constant
        n_force_warn = vqi[vqi < 0.]
        if n_force_warn.any() == True:
            # TODO: point to exactly which frequencies are negative
            negative = np.where(vqi<0)[0]
            text = ''
            # frequencies are base 0
            for n in negative[:-1]: text += str(n)+', '
            text += str(negative[-1])
            warnings.warn("Negative force constants have been calculated for frequencies {} be wary of results".format(text),
                            Warning)
        # return calculated frequencies
        frequencies = np.sqrt(vqi).reshape(snmodes,)*Energy['Ha', 'cm^-1']
        return frequencies

    def vroa(self, uni, delta, units='nm', raman_units=False, temp=None, assume_real=False):
        """
        Here we implement the Vibrational Raman Optical Activity (VROA) equations as outlined in
        the paper J. Chem. Phys. 2007, 127,
        134101. The general workflow is that we must read in the data from a Raman Optical Activity
        calculation with your software of choice and this script will take that data and generate
        the forward and back scattering intensities for VROA. From here you will be able to plot
        the spectra with another method in this same class.

        Note:
            It is extremely important that the delta values that you pass into the function are the
            exact same as the ones that were used to generate the displaced structures. We do not
            currently have a method to do this automatically but we are working on it.

        Args:
            uni (:class:`~exatomic.Universe`): Universe containing all dataframes from the
                                               frequency calculation
            delta (numpy.ndarray): Array containing all of the delta values used for the generation
                                of the displaced structures.
            units (str): Units of the excitation frequencies. Default to nm.
            temp (float): Temperature value for the calculation
            raman_units (bool): Convert from atomic units to the Raman intensity units see raman_int_units for more information
            assume_real (bool): Neglect the contribution from the imaginary tensor values
        """
        if not hasattr(self, 'roa'):
            raise AttributeError("Please set roa attribute.")
        if not hasattr(self, 'gradient'):
            raise AttributeError("Please set gradient attribute.")
        if not hasattr(uni, 'frequency_ext'):
            raise AttributeError("Please compute frequency_ext dataframe.")
        if not hasattr(uni, 'frequency'):
            raise AttributeError("Please compute frequency dataframe.")
        # we must remove the 0 index file as by default our displaced coordinate generator will
        # include these values and they have no significane in this code as of yet
        try:
            roa_0 = self.roa.groupby('file').get_group(0)
            idxs = roa_0.index.values
            roa = self.roa.loc[~self.roa.index.isin(idxs)]
        except KeyError:
            roa = self.roa.copy()
        # number of normal modes
        nmodes = len(uni.frequency_ext.index.values)
        # initialize scatter array
        scatter = []
        raman = []
        # set some variables that will be used throughout
        # speed of light in au
        C_au = C*Length['m', 'au']/Time['s','au']
        # a conversion factor for the beta_g beta_A and alpha_g tensor invariants
        # TODO: make the conversion for the alha_squared and beta_alpha invariants
        au2angs = Length['au', 'Angstrom']
        #conver = 1/C_au
        # get the square roots of the reduced masses
        rmass = np.sqrt(uni.frequency_ext['r_mass'].values)
        # generate a Levi Civita 3x3x3 tensor
        epsilon = np.array([[0,0,0,0,0,1,0,-1,0],[0,0,-1,0,0,0,1,0,0],[0,1,0,-1,0,0,0,0,0]])
        # some dictionaries to replace the string labels with integers
        # this is important so we can speed up the code with jit
        rep_label = {'Ax': 0, 'Ay': 1, 'Az': 2, 'alpha': 3, 'g_prime': 4}
        rep_type = {'real': 0, 'imag': 1}
        # replace the columns
        roa.replace(rep_label, inplace=True)
        roa.replace(rep_type, inplace=True)
        # get rid of the frame column serves no purpose here
        roa.drop('frame', axis=1, inplace=True)
        # the excitation frequencies
        try:
            exc_freq = roa['exc_freq'].drop_duplicates().values
            text = ''
            for f in exc_freq: text += str(f)+', '
            print("Found excitation frequencies: {}".format(text))
        except KeyError:
            exc_freq = [-1]
            roa['exc_freq'] = np.repeat(-1, len(roa))
            print("No excitation frequency column (exc_freq) found in va_corr.roa."+ \
                         "Continuing assuming single excitation frequency.")
        # loop over all of the excitation frequencies performing the needed calculations
        for val in exc_freq:
            # omega parameter
            if units == 'nm':
                try:
                    omega = H*C/(val*Length['nm', 'm'])*Energy['J', 'Ha']
                except ZeroDivisionError:
                    omega = 0.0
                    warnings.warn("Omega parameter has been set to 0. beta(A)**2 will be zero by extension.", Warning)
            else:
                omega = val*Energy[units, 'Ha']
            if val == -1:
                omega = 0.0
                warnings.warn("Omega parameter has been set to 0. beta(A)**2 will be zero by extension.", Warning)
            #print(omega)
            sel_roa = roa.groupby('exc_freq').get_group(val)
            grad = self.gradient.groupby('exc_freq').get_group(val)
            # check for any missing files
            sel_roa = self._check_file_continuity(sel_roa, "ROA", nmodes)
            grad = self._check_file_continuity(grad, "gradient", nmodes)
            # get the frequencies that have been calculated
            # our code allows the user to calculate the roa of certain normal modes
            # this allows the user to decrease the computational cost significantly
            select_freq = sel_roa['file'].drop_duplicates().values-1
            mask = select_freq > nmodes-1
            select_freq = select_freq[~mask]
            #print(select_freq)
            #print(select_freq)
            snmodes = len(select_freq)
            # get the reduced mass and delta parameters of the calculated normal modes
            if snmodes < nmodes:
                sel_rmass = rmass[select_freq].reshape(snmodes,1)
                sel_delta = delta[select_freq].reshape(snmodes,1)
            else:
                sel_rmass = rmass.reshape(snmodes, 1)
                sel_delta = delta.reshape(snmodes, 1)
            # create a numpy array with the necessary dimensions
            # number_of_files/2 x 9
            value_complex = np.zeros((int(len(sel_roa)/2),9), dtype=np.complex128)
            labels = np.zeros(int(len(sel_roa)/2), dtype=np.int8)
            files = np.zeros(int(len(sel_roa)/2), dtype=np.int8)
            #print(sel_delta)
            #start = time.time()
            # combine the real and imaginary values
            # passed through jitted code
            _sum(sel_roa.values, value_complex, labels, files)
            #print("Completed sum: {}".format(time.time()-start))
            labels = pd.Series(labels)
            files = pd.Series(files)
            # replace the integer labels with the strings again
            # TODO: is this really necessary?
            labels.replace({v: k for k, v in rep_label.items()}, inplace=True)
            complex_roa= pd.DataFrame(value_complex)
            complex_roa.index = labels
            complex_roa['file'] = np.repeat(range(2*snmodes),5)
            #print(complex_roa)
            #complex_roa['exc_freq'] = np.repeat(exc_freq, 10*nmodes)
            # because I could not use range(9)............ugh
            cols = [0,1,2,3,4,5,6,7,8]
            # splice the data into the respective tensor dataframes
            # we want all of the tensors in a 1d vector like form
            A = pd.DataFrame.from_dict(complex_roa.loc[['Ax','Ay','Az']].groupby('file').
                                       apply(lambda x: np.array([x[cols].values[0],x[cols].values[1],
                                                                 x[cols].values[2]]).flatten()).
                                       reset_index(drop=True).to_dict()).T
            alpha = pd.DataFrame.from_dict(complex_roa.loc['alpha',range(9)].reset_index(drop=True).
                                           to_dict())
            g_prime = pd.DataFrame.from_dict(complex_roa.loc['g_prime',range(9)].
                                             reset_index(drop=True).to_dict())
            #***********DEBUG***********#
            #self.A = A
            #self.alpha = alpha
            #self.g_prime = g_prime
            #for i in g_prime.values.reshape(snmodes*9*2):
            #    print("{} {}".format(i.real, i.imag))
            #********END DEBUG**********#

            # get gradient calculated frequencies
            # this is just to make sure that we are calculating the right frequency
            # this comes from the init_va code
            grad_derivs = self.get_pos_neg_gradients(grad, uni.frequency.copy())
            frequencies = self.calculate_frequencies(*grad_derivs, sel_rmass**2*Mass['u','au_mass'],
                                                     select_freq, sel_delta)

            # TODO: here we could compare the real frequencies to the ones calculated from the gradients
            #       need to look into how stable this is.
            #if not np.allclose(np.sort(frequencies), uni.frequency.loc[select_freq, 'frequency'].values):
            #    warnings.warn("The calculated frequencies are not within a relative tolerance of 1e-6 to the real frequencies.", Warning)

            # separate tensors into positive and negative displacements
            # highly dependent on the value of the index
            # we neglect the equilibrium coordinates
            # 0 corresponds to equilibrium coordinates
            # 1 - nmodes corresponds to positive displacements
            # nmodes+1 - 2*nmodes corresponds to negative displacements
            alpha_plus = np.divide(alpha.loc[range(0,snmodes)].values, sel_rmass)
            alpha_minus = np.divide(alpha.loc[range(snmodes, 2*snmodes)].values, sel_rmass)
            g_prime_plus = np.divide(g_prime.loc[range(0,snmodes)].values, sel_rmass)
            g_prime_minus = np.divide(g_prime.loc[range(snmodes, 2*snmodes)].values, sel_rmass)
            A_plus = np.divide(A.loc[range(0, snmodes)].values, sel_rmass)
            A_minus = np.divide(A.loc[range(snmodes, 2*snmodes)].values, sel_rmass)

            # generate derivatives by two point difference method
            dalpha_dq = np.divide((alpha_plus - alpha_minus), 2 * sel_delta)
            dg_dq = np.divide((g_prime_plus - g_prime_minus), 2 * sel_delta)
            dA_dq = np.array([np.divide((A_plus[i] - A_minus[i]), 2 * sel_delta[i])
                                                                    for i in range(snmodes)])
            #***********DEBUG***********#
            #self.dalpha_dq = dalpha_dq
            #self.dg_dq = dg_dq
            #self.dA_dq = dA_dq
            #print("#################{}################".format(val))
            #for i in dg_dq:
            #    for k in i:
            #        print("{:.6f} {:.6f}".format(k.real, k.imag))
            #********END DEBUG**********#

            # generate properties as shown on equations 5-9 in paper
            # J. Chem. Phys. 2007, 127, 134101
            alpha_squared, beta_alpha, beta_g, beta_A, alpha_g = _make_derivatives(dalpha_dq,
                                  dg_dq, dA_dq, omega, epsilon, snmodes, au2angs**4, C_au, assume_real)

            #********************************DEBUG**************************************************#
            #self.alpha_squared = pd.Series(alpha_squared*Length['au', 'Angstrom']**4)
            #self.beta_alpha = pd.Series(beta_alpha*Length['au', 'Angstrom']**4)
            #self.beta_g = pd.Series(beta_g*Length['au', 'Angstrom']**4/
            #                                                    (C*Length['m', 'au']/Time['s','au']))
            #self.beta_A = pd.Series(beta_A*Length['au', 'Angstrom']**4/
            #                                                    (C*Length['m', 'au']/Time['s','au']))
            #self.alpha_g = pd.Series(alpha_g*Length['au', 'Angstrom']**4/
            #                                                    (C*Length['m', 'au']/Time['s','au']))
            #*******************************END DEBUG***********************************************#

            # calculate Raman intensities
            raman_int = 4 * (45 * alpha_squared + 8 * beta_alpha)

            # calculate VROA back scattering and forward scattering intensities
            backscat_vroa = _backscat(beta_g, beta_A)
            #backscat_vroa *= 1e4
            # TODO: check the units of this because we convert the invariants from
            #       au to Angstrom and here we convert again from au to Angstrom
            #backscat_vroa *= Length['au', 'Angstrom']**4*Mass['u', 'au_mass']
            #backscat_vroa *= Mass['u', 'au_mass']
            forwscat_vroa = _forwscat(alpha_g, beta_g, beta_A)
            #forwscat_vroa *= 1e4
            if raman_units:
                lambda_0 = 1/(val*Length['nm', 'm'])
                lambda_p = frequencies/Length['cm', 'm']
                kp = self.raman_int_units(lambda_0=lambda_0, lambda_p=lambda_p, temp=temp)*Length['m', 'cm']**2
                raman_int *= kp
                backscat_vroa *= kp
                forwscat_vroa *= kp
            # TODO: check the units of this because we convert the invariants from
            #       au to Angstrom and here we convert again from au to Angstrom
            #forwscat_vroa *=Length['au', 'Angstrom']**4*Mass['u', 'au_mass']
            # we set this just so it is easier to view the data
            pd.options.display.float_format = '{:.6f}'.format
            # generate dataframe with all pertinent data for vroa scatter
            df = pd.DataFrame.from_dict({"freq": frequencies, "freqdx": select_freq, "beta_g*1e6":beta_g*1e6,
                                        "beta_A*1e6": beta_A*1e6, "alpha_g*1e6": alpha_g*1e6,
                                        "backscatter": backscat_vroa, "forwardscatter":forwscat_vroa})
            df['exc_freq'] = np.repeat(val, len(df))
            rdf = pd.DataFrame.from_dict({"freq": frequencies, "freqdx": select_freq,
                                          "alpha_squared": alpha_squared,
                                          "beta_alpha": beta_alpha, "raman_int": raman_int})
            rdf['exc_freq'] = np.repeat(val, len(rdf))
            scatter.append(df)
            raman.append(rdf)
        self.scatter = pd.concat(scatter)
        self.scatter.sort_values(by=['exc_freq','freq'], inplace=True)
        # added this as there seems to be some issues with the indexing when there are
        # nearly degenerate modes
        self.scatter.reset_index(drop=True, inplace=True)
        # check ordering of the freqdx column
        self.raman = pd.concat(raman)
        self.raman.sort_values(by=['exc_freq', 'freq'], inplace=True)
        self.scatter.reset_index(drop=True, inplace=True)
        if not np.allclose(self.scatter['freqdx'].values, np.sort(self.scatter['freqdx'].values)):
            warnings.warn("Found an ordering issue with the calculated frequencies. Make sure to check the frequency values.", Warning)

    def zpvc(self, uni, delta, temperature=None, geometry=True, print_results=False):
        """
        Method to compute the Zero-Point Vibrational Corrections. We implement the equations as
        outlined in the paper J. Phys. Chem. A 2005, 109, 8617-8623 (doi:10.1021/jp051685y).
        Here we compute the effect of vibrations on a specified property given as a n x 2 array
        where one of the columns are the file indexes and the other is the property.
        We use a two and three point difference method to calculate the first and second derivatives
        respectively.

        We have also implemented a way to calculate the ZPVC and effective geometries at
        different temperatures given in Kelvin.

        Note:
            The code has been designed such that the property input array must have one column
            labeled file corresponding to the file indexes.

        Args:
            uni (:class:`exatomic.Universe`): Universe containing all pertinent data
            delta (numpy.array): Array of the delta displacement parameters
            temperature (list): List object containing all of the temperatures of interest
            geometry (bool): Bool value that tells the program to also calculate the effective geometry
            print_results(bool): Bool value to print the results from the zpvc calcualtion to stdout
        """
        if not hasattr(self, 'gradient'):
            raise AttributeError("Please set gradient attribute.")
        if not hasattr(self, 'property'):
            raise AttributeError("Please set property attribute.")
        if self.property.shape[1] != 2:
            raise ValueError("Property dataframe must have a second dimension of 2 not {}".format(
                                                                                    self.property.shape[1]))
        if not hasattr(uni, 'frequency_ext'):
            raise AttributeError("Please compute frequency_ext dataframe.")
        if not hasattr(uni, 'frequency'):
            raise AttributeError("Please compute frequency dataframe.")
        if temperature is None: temperature = [0]

        # get the total number of normal modes
        nmodes = len(uni.frequency_ext.index.values)
        # check for any missing files and remove the respective counterpart
        grad = self._check_file_continuity(self.gradient, 'gradient', nmodes)
        prop = self._check_file_continuity(self.property, 'property', nmodes)
        # check that the equlibrium coordinates are included
        # these are required for the three point difference methods
        try:
            tmp = grad.groupby('file').get_group(0)
        except KeyError:
            raise KeyError("Equilibrium coordinate gradients not found")

        try:
            tmp = prop.groupby('file').get_group(0)
        except KeyError:
            raise KeyError("Equilibrium coordinate property not found")
        # check that the gradient and property dataframe have the same length of data
        grad_files = grad[grad['file'].isin(range(0,nmodes+1))]['file'].drop_duplicates()
        prop_files = prop[prop['file'].isin(range(nmodes+1,2*nmodes+1))]['file'].drop_duplicates()
        # compare lengths
        # TODO: make sure the minus 1 is in the right place
        #       we suppose that it is because we grab the file number 0 as an extra
        if grad_files.shape[0]-1 != prop_files.shape[0]:
            print("Length mismatch of gradient and property arrays.")
            # we create a dataframe to make use of the existing file continuity checker
            df = pd.DataFrame(np.concatenate([grad_files, prop_files]), columns=['file'])
            df = self._check_file_continuity(df, 'grad/prop', nmodes)
            # overwrite the property and gradient dataframes
            grad = grad[grad['file'].isin(df['file'])]
            prop = prop[prop['file'].isin(df['file'])]
        # get the gradients multiplied by the normal modes
        delfq_zero, delfq_plus, delfq_minus = self.get_pos_neg_gradients(grad, uni.frequency.copy())
        # get the selected frequencies
        select_freq = grad[grad['file'].isin(range(1,nmodes+1))]
        select_freq = select_freq['file'].drop_duplicates().values - 1
        snmodes = len(select_freq)
        #print(select_freq, snmodes)
        # get the actual frequencies
        # TODO: check if we should use the real or calculated frequencies
        frequencies = uni.frequency_ext['freq'].values*Energy['cm^-1','Ha']
        rmass = uni.frequency_ext['r_mass'].values*Mass['u', 'au_mass']
        if snmodes < nmodes:
            raise NotImplementedError("We do not currently have support to handle missing frequencies")
            #sel_delta = delta[select_freq]
            #sel_rmass = uni.frequency_ext['r_mass'].values[select_freq]*Mass['u', 'au_mass']
            #sel_freq = uni.frequency_ext['freq'].values[select_freq]*Energy['cm^-1','Ha']
        else:
            sel_delta = delta
            sel_rmass = rmass
            sel_freq = frequencies
        _ = self.calculate_frequencies(delfq_zero, delfq_plus, delfq_minus, sel_rmass, select_freq,
                                                 sel_delta)
        # calculate cubic force constant
        # we use a for loop because we need the diagonal values
        # if we select a specific number of modes then the diagonal elements
        # are tricky
        kqiii = np.zeros(len(select_freq))
        for fdx, sval in enumerate(select_freq):
            kqiii[fdx] = (delfq_plus[fdx][sval] - 2.0 * delfq_zero[fdx][sval] + \
                                                delfq_minus[fdx][sval]) / (sel_delta[fdx]**2)
        # calculate anharmonic cubic force constant
        # this will have nmodes rows and snmodes cols
        kqijj = np.divide(delfq_plus - 2.0 * delfq_zero + delfq_minus,
                          np.multiply(sel_delta, sel_delta).reshape(snmodes,1))
        # get property values
        prop_grouped = prop.groupby('file')
        # get the property value for the equilibrium coordinate
        prop_zero = prop_grouped.get_group(0)
        prop_zero.drop(columns=['file'],inplace=True)
        prop_zero = np.repeat(prop_zero.values, snmodes)
        # get the property values for the positive displaced structures
        prop_plus = prop_grouped.filter(lambda x: x['file'].drop_duplicates().values in range(1,nmodes+1))
        prop_plus.drop(columns=['file'], inplace=True)
        prop_plus = prop_plus.values.reshape(snmodes,)
        # get the property values for the negative displaced structures
        prop_minus= prop_grouped.filter(lambda x: x['file'].drop_duplicates().values in
                                                                              range(nmodes+1, 2*nmodes+1))
        prop_minus.drop(columns=['file'], inplace=True)
        prop_minus = prop_minus.values.reshape(snmodes,)
        # generate the derivatives of the property
        dprop_dq = np.divide(prop_plus - prop_minus, 2*sel_delta)
        d2prop_dq2 = np.divide(prop_plus - 2*prop_zero + prop_minus, np.multiply(sel_delta, sel_delta))
        # done with setting up everything
        # moving on to the actual calculations

        atom_frames = uni.atom['frame'].values
        eqcoord = uni.atom.groupby('frame').get_group(atom_frames[-1])[['x','y','z']].values
        atom_order = uni.atom['symbol']
        coor_dfs = []
        zpvc_dfs = []
        va_dfs = []

        # calculate the ZPVC's at different temperatures by iterating over them
        for t in temperature:
            # calculate anharmonicity in the potential energy surface
            anharm = np.zeros(snmodes)
            for i in range(snmodes):
                temp1 = 0.0
                for j in range(nmodes):
                    # calculate the contribution of each vibration
                    temp_fac = self._get_temp_factor(t, frequencies[j])
                    # TODO: check the snmodes and nmodes indexing for kqijj
                    #       pretty sure that the rows are nmodes and the columns are snmodes
                    # TODO: check which is in the sqrt
                    # sum over the first index
                    temp1 += kqijj[j][i]/(frequencies[j]*rmass[j]*np.sqrt(sel_rmass[i]))*temp_fac
                # sum over the second index and set anharmonicity at each vibrational mode
                anharm[i] = -0.25*dprop_dq[i]/(sel_freq[i]**2*np.sqrt(sel_rmass[i]))*temp1
            # calculate curvature of property
            curva = np.zeros(snmodes)
            for i in range(snmodes):
                # calculate the contribution of each vibration
                temp_fac = self._get_temp_factor(t, sel_freq[i])
                # set the curvature at each vibrational mode
                curva[i] = 0.25*d2prop_dq2[i]/(sel_freq[i]*sel_rmass[i])*temp_fac

            # generate one of the zpvc dataframes
            va_dfs.append(pd.DataFrame.from_dict({'freq': sel_freq*Energy['Ha','cm^-1'], 'freqdx': select_freq,
                                                    'anharm': anharm, 'curva': curva, 'sum': anharm+curva,
                                                    'temp': np.repeat(t, snmodes)}))
            zpvc = np.sum(anharm+curva)
            tot_anharm = np.sum(anharm)
            tot_curva = np.sum(curva)
            zpvc_dfs.append([prop_zero[0], zpvc, prop_zero[0] + zpvc, tot_anharm, tot_curva, t])
            if print_results:
                print("========Results from Vibrational Averaging at {} K==========".format(t))
                # print results to stdout
                print("----Result of ZPVC calculation for {} of {} frequencies".format(snmodes, nmodes))
                print("    - Total Anharmonicity:   {:+.6f}".format(tot_anharm))
                print("    - Total Curvature:       {:+.6f}".format(tot_curva))
                print("    - Zero Point Vib. Corr.: {:+.6f}".format(zpvc))
                print("    - Zero Point Vib. Avg.:  {:+.6f}".format(prop_zero[0] + zpvc))
            if geometry:
                # calculate the effective geometry
                # we do not check this at the beginning as it will not always be computed
                if not hasattr(uni, 'atom'):
                    raise AttributeError("Please set the atom dataframe")
                sum_to_eff_geo = np.zeros((eqcoord.shape[0], 3))
                for i in range(snmodes):
                    temp1 = 0.0
                    for j in range(nmodes):
                        # calculate the contribution of each vibration
                        temp_fac = self._get_temp_factor(t, frequencies[j])
                        temp1 += kqijj[j][i]/(frequencies[j]*rmass[j]*np.sqrt(sel_rmass[i])) * temp_fac
                    # get the temperature correction to the geometry in Bohr
                    sum_to_eff_geo += -0.25 * temp1 / (sel_freq[i]**2 * np.sqrt(sel_rmass[i])) * \
                                        uni.frequency.groupby('freqdx').get_group(i)[['dx','dy','dz']].values
                # get the effective geometry
                tmp_coord = np.transpose(eqcoord + sum_to_eff_geo)
                # generate one of the coordinate dataframes
                # we write the frame to be the same as the temp column so that one can take
                # advantage of the exatomic.core.atom.Atom.to_xyz method
                coor_dfs.append(pd.DataFrame.from_dict({'set': list(range(len(eqcoord))),
                                                        'Z': atom_order.map(sym2z), 'x': tmp_coord[0],
                                                        'y': tmp_coord[1], 'z': tmp_coord[2],
                                                        'symbol': atom_order,
                                                        'temp': np.repeat(t, eqcoord.shape[0]),
                                                        'frame': np.repeat(t, len(eqcoord))}))
                # print out the effective geometry in Angstroms
                if print_results:
                    print("----Effective geometry in Angstroms")
                    xyz = coor_dfs[-1][['symbol','x','y','z']].copy()
                    xyz['x'] *= Length['au', 'Angstrom']
                    xyz['y'] *= Length['au', 'Angstrom']
                    xyz['z'] *= Length['au', 'Angstrom']
                    stargs = {'columns': None, 'header': False, 'index': False,
                              'formatters': {'symbol': '{:<5}'.format}, 'float_format': '{:6f}'.format}
                    print(xyz.to_string(**stargs))
        if geometry:
            self.eff_coord = pd.concat(coor_dfs, ignore_index=True)
        self.zpvc_results = pd.DataFrame(zpvc_dfs,
                                         columns=['property', 'zpvc', 'zpva', 'tot_anharm', 'tot_curva', 'temp'])
        self.vib_average = pd.concat(va_dfs, ignore_index=True)

