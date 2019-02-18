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
import time
from exa.util.constants import speed_of_light_in_vacuum as C, Planck_constant as H
from exa.util.units import Length, Energy, Mass, Time
from exa.util.utility import mkp
from exatomic.core import Atom, Gradient
from exa import TypedMeta
from .vroa_funcs import _sum, _make_derivatives, _forwscat, _backscat
import warnings
warnings.simplefilter("default")

def get_data(path, attr, soft, f_end='', f_start='', sort_index=['']):
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
                print("The property {} cannot be found in output {}".format(attr, file))
                continue
            # We assume that the file identifier is an integer
            fdx = list(map(int, re.findall('\d+', file.split('/')[-1].replace(
                                                                   f_start, '').replace(f_end, ''))))
            #fdx = float(file.split(os.sep)[-1].replace(f_start, '').replace(f_end, ''))
            df['file'] = np.tile(fdx, len(df))
        else:
            continue
        array.append(df)
    cdf = pd.concat([arr for arr in array])
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
            raise KeyError("Please make sure that the keys {} exist in the dataframe "+ \
                                        "created by {}.parse_{}.".format(sort_values, soft, attr))
    cdf.reset_index(drop=True, inplace=True)
    return cdf

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
#    @staticmethod
#    def _calc_kp(lambda_0, lambda_p):
#        '''
#        Function to calculate the K_p value as given in equation 2 on J. Chem. Phys. 2007, 127, 134101.
#        We assume the temperature to be 298.15 as a hard coded value. Must get rid of this in future
#        iterations. The final units of the equation is in m^2.
#        Input values lambda_0 and lambda_p must be in the units of m^-1
#        '''
#        # epsilon_0 = 1/(4*np.pi*1e-7*C**2)
#        # another hard coded value
#        temp = 298.15 # Kelvin
#        boltz = 1.0/(1.0-np.exp(-H*C*lambda_p/(KB*temp)))
#        constants = H * np.pi**2 / C
#        variables = (lambda_0 - lambda_p)**4/lambda_p
#        kp = 2 * variables * constants * boltz * (Length['au', 'm']**4 / Mass['u', 'kg'])
#        return kp

    def get_pos_neg_gradients(self, grad, freq):
        '''
        Here we get the gradients of the equilibrium, positive and negative displaced structures.
        We extract them from the gradient dataframe and convert them into normal coordinates
        by multiplying them by the frequency normal mode displacement values.

        Args:
            grad (:class:`exatomic.gradient.Gradient`): DataFrame containing all of the gradient
                                                        data
            freq (:class:`exatomic.atom.Frquency`): DataFrame containing all of the frequency data

        Returns:
            delfq_zero (pandas.DataFrame): Normal mode converted gradients of equilibrium structure
            delfq_plus (pandas.DataFrame): Normal mode converted gradients of positive displaced
                                           structure
            delfq_minus (pandas.DataFrame): Normal mode converted gradients of negative displaced
                                            structure
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
        snmodes = len(grad_plus)
        # get gradients of the displaced coordinates in the negative direction
        grad_minus = grouped.filter(lambda x: x['file'].drop_duplicates().values in
                                                                        range(nmodes+1, 2*nmodes+1))
        # TODO: Check if we can make use of numba to speed up this code
        delfq_zero = freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda x:
                                    np.sum(np.multiply(grad_0[['fx', 'fy', 'fz']].values, x.values)))
        delfq_zero = np.tile(delfq_zero, snmodes).reshape(snmodes, nmodes)
        delfq_plus = grad_plus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                                freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                    np.sum(np.multiply(y.values, x.values)))).values
        delfq_minus = grad_minus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                                freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                    np.sum(np.multiply(y.values, x.values)))).values
        return [delfq_zero, delfq_plus, delfq_minus]

    def calculate_frequencies(self, delfq_0, delfq_plus, delfq_minus, redmass, select_freq, delta=None):
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
        if len(redmass) > snmodes:
            redmass_sel = redmass[select_freq] * Mass['u','au_mass']
        else:
            redmass_sel = redmass * Mass['u','au_mass']
        if len(delta) > snmodes:
            delta_sel = delta[select_freq]
        else:
            delta_sel = delta
        # calculate force constants
        kqi = np.zeros(len(select_freq))
        for fdx, sval in enumerate(select_freq):
            kqi[fdx] = (delfq_plus[fdx][sval] - delfq_minus[fdx][sval]) / (2.0*delta_sel[fdx])
        vqi = np.divide(kqi, redmass_sel)
        # TODO: Check if we want to exit the program if we get a negative force constant
        n_force_warn = vqi[vqi < 0.]
        if n_force_warn.any() == True:
            negative = np.where(n_force_warn)
            warnings.warn("Negative force constants have been calculated be wary of results",
                            Warning)
        # return calculated frequencies
        frequencies = np.sqrt(vqi).reshape(1,)*Energy['Ha', 'cm^-1']
        return frequencies

    def vroa(self, uni, delta, units='nm', assume_real=False, no_conj=False):
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
            delta (np.ndarray): Array containing all of the delta values used for the generation
                                of the displaced structures.
            units (string): Units of the excitation frequencies. Default to nm.
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
        # initialize scatter array
        scatter = []
        raman = []
        # set some variables that will be used throughout
        # number of normal modes
        nmodes = len(uni.frequency_ext.index.values)
        # speed of light in au
        C_au = C*Length['m', 'au']/Time['s','au']
        # a conversion factor for the beta_g beta_A and alpha_g tensor invariants
        # TODO: make the conversion for the alha_squared and beta_alpha invariants
        au2angs = Length['au', 'Angstrom']**4
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
        for idx, val in enumerate(exc_freq):
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
            # get the frequencies that have been calculated
            # our code allows the user to calculate the roa of certain normal modes
            # this allows the user to decrease the computational cost significantly
            select_freq = sel_roa['file'].drop_duplicates().values-1
            mask = select_freq > nmodes-1
            select_freq = select_freq[~mask]
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
            try:
                grad = self.gradient.groupby('exc_freq').get_group(val)
                grad_derivs = self.get_pos_neg_gradients(grad, uni.frequency.copy())
                frequencies = self.calculate_frequencies(*grad_derivs, sel_rmass**2, select_freq, sel_delta)
            except KeyError:
                raise KeyError("Something went wrong check that self.calcfreq has column names "+ \
                               "calc_freq and exc_freq")
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
                                  dg_dq, dA_dq, omega, epsilon, snmodes, au2angs, C_au, assume_real,
                                  no_conj)

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
            backscat_vroa = _backscat(C_au, beta_g, beta_A)
            #backscat_vroa *= 1e4
            # TODO: check the units of this because we convert the invariants from
            #       au to Angstrom and here we convert again from au to Angstrom
            backscat_vroa *= Length['au', 'Angstrom']**4*Mass['u', 'au_mass']
            #backscat_vroa *= Mass['u', 'au_mass']
            forwscat_vroa = _forwscat(C_au, alpha_g, beta_g, beta_A)
            #forwscat_vroa *= 1e4
            # TODO: check the units of this because we convert the invariants from
            #       au to Angstrom and here we convert again from au to Angstrom
            forwscat_vroa *=Length['au', 'Angstrom']**4*Mass['u', 'au_mass']
            # we set this just so it is easier to view the data
            pd.options.display.float_format = '{:.6f}'.format
            # generate dataframe with all pertinent data for vroa scatter
            df = pd.DataFrame.from_dict({"freq": frequencies, "beta_g*1e6":beta_g*1e6,
                                        "beta_A*1e6": beta_A*1e6, "alpha_g*1e6": alpha_g*1e6,
                                        "backscatter": backscat_vroa, "forwardscatter":forwscat_vroa})
            df['exc_freq'] = np.repeat(val, len(df))
            rdf = pd.DataFrame.from_dict({"freq": frequencies, "alpha_squared": alpha_squared,
                                          "beta_alpha": beta_alpha, "raman_int": raman_int})
            rdf['exc_freq'] = np.repeat(val, len(rdf))
            scatter.append(df)
            raman.append(rdf)
        self.scatter = pd.concat(scatter, ignore_index=True)
        self.scatter.sort_values(by=['exc_freq','freq'], inplace=True)
        self.raman = pd.concat(raman, ignore_index=True)
        self.raman.sort_values(by=['exc_freq', 'freq'], inplace=True)

#    def init_va(self, uni, delta=None):
#        """
#        This is a method to initialize all of the variables that will be needed to execute the VA
#        program. As a sanity check we calculate the frequencies from the force constants. If we
#        have any negative force constants the results may not be as reliable.
#
#        Args:
#            uni (:class:`~exatomic.Universe`): Universe object containg pertinent data from
#                                               frequency calculation
#        """
#        if delta is None:
#            delta_df = gen_delta(freq=uni.frequency.copy(), delta_type=2)
#            delta = delta_df['delta'].values
#        if not hasattr(self, "gradient"):
#            raise AttributeError("Please set gradient attribute first")
#        if not hasattr(uni, "frequency_ext"):
#            raise AttributeError("Cannot find frequency extended dataframe in universe")
#        if not hasattr(uni, "frequency"):
#            raise AttributeError("Cannot find frequency dataframe in universe")
#        # check that all attributes to be used exist
#        # group the gradients by file (normal mode)
#        grad = self.gradient.copy()
#        try:
#            exc_freq = grad['exc_freq'].drop_duplicates().values
#            text = ''
#            for f in exc_freq: text += str(f)+', '
#            print("Found excitation frequencies: {}".format(text))
#        except KeyError:
#            exc_freq = [-1]
#            grad['exc_freq'] = np.repeat(-1, len(grad))
#            print("No excitation frequency column (exc_freq) found in va_corr.gradient."+ \
#                         "Continuing assuming single excitation frequency.")
#        # get number of normal modes
#        nmodes = len(uni.frequency_ext.index.values)
#        calcfreq = []
#        for idx, val in enumerate(exc_freq):
#            grouped = grad.groupby('exc_freq').get_group(val).groupby('file')
#            # generate delta dataframe
#            # TODO: make something so delta can be set
#            #       possible issues are a user using a different type of delta
#            # get gradient of the equilibrium coordinates
#            grad_0 = grouped.get_group(0)
#            # get gradients of the displaced coordinates in the positive direction
#            grad_plus = grouped.filter(lambda x: x['file'].drop_duplicates().values in
#                                                                            range(1,nmodes+1))
#            select_freq = grad_plus['file'].drop_duplicates().values-1
#            # get number of selected normal modes
#            # TODO: check stability of using this parameter
#            snmodes = len(select_freq)
#            if snmodes < nmodes:
#                delta_sel = delta[select_freq]
#                # convert force constants to reduced normal coordinate force constants
#                redmass = uni.frequency_ext.loc[select_freq,'r_mass'].values*Mass['u', 'au_mass']
#            else:
#                delta_sel = delta
#                redmass = uni.frequency_ext['r_mass'].values*Mass['u','au_mass']
#            #print(delta_sel, select_freq, redmass)
#            # get gradients of the displaced coordinates in the negative direction
#            grad_minus = grouped.filter(lambda x: x['file'].drop_duplicates().values in
#                                                                            range(nmodes+1, 2*nmodes+1))
#            # TODO: Check if we can make use of numba to speed up this code
#            delfq_zero = uni.frequency.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda x:
#                                        np.sum(np.multiply(grad_0[['fx', 'fy', 'fz']].values, x.values)))
#            #print(delfq_zero)
#            delfq_zero = np.tile(delfq_zero, snmodes).reshape(snmodes, nmodes)
#            #print(delfq_zero.shape)
#
#            delfq_plus = grad_plus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
#                                    uni.frequency.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
#                                        np.sum(np.multiply(y.values, x.values)))).values
#            #delfq_plus.reset_index(drop=True, inplace=True)
#            #print("delfq_plus shape: {}".format(delfq_plus.shape))
#
#            delfq_minus = grad_minus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
#                                    uni.frequency.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
#                                        np.sum(np.multiply(y.values, x.values)))).values
#            #delfq_minus.reset_index(drop=True, inplace=True)
#            #print("delfq_minus shape: {}".format(delfq_minus.shape))
#
#            # get diagonal elements of respqective matrix
#            #diag_plus = np.diag(delfq_plus)
#            #diag_minus = np.diag(delfq_minus)
#            #diag_zero = np.diag(delfq_zero)
#
#            # calculate force constants
#            #kqi   = np.divide(diag_plus - diag_minus, 2.0*delta_sel)
#            #kqiii = np.divide(diag_plus - 2.0 * diag_zero + diag_minus, np.multiply(delta_sel, delta_sel))
#            kqi = np.zeros(len(select_freq))
#            kqiii = np.zeros(len(select_freq))
#            for fdx, sval in enumerate(select_freq):
#                kqi[fdx] = (delfq_plus[fdx][sval] - delfq_minus[fdx][sval]) / (2.0*delta_sel[fdx])
#                kqiii[fdx] = (delfq_plus[fdx][sval] - 2.0 * delfq_zero[fdx][sval] + \
#                                                    delfq_minus[fdx][sval]) / (delta_sel[fdx]**2)
#            kqijj = np.divide(delfq_plus - 2.0 * delfq_zero + delfq_minus,
#                              np.multiply(delta_sel, delta_sel).reshape(snmodes, 1))
#            vqi = np.divide(kqi, redmass)
#            vqijj = np.divide(kqijj, np.sqrt(np.power(redmass, 3)).reshape(snmodes,1))
#
#            # TODO: Check if we want to exit the program if we get a negative force constant
#            n_force_warn = vqi[vqi < 0.]
#            #print(Energy['Ha','cm^-1'])
#
#            # calculate frequencies
#            df = pd.DataFrame.from_dict({'calc_freq': np.sqrt(vqi)*Energy['Ha', 'cm^-1'],
#                                         'exc_freq': np.repeat(val, snmodes),
#                                         'freqdx': select_freq})
#            calcfreq.append(df)
#        self.calcfreq = pd.concat(calcfreq, ignore_index=True)
#        #self.calcfreq = pd.DataFrame.from_dict({'calc_freq': calcfreq,
#        #                              'real_freq': uni.frequency_ext['freq']*Energy['Ha', 'cm^-1']})


