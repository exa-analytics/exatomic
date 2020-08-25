# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
VROA functions
##################
Functions for VROA calculations
"""
from numba import jit, prange, vectorize, float64
import numpy as np

@vectorize([float64(float64, float64)])
def _backscat(beta_g, beta_A):
    return 4.* (24 * beta_g + 8 * beta_A)

@vectorize([float64(float64, float64, float64)])
def _forwscat(alpha_g, beta_g, beta_A):
    return 4.* (180 * alpha_g + 4 * beta_g - 4 * beta_A)

@jit(nopython=True, parallel=False, cache=True)
def _make_derivatives(dalpha_dq, dg_dq, dA_dq, omega, epsilon, nmodes, au2angs, C_au, assume_real):
    alpha_squared = np.zeros(nmodes,dtype=np.complex128)
    for fdx in prange(nmodes):
        for al in prange(3):
            for be in prange(3):
                alpha_squared[fdx] += (1./9.)*(dalpha_dq[fdx][al*3+al]* \
                                                    np.conj(dalpha_dq[fdx][be*3+be]))
    alpha_squared = np.real(alpha_squared).astype(np.float64)*au2angs

    beta_alpha = np.zeros(nmodes,dtype=np.complex128)
    for fdx in prange(nmodes):
        for al in prange(3):
            for be in prange(3):
                beta_alpha[fdx] += 0.5*(3*dalpha_dq[fdx][al*3+be]*np.conj(dalpha_dq[fdx][al*3+be])- \
                                            dalpha_dq[fdx][al*3+al]*np.conj(dalpha_dq[fdx][be*3+be]))
    beta_alpha = np.real(beta_alpha).astype(np.float64)*au2angs

    beta_g = np.zeros(nmodes,dtype=np.complex128)
    beta_A = np.zeros(nmodes,dtype=np.complex128)
    if not assume_real:
        for fdx in prange(nmodes):
            for al in prange(3):
                for be in prange(3):
                    beta_g[fdx] += 0.5*(3*dalpha_dq[fdx][al*3+be]*np.conj(dg_dq[fdx][al*3+be])- \
                                               dalpha_dq[fdx][al*3+al]*np.conj(dg_dq[fdx][be*3+be]))
        for fdx in prange(nmodes):
            for al in prange(3):
                for be in prange(3):
                    for ga in prange(3):
                        for de in prange(3):
                            beta_A[fdx] += 0.5*omega*dalpha_dq[fdx][al*3+be]* \
                                           epsilon[al][ga*3+de]*np.conj(dA_dq[fdx][ga*9+de*3+be])
    else:
        for fdx in prange(nmodes):
            for al in prange(3):
                for be in prange(3):
                    beta_g[fdx] += 0.5*(3*np.real(dalpha_dq[fdx][al*3+be])*np.real(dg_dq[fdx][al*3+be])- \
                                               np.real(dalpha_dq[fdx][al*3+al])*np.real(dg_dq[fdx][be*3+be]))
        for fdx in prange(nmodes):
            for al in prange(3):
                for be in prange(3):
                    for ga in prange(3):
                        for de in prange(3):
                            beta_A[fdx] += 0.5*omega*np.real(dalpha_dq[fdx][al*3+be])* \
                                           epsilon[al][ga*3+de]*np.real(dA_dq[fdx][ga*9+de*3+be])

    beta_g = np.real(beta_g).astype(np.float64)*au2angs/C_au
    beta_A = np.real(beta_A).astype(np.float64)*au2angs/C_au

    alpha_g = np.zeros(nmodes, dtype=np.complex128)
    for fdx in prange(nmodes):
        for al in prange(3):
            for be in prange(3):
                # This equation seems to match the resonance ROA calculation by movipac
                #alpha_g[fdx] += np.real(dalpha_dq[fdx][al*3+al])*np.real(dg_dq[fdx][be*3+be])/3.
                # This equation matches what is on the listed paper in the docs (equation 9)
                alpha_g[fdx] += dalpha_dq[fdx][al*3+al]*np.conj(dg_dq[fdx][be*3+be])/9.
    alpha_g = np.real(alpha_g).astype(np.float64)*au2angs/C_au
    return alpha_squared, beta_alpha, beta_g, beta_A, alpha_g

@jit(nopython=True, parallel=False, cache=True)
def _sum(arr, out, labels, files):
    for fdx in range(0,len(arr),2):
        if arr[fdx][-3] == 0:
            out[int(fdx/2)] += arr[fdx][:9].astype(np.complex128)
            out[int(fdx/2)] += 1j*arr[fdx+1][:9].astype(np.complex128)
        else:
            out[int(fdx/2)] += 1j*arr[fdx][:9].astype(np.complex128)
            out[int(fdx/2)] += arr[fdx+1][:9].astype(np.complex128)
        labels[int(fdx/2)] = arr[fdx][-4]
        files[int(fdx/2)] = arr[fdx][-2]

