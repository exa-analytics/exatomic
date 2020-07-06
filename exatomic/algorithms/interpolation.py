# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Interpolation
################################
Hidden wrapper function that makes it convenient to choose
an interpolation scheme available in scipy.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import (interp1d, interp2d, griddata,
                               pchip_interpolate, krogh_interpolate,
                               barycentric_interpolate, Akima1DInterpolator,
                               CloughTocher2DInterpolator, RectBivariateSpline,
                               RegularGridInterpolator)
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


def _interpolate(df, x, y, z, method, kind, yfirst, dim, minimum):
    # Check that columns are in df
    if len(set([x, y, z]) & set(df.columns)) != 3:
        raise Exception('{!r}, {!r} and {!r} must be in df.columns'.format(x, y, z))
    oned = None
    if len(df[x].unique()) == 1:
        oned, dud = y, x
    elif len(df[y].unique()) == 1:
        oned, dud = x, y
    # Map the method to the function or class in scipy
    convenience = {'cloughtocher': CloughTocher2DInterpolator,
                    'barycentric': barycentric_interpolate,
                    'regulargrid': RegularGridInterpolator,
                      'bivariate': RectBivariateSpline,
                          'akima': Akima1DInterpolator,
                          'krogh': krogh_interpolate,
                          'pchip': pchip_interpolate,
                       'interp1d': interp1d,
                       'interp2d': interp2d,
                       'griddata': griddata}
    # Handle 1-dimensional data first
    if oned is not None:
        if method not in ['interp1d', 'akima']:
            raise Exception('One-dimensional interpolation must use '
                            '"interp1d" or "aklima" method')
        kwargs = {'kind': kind} if method == 'interp1d' else {}
        xdat = df[oned].values
        if not df[z].isnull().values.any(): zdat = df[z].values
        else:
            print('Missing data is interpolated with a 3rd order polynomial.')
            zdat = df[z].interpolate(method='piecewise_polynomial', order=3)
        newx = np.linspace(xdat.min(), xdat.max(), dim)
        interpz = convenience[method](xdat, zdat, **kwargs)
        newz = interpz(newx)
        return {'x': newx, 'z': newz, 'y': df[dud].unique(),
                'min': (newx[newz.argmin()], newz.min())}
    # Check that the interpolation method is supported
    if method not in convenience.keys():
        raise Exception('method must be in {}'.format(convenience.keys()))
    # Shape the data in df
    pivot = df.pivot(x, y, z)
    if pivot.isnull().values.any():
        print('Missing data is interpolated with a 3rd order piecewise polynomial.\n'
              'End points are extrapolated with a fit function of 3rd order.')
        pivot.interpolate(method='piecewise_polynomial', order=3, axis=1, inplace=True)
        # Obtained from SO: http://stackoverflow.com/questions/22491628/extrapolate-values-in-pandas-dataframe
        # Function to curve fit to the data
        def func(x, a, b, c, d):
            return a * (x ** 3) + b * (x ** 2) + c * x + d
        # Initial parameter guess, just to kick off the optimization
        guess = (0.5, 0.5, 0.5, 0.5)
        # Create copy of data to remove NaNs for curve fitting
        fit_df = pivot.dropna()
        # Place to store function parameters for each column
        col_params = {}
        # Curve fit each column
        for col in fit_df.columns:
            # Get x & y
            x = fit_df.index.astype(float).values
            y = fit_df[col].values
            # Curve fit column and get curve parameters
            params = curve_fit(func, x, y, guess)
            # Store optimized parameters
            col_params[col] = params[0]
        # Extrapolate each column
        for col in pivot.columns:
        # Get the index values for NaNs in the column
            x = pivot[pd.isnull(pivot[col])].index.astype(float).values
            # Extrapolate those points with the fitted function
            pivot.loc[x, col] = func(x, *col_params[col])

    xdat = pivot.index.values
    ydat = pivot.columns.values
    zdat = pivot.values
    # New (x, y) values
    newx = np.linspace(xdat.min(), xdat.max(), dim)
    newy = np.linspace(ydat.min(), ydat.max(), dim)
    # Details of the implementation in scipy
    # First 5 are explicitly 2D interpolation
    if method == 'bivariate':
        interpz = convenience[method](xdat, ydat, zdat)
        newz = interpz(newx, newy).T
    elif method == 'interp2d':
        interpz = convenience[method](xdat, ydat, zdat.T, kind=kind)
        newz = interpz(newx, newy)
    elif method in ['griddata', 'cloughtocher', 'regulargrid']:
        meshx, meshy = np.meshgrid(xdat, ydat)
        newmeshx, newmeshy = np.meshgrid(newx, newy)
        points = np.array([meshx.flatten(order='F'),
                           meshy.flatten(order='F')]).T
        newpoints = np.array([newmeshx.flatten(order='F'),
                              newmeshy.flatten(order='F')]).T
        if method == 'cloughtocher':
            interpz = convenience[method](points, zdat.flatten())
            newz = interpz(newpoints)
            newz = newz.reshape((dim, dim), order='F')
        elif method == 'regulargrid':
            interpz = convenience[method]((xdat, ydat), zdat)
            newz = interpz(newpoints)
            newz = newz.reshape((dim, dim), order='F')
        else:
            newz = convenience[method](points, zdat.flatten(), newpoints)
            newz = newz.reshape((dim, dim), order='F')
    # 1D interpolation applied across both x and y
    else:
        # Not sure if we need this complexity but interesting to see if
        # the order of interpolation matters (based on method)
        newz = np.empty((dim, dim), dtype=np.float64)
        kwargs = {'kind': kind} if method == 'interp1d' else {}
        if yfirst:
            partz = np.empty((xdat.shape[0], dim), dtype=np.float64)
            if method in ['interp1d', 'akima']:
                for i in range(xdat.shape[0]):
                    zfunc = convenience[method](ydat, zdat[i,:], **kwargs)
                    partz[i] = zfunc(newy)
                for i in range(dim):
                    zfunc = convenience[method](xdat, partz[:,i], **kwargs)
                    newz[i,:] = zfunc(newy)
                newz = newz[::-1,::-1]
            else:
                for i in range(xdat.shape[0]):
                    partz[i] = convenience[method](ydat, zdat[i,:], newy)
                for i in range(dim):
                    newz[i,:] = convenience[method](xdat, partz[:,i], newx)
        else:
            partz = np.empty((ydat.shape[0], dim), dtype=np.float64)
            if method in ['interp1d', 'akima']:
                for i in range(ydat.shape[0]):
                    zfunc = convenience[method](xdat, zdat[:,i], **kwargs)
                    partz[i] = zfunc(newx)
                for i in range(dim):
                    zfunc = convenience[method](ydat, partz[:,i], **kwargs)
                    newz[:,i] = zfunc(newy)
            else:
                for i in range(ydat.shape[0]):
                    partz[i] = convenience[method](xdat, zdat[:,i], newx)
                for i in range(dim):
                    newz[:,i] = convenience[method](ydat, partz[:,i], newy)
    # Find minimum values for the interpolated data set
    minima = None
    if minimum:
        minima = np.empty((dim, 3), dtype=np.float64)
        for i, arr in enumerate(newz):
            minima[i] = (newx[arr.argmin()], newy[i], arr.min())
        minima = pd.DataFrame(minima)
        # Smooth this out as it can be quite jagged
        window = dim - (1 - dim % 2)
        minima[1] = savgol_filter(minima[1], window, 3)
    return {'x': newx, 'y': newy, 'z': newz, 'min': minima}


# Sample of a wrapper around the hidden function for public API
def interpolate_j2(df, method='interp2d', kind='cubic', yfirst=False,
                   dim=21, minimum=False):
    """
    Given a dataframe containing alpha, gamma, j2 columns,
    return a dictionary for plotting.
    """
    interped = _interpolate(df, 'alpha', 'gamma', 'j2',
                            method, kind, yfirst, dim, minimum)
    for key, cart in [('alpha', 'x'), ('gamma', 'y'), ('j2', 'z')]:
        interped[key] = interped.pop(cart)
    return interped
