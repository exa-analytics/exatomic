# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
'''
Plotter
#################
This is supposed to be a collection of classes and functions to aid in plotting
'''
from bokeh.plotting import output_notebook, show, figure
from bokeh.models.ranges import Range1d
import numpy as np

def lorentzian(freq, x, fwhm, inten=None):
    '''
    Plot lorentzian lineshapes

    Args:
        freq (np.ndarray): Frequencies where the peaks will be located
        x (np.ndarray): X-axis data points
        fwhm (float): Full-width at half maximum
        inten (np.ndarray): Intensities of the peaks

    Returns:
        y (np.ndarray): Y values of the lorentzian lineshapes
    '''
    y = np.zeros(len(x))
    if inten is None:
        for fdx in freq:
            y += 1/(2*np.pi)*fwhm/((fdx-x)**2+(0.5*fwhm)**2)
    else:
        for fdx, idx in zip(freq, inten):
            y += 1/(2*np.pi)*idx*fwhm/((fdx-x)**2+(0.5*fwhm)**2)
    return y

def gaussian(freq, x, fwhm, inten=None):
    '''
    Plot gaussian lineshapes

    Args:
        freq (np.ndarray): Frequencies where the peaks will be located
        x (np.ndarray): X-axis data points
        fwhm (float): Full-width at half maximum
        inten (np.ndarray): Intensities of the peaks

    Returns:
        y (np.ndarray): Y values of the gaussian lineshapes
    '''
    y = np.zeros(len(x))
    sigma = fwhm/(np.sqrt(8*np.log(2)))
    if inten is None:
        for fdx in freq:
            y += 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-fdx)**2/(2*sigma**2))
    else:
        for idx, fdx in zip(inten, freq):
            y += 1/(sigma*np.sqrt(2*np.pi))*idx*np.exp(-(x-fdx)**2/(2*sigma**2))
    return y

class Plot:
    '''
    Class that has a collection of methods to make plotting easier. Some of the bokeh functions
    require importing specific functions like 'show' to display the figure. We want to make
    this easier by defining methods like show so we can just import the class and it takes
    care of everything.
    '''
    def show(self):
        # method just to have simple show function like in matplotlib
        show(self.fig)

    def set_xrange(self, xmin, xmax):
        # set the xrange
        # makes it simple to flip the xaxis by giving the max value as the
        # xmin and the min value as the xmax
        self.fig.x_range = Range1d(xmin, xmax)

    def set_yrange(self, ymin, ymax):
        # set the yrange
        self.fig.y_range = Range1d(ymin, ymax)

    def __init__(self, *args, **kwargs):
        # this worries me a bit and not sure if this is the proper way to do this
        output_notebook()
        # set the title
        title = kwargs.pop('title', '')
        # set the plot area parameters
        plot_width = kwargs.pop('plot_width', 500)
        plot_height = kwargs.pop('plot_height', 500)
        # set the tools to be used
        tools = kwargs.pop('tools', 'hover, crosshair, pan, wheel_zoom, box_zoom, reset, save,')
        # create the figure
        self.fig = figure(title=title, plot_height=plot_height, plot_width=plot_width, tools=tools)

# a matplotlib example
# maybe we can make some conditional so you can use a bokeh plot or a matplotlib plot
# might be useful if we just want to display the plot right on the screen as opposed
# to having the plot on the web browser
#
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import ticker, rc
#
#class Plot:
#    def lorentz(freq, x, fwhm, inten=None):
#        y = np.zeros(len(x))
#        if inten is None:
#            for fdx in freq:
#                y += 1/(2*np.pi)*fwhm/((fdx-x)**2+(0.5*fwhm)**2)
#        else:
#            for fdx, idx in zip(freq, inten):
#                y += 1/(2*np.pi)*idx*fwhm/((fdx-x)**2+(0.5*fwhm)**2)
#        return y
#
#    def __init__(self, *args, **kwargs):
#        title = kwargs.pop('title', '')
#        xlabel = kwargs.pop('xlabel', '')
#        ylabel = kwargs.pop('ylabel', '')
#        marker = kwargs.pop('marker', '')
#        line = kwargs.pop('line', '-')
#        figsize = kwargs.pop('figsize', (8,8))
#        dpi = kwargs.pop('dpi', 50)
#        xrange = kwargs.pop('xrange', None)
#        yrange = kwargs.pop('yrange', None)
#        fwhm = kwargs.pop('fwhm', 15)
#        res = kwargs.pop('res', 1)
#        grid = kwargs.pop('grid', False)
#        legend = kwargs.pop('legend', True)
#        invert_x = kwargs.pop('invert_x', False)
#        font = kwargs.pop('font', 10)
#        lorentz = kwargs.pop('lorentz', True)
#        self.fig = plt.figure(figsize=figsize, dpi=dpi)
#        rc('font', size=font)


