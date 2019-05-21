# -*- coding: utf-8 -*-
# Copyright (c) 2015-2019, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

from bokeh.plotting import output_notebook, show, figure
from bokeh.models.ranges import Range1d
import numpy as np

def lorentzian(freq, x, fwhm, inten=None):
    y = np.zeros(len(x))
    if inten is None:
        for fdx in freq:
            y += 1/(2*np.pi)*fwhm/((fdx-x)**2+(0.5*fwhm)**2)
    else:
        for fdx, idx in zip(freq, inten):
            y += 1/(2*np.pi)*idx*fwhm/((fdx-x)**2+(0.5*fwhm)**2)
    return y

def gaussian(freq, x, fwhm, inten=None):
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
    def show(self):
        show(self.fig)

    def set_xrange(self, xmin, xmax):
        self.fig.x_range = Range1d(xmin, xmax)

    def set_yrange(self, ymin, ymax):
        self.fig.y_range = Range1d(ymin, ymax)

    def __init__(self, *args, **kwargs):
        output_notebook()
        title = kwargs.pop('title', '')
        plot_width = kwargs.pop('plot_width', 500)
        plot_height = kwargs.pop('plot_height', 500)
        tools = kwargs.pop('tools', 'hover, crosshair, pan, wheel_zoom, box_zoom, reset, save,')

        self.fig = figure(title=title, plot_height=plot_height, plot_width=plot_width, tools=tools)

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
        
 
