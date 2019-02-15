# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import ticker

class PlotVROA:
    @staticmethod
    def _lorentz(freq, inten, x, fwhm):
        y = np.zeros(len(x))
        for i, (fdx, idx) in enumerate(zip(freq, inten)):
            y += 1/(2*np.pi)*idx*fwhm/((fdx-x)**2+(0.5*fwhm)**2)
        return y

#    @staticmethod
#    def _normalize
        
    def single_frequency(self, vroa, **kwargs):
        #if not (isinstance(freq, list) or isinstance(freq, np.ndarray)):
        #    raise TypeError("freq data input array must be a list or np.ndarray not {}".format(type(freq)))
        #if not (isinstance(inten, list) or isinstance(inten, np.ndarray)):
        #    raise TypeError("inten data input arrainten must be a list or np.ndarray not {}".format(tintenpe(inten)))
        #if not hasattr(vroa, "calcfreq"):
        #    raise AttributeError("Please compute calcfreq dataframe")
        if not hasattr(vroa, "scatter"):
            raise AttributeError("Please compute scatter dataframe")

        forward = kwargs.pop('forw', False)
        backward = kwargs.pop('back', False)
        if not(forward or backward):
            raise ValueError("Must set forward (forw) or backward (back) scattering variables to True")
        elif forward and backward:
            raise ValueError("Can only set forward (forw) or backward (back) scattering variables to True. Both are True.")
        if forward:
            sct = 'forwardscatter'
        elif backward:
            sct = 'backscatter'
        title = kwargs.pop('title', '')
        xlabel = kwargs.pop('xlabel', '')
        ylabel = kwargs.pop('ylabel', '')
        marker = kwargs.pop('marker', 'o')
        line = kwargs.pop('line', '-')
        figsize = kwargs.pop('figsize', (8,8))
        dpi = kwargs.pop('dpi', 50)
        xlim = kwargs.pop('xrange', None)
        ylim = kwargs.pop('yrange', None)
        fwhm = kwargs.pop('fwhm', 15) #in cm^-1
        res = kwargs.pop('res', 1) #in cm^-1
        grid = kwargs.pop('grid', False)
        legend = kwargs.pop('legend', True)
        exc_units = kwargs.pop('exc_units', 'nm')

        if not isinstance(figsize, tuple):
            raise TypeError("figsize must be a tuple not {}".format(type(figsize)))
        grouped = vroa.scatter.groupby('exc_freq')
        exc_freq = vroa.scatter['exc_freq'].drop_duplicates().values
        self.single_freq = []
        for idx, val in enumerate(exc_freq):
            fig = plt.figure(idx, figsize=figsize, dpi=dpi)
            inten = grouped.get_group(val)[sct].values
            freq = grouped.get_group(val)['freq'].values
            if xlim is None:
                x = np.arange(freq[0]-fwhm*3, freq[-1]+fwhm*3, res)
            else:
                x = np.arange(xlim[0], xlim[1], res)
            y = self._lorentz(freq=freq, inten=inten, x=x, fwhm=fwhm)

            ax = fig.add_subplot(111)
            ax.plot(x,y,marker=marker,linestyle=line,
                    label=str(val)+' '+exc_units if val is not -1 else "unk")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            if grid:
                ax.grid(grid)
            if legend:
                ax.legend()
            self.single_freq.append(fig)

    def multiple_frequency(self, vroa, **kwargs):
        #if not (isinstance(freq, list) or isinstance(freq, np.ndarray)):
        #    raise TypeError("freq data input array must be a list or np.ndarray not {}".format(type(freq)))
        #if not (isinstance(inten, list) or isinstance(inten, np.ndarray)):
        #    raise TypeError("inten data input arrainten must be a list or np.ndarray not {}".format(tintenpe(inten)))
        #if not hasattr(vroa, "calcfreq"):
        #    raise AttributeError("Please compute calcfreq dataframe")
        if not hasattr(vroa, "scatter"):
            raise AttributeError("Please compute scatter dataframe")

        forward = kwargs.pop('forw', False)
        backward = kwargs.pop('back', False)
        if not(forward or backward):
            raise ValueError("Must set forward (forw) or backward (back) scattering variables to True")
        elif forward and backward:
            raise ValueError("Can only set forward (forw) or backward (back) scattering variables to True. Both are True.")
        if forward:
            sct = 'forwardscatter'
        elif backward:
            sct = 'backscatter'
        title = kwargs.pop('title', '')
        xlabel = kwargs.pop('xlabel', '')
        ylabel = kwargs.pop('ylabel', '')
        marker = kwargs.pop('marker', 'o')
        line = kwargs.pop('line', '-')
        figsize = kwargs.pop('figsize', (8,8))
        dpi = kwargs.pop('dpi', 50)
        xlim = kwargs.pop('xrange', None)
        ylim = kwargs.pop('yrange', None)
        fwhm = kwargs.pop('fwhm', 15) #in cm^-1
        res = kwargs.pop('res', 1) #in cm^-1
        grid = kwargs.pop('grid', True)
        legend = kwargs.pop('legend', True)
        exc_units = kwargs.pop('exc_units', 'nm')
        normalize = kwargs.pop('normalize', 'all')

        if not isinstance(figsize, tuple):
            raise TypeError("figsize must be a tuple not {}".format(type(figsize)))
        grouped = vroa.scatter.groupby('exc_freq')
        exc_freq = vroa.scatter['exc_freq'].drop_duplicates().values
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        norm = []
        if normalize == 'max':
            norm = round(abs(vroa.scatter[sct].abs().max())*2/(np.pi*fwhm),4)
        for idx, val in enumerate(exc_freq):
            inten = grouped.get_group(val)[sct].values
            freq = grouped.get_group(val)['freq'].values
            if xlim is None:
                x = np.arange(freq[0]-fwhm*3, freq[-1]+fwhm*3, res)
            else:
                x = np.arange(xlim[0], xlim[1], res)
            y = self._lorentz(freq=freq, inten=inten, x=x, fwhm=fwhm)
            if normalize == 'max':
                y /= norm
            else:
                norm.append(round(max(abs(y)),4))
                y /= max(abs(y))
            y += idx*2
            ax.plot(x,y,marker=marker,linestyle=line,
                    label=str(val)+' '+exc_units if val is not -1 else "unk")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if grid:
            ax.yaxis.grid(b=grid, which='major', color='k')
            ax.yaxis.grid(b=grid, which='minor', color='k', linestyle='-', linewidth=4.0)
            ax.xaxis.grid(b=grid)
        if legend:
            ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
        majors = np.arange(0, len(exc_freq)*2, 2)
        minors = majors + 1
        minors = np.insert(minors, 0, majors[0]-1)
        if normalize == 'max':
            norm = np.repeat(norm, len(majors))
        ax.yaxis.set_major_locator(ticker.FixedLocator(majors))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(minors))
        #ax.set_yticks(np.arange(len(exc_freq)*2, 2), norm)
        ax.set_yticklabels(['{:4.0f}'.format(n) for n in norm])
        fig.tight_layout()
        self.multiple_freq = fig

