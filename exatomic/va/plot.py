# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, rc

def _lorentz(freq, inten, x, fwhm):
    y = np.zeros(len(x))
    for fdx, idx in zip(freq, inten):
        y += 1/(2*np.pi)*idx*fwhm/((fdx-x)**2+(0.5*fwhm)**2)
    return y

class PlotVROA:
    # variable to keep track of the figure count
    # more important when using it outside of a jupyter notebook
    _fig_count = 0
    # list object that will hold all of the plots that are created
    vroa = []
    raman = []
    def single_vroa(self, vroa, **kwargs):
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
        marker = kwargs.pop('marker', '')
        line = kwargs.pop('line', '-')
        figsize = kwargs.pop('figsize', (8,8))
        dpi = kwargs.pop('dpi', 50)
        xrange = kwargs.pop('xrange', None)
        yrange = kwargs.pop('yrange', None)
        fwhm = kwargs.pop('fwhm', 15)
        res = kwargs.pop('res', 1)
        grid = kwargs.pop('grid', False)
        legend = kwargs.pop('legend', True)
        exc_units = kwargs.pop('exc_units', 'nm')
        invert_x = kwargs.pop('invert_x', False)
        font = kwargs.pop('font', 10)

        if not isinstance(figsize, tuple):
            raise TypeError("figsize must be a tuple not {}".format(type(figsize)))
        rc('font', size=font)
        grouped = vroa.scatter.groupby('exc_freq')
        exc_freq = vroa.scatter['exc_freq'].drop_duplicates()
        for _, val in enumerate(exc_freq):
            fig = plt.figure(self._fig_count, figsize=figsize, dpi=dpi)
            inten = grouped.get_group(val)[sct].values
            freq = grouped.get_group(val)['freq'].values
            if xrange is None:
                x = np.arange(freq[0]-fwhm*3, freq[-1]+fwhm*3, res)
            else:
                x = np.arange(xrange[0], xrange[1], res)
            y = _lorentz(freq=freq, inten=inten, x=x, fwhm=fwhm)
            #y_bar = _lorentz(freq=freq, inten=inten, x=x, fwhm=fwhm)

            ax = fig.add_subplot(111)
            ax.plot(x,y,marker=marker,linestyle=line,
                    label=str(val)+' '+exc_units if val is not -1 else "unk")
            #ax.bar(freq, y_bar*0.5, width=fwhm*0.35)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if xrange is not None:
                if invert_x:
                    ax.set_xlim(xrange[1],xrange[0])
                else:
                    ax.set_xlim(xrange)
            if yrange is not None:
                ax.set_ylim(yrange)
            if grid:
                ax.grid(grid)
            if legend:
                ax.legend()
            fig.tight_layout()
            self._fig_count += 1
            self.vroa.append(fig)

    def multiple_vroa(self, vroa, **kwargs):
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
        marker = kwargs.pop('marker', '')
        line = kwargs.pop('line', '-')
        figsize = kwargs.pop('figsize', (8,8))
        dpi = kwargs.pop('dpi', 50)
        xrange = kwargs.pop('xrange', None)
        yrange = kwargs.pop('yrange', None)
        fwhm = kwargs.pop('fwhm', 15)
        res = kwargs.pop('res', 1)
        grid = kwargs.pop('grid', True)
        legend = kwargs.pop('legend', True)
        exc_units = kwargs.pop('exc_units', 'nm')
        normalize = kwargs.pop('normalize', 'all')
        invert_x = kwargs.pop('invert_x', False)
        font = kwargs.pop('font', 10)

        if not isinstance(figsize, tuple):
            raise TypeError("figsize must be a tuple not {}".format(type(figsize)))
        rc('font', size=font)
        grouped = vroa.scatter.groupby('exc_freq')
        exc_freq = vroa.scatter['exc_freq'].drop_duplicates().values
        fig = plt.figure(self._fig_count, figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        norm = []
        if normalize == 'max':
            norm = round(abs(vroa.scatter[sct].abs().max())*2/(np.pi*fwhm),4)
        for idx, val in enumerate(exc_freq):
            inten = grouped.get_group(val)[sct].values
            freq = grouped.get_group(val)['freq'].values
            if xrange is None:
                x = np.arange(freq[0]-fwhm*3, freq[-1]+fwhm*3, res)
            else:
                x = np.arange(xrange[0], xrange[1], res)
            y = _lorentz(freq=freq, inten=inten, x=x, fwhm=fwhm)
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
        if xrange is not None:
            if invert_x:
                ax.set_xlim(xrange[1],xrange[0])
            else:
                ax.set_xlim(xrange)
        if yrange is not None:
            ax.set_ylim(yrange)
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
        ax.set_yticklabels(['{:4.3E}'.format(n) for n in norm])
        fig.tight_layout()
        self._fig_count += 1
        self.vroa.append(fig)

    def single_raman(self, raman, **kwargs):
        if not hasattr(raman, "raman"):
            raise AttributeError("Please compute raman dataframe")
        title = kwargs.pop('title', '')
        xlabel = kwargs.pop('xlabel', '')
        ylabel = kwargs.pop('ylabel', '')
        marker = kwargs.pop('marker', '')
        line = kwargs.pop('line', '-')
        figsize = kwargs.pop('figsize', (8,8))
        dpi = kwargs.pop('dpi', 50)
        xrange = kwargs.pop('xrange', None)
        yrange = kwargs.pop('yrange', None)
        fwhm = kwargs.pop('fwhm', 15)
        res = kwargs.pop('res', 1)
        grid = kwargs.pop('grid', False)
        legend = kwargs.pop('legend', True)
        exc_units = kwargs.pop('exc_units', 'nm')
        invert_x = kwargs.pop('invert_x', False)
        font = kwargs.pop('font', 10)

        if not isinstance(figsize, tuple):
            raise TypeError("figsize must be a tuple not {}".format(type(figsize)))
        rc('font', size=font)
        grouped = raman.raman.groupby('exc_freq')
        exc_freq = raman.raman['exc_freq'].drop_duplicates()
        for _, val in enumerate(exc_freq):
            fig = plt.figure(self._fig_count, figsize=figsize, dpi=dpi)
            inten = grouped.get_group(val)['raman_int'].values
            freq = grouped.get_group(val)['freq'].values
            if xrange is None:
                x = np.arange(freq[0]-fwhm*3, freq[-1]+fwhm*3, res)
            else:
                x = np.arange(xrange[0], xrange[1], res)
            y = _lorentz(freq=freq, inten=inten, x=x, fwhm=fwhm)
            #y_bar = _lorentz(freq=freq, inten=inten, x=x, fwhm=fwhm)

            ax = fig.add_subplot(111)
            ax.plot(x,y,marker=marker,linestyle=line,
                    label=str(val)+' '+exc_units if val is not -1 else "unk")
            #ax.bar(freq, y_bar*0.35, width=fwhm*0.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if xrange is not None:
                if invert_x:
                    ax.set_xlim(xrange[1],xrange[0])
                else:
                    ax.set_xlim(xrange)
            if yrange is not None:
                ax.set_ylim(yrange)
            if grid:
                ax.grid(grid)
            if legend:
                ax.legend()
            self._fig_count += 1
            self.raman.append(fig)

