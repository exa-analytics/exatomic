# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Custom Axes
###################
"""

import seaborn as sns

from exa.mpl import _plot_contour, _plot_surface
from exatomic import Energy


def _get_minimum(mindf):
    absmin = mindf[mindf[2] == mindf[2].min()]
    idxs = mindf[(mindf[0] > 0) & (mindf[1] > 0)].index.values
    id0, id1 = idxs[:2]
    cnt = 1
    try:
        while np.isclose(id0 + 1, id1):
            id0, id1 = idxs[cnt:cnt + 2]
            cnt += 1
        slc = slice(idxs[0], id0 + 1)
        amin = mindf.ix[idxs[0]:id0 + 1]
    except:
        if absmin.index[0] in idxs:
            slc = list(idxs) + [idxs[-1] + 1]
            amin = mindf.ix[idxs]
        else:
            slc = list(idxs) + list(absmin.index.values)
    return mindf.ix[slc]


def plot_j2_surface(data, key='j2', method='wireframe', nxlabel=6,
                    nylabel=6, nzlabel=6, minimum=False, figsize=(8,6),
                    alpha=0.5, cmap=None, title=None):
    cmap = sns.mpl.pyplot.cm.get_cmap('coolwarm') if cmap is None else cmap
    figargs = {'figsize': figsize}
    axargs = {'alpha': alpha, 'cmap': cmap}
    fig = _plot_surface(data['alpha'], data['gamma'], data['j2'],
                        nxlabel, nylabel, nzlabel, method, figargs, axargs)
    ax = fig.gca()
    if 'min' in data and minimum:
        mindf = _get_minimum(data['min'])
        ax.plot(mindf[0], mindf[1], mindf[2], color='k', zorder=2)
    ax.set_ylabel(r'$\gamma$')
    ax.set_xlabel(r'$\\\alpha$')
    ax.set_zlabel(r'J$^{2}$')
    if title is not None:
        ax.set_title(title)
    return fig


def plot_j2_contour(data, vmin=None, vmax=None, key='j2', figsize=(8,6),
                    nxlabel=6, nylabel=6, method='pcolor', cmap=None, title=None,
                    minline=False, minpoint=False, legend=False, colorbar=False):
    vmin = data[key].min() if vmin is None else vmin
    vmax = data[key].max() if vmax is None else vmax
    cmap = sns.mpl.pyplot.cm.get_cmap('coolwarm') if cmap is None else cmap
    figargs = {'figsize': figsize}
    axargs = {'vmin': vmin, 'vmax': vmax, 'cmap': cmap,
              'zorder': 1, 'rasterized': True}
    fig, cbar = _plot_contour(data['alpha'], data['gamma'], data[key],
                              nxlabel, nylabel, method, colorbar, figargs, axargs)
    ax = fig.gca()
    if (minline or minpoint) and 'min' in data:
        mindf = _get_minimum(data['min'])
        if minline:
            ax.plot(mindf[0], mindf[1], label='Min.(J$^{2}$)', color='k', zorder=2)
        if minpoint:
            jmin = mindf[2].argmin()
            labl = '({:.4f},{:.4f})'.format(mindf[0][jmin], mindf[1][jmin])
            ax.scatter([mindf[0][jmin]], [mindf[1][jmin]], label=labl,
                       marker='*', color='y', s=200, zorder=3)
        if legend:
            hdls, lbls = ax.get_legend_handles_labels()
            leg = ax.legend(hdls, lbls)
            leg.get_frame().set_alpha(0.5)
    ax.set_ylabel(r'$\gamma$')
    ax.set_xlabel(r'$\\\alpha$')
    if title is not None:
        ax.set_title(title)
    return fig

def photoelectron_spectrum(unis, broaden=0.06, window=None, shift=0,
                           label=None, color=None, stepe=1, units='eV',
                           fontsize=20, peaklabels=True, xlim=None, ylim=None,
                           extra=None, figsize=(10,10)):
    unis = [unis] if not isinstance(unis, list) else unis
    if window is None:
        window = []
        for i, uni in enumerate(unis):
            uni.orbital[units] = uni.orbital['energy'] * Energy['Ha', units]
            window.append([uni.orbital.get_orbital(orb=-15)[units],
                           uni.orbital.get_orbital()[units]])
    else:
        if not isinstance(window, list): window = window * len(unis)
    if shift or not isinstance(shift, list):
        shift = [shift] * len(unis)
    else:
        if not isinstance(shift, list): shift = shift * len(unis)

#def photoelectron_spectrum(ax, unis, window=[-10, 0], broaden=0.6,
#                           shift=0, label='', color=None, stepe=1, units='eV',
#                           loc='upper left', fontsize=26, peaks=True,
#                           xlim=None, ylim=None):
#    color = ['r', 'g', 'b', 'c', 'y', 'm', 'k'] if color is None else color
#    arrowprops = {'arrowstyle': '->', 'connectionstyle': 'arc3'}
#    arrowargs = {'xycoords': 'data', 'textcoords': 'data',
#                 'arrowprops': arrowprops, 'fontsize': fontsize}
#    unis = [unis] if not isinstance(unis, list) else unis
#    xmin, xmax = [], []
#    if (len(window) != len(unis) or len(unis) == 2): window = window * len(unis)
#    if not isinstance(shift, list): shift = [shift] * len(unis)
#    if not isinstance(label, list): label = [label] * len(unis)
#    for i, uni in enumerate(unis):
#        height = len(unis) - 1 - i
#        lo, hi = window[i]
#        pes = uni.orbital.convolve(ewin=[lo,hi], broaden=broaden, units=units)[::-1]
#        pes[units] = -pes[units]
#        pes['shifted'] = pes[units] + shift[i]
#        heightened = pes['signal'] + height
#        lab = uni.name if uni.name and not label[i] else label[i]
#        ax.axhline(height, color='k', linewidth=1.2)
#        ax.plot(pes['shifted'], heightened, label=lab, color=color[i % len(color)])
#        o = uni.orbital
#        o = o[(o[units] > lo) & (o[units] < hi) & (o['occupation'] > 0)].drop_duplicates(
#            units).copy().drop_duplicates('vector').sort_values(
#            by=units, ascending=False).reset_index()
#        o[units] = -o[units]
#        leno = len(o)
#        switch = leno // 2
#        nonzero = pes[pes['signal'] > 0.1]['shifted']
#        small = nonzero.min()
#        esmall = small - stepe * switch
#        elarge = nonzero.max()
#        xmin.append(esmall)
#        dh = 1 / (switch + 3)
#        hlo = height + dh
#        hhi = height + (switch + switch % 2) * dh
#        for t in range(-20, 20):
#            ax.plot([t] * 2, [height, height - 0.05], color='k', linewidth=1)
#        if peaks:
#            for c, (sym, en) in enumerate(zip(o['symmetry'], o[units] + shift[i])):
#                ax.plot([en] * 2, [height, height + 0.05], color='k', linewidth=1)
#                astr = r'$' + sym[0].lower() + '_{' + sym[1:].lower() + '}$'
#                e = esmall if c < switch else elarge
#                h = hlo if c < switch else hhi
#                ax.annotate(astr, xy=(en, height + 0.05), xytext=(e, h), **arrowargs)
#                if c < switch:
#                    esmall += stepe
#                    hlo += dh
#                else:
#                    elarge += stepe * 1.5
#                    hhi -= dh
#            xmax.append(elarge)
#    xax = 'E* (' + units + ')' if any((i for i in shift)) else 'E (' + units + ')'
#    xlim = (min(xmin), max(xmax)) if xlim is None else xlim
#    ylim = (0, len(unis)) if ylim is None else ylim
#    ax.set_xlim(xlim)
#    ax.set_ylim(ylim)
#    ax.set_xlabel(xax)
#    ax.legend(loc=loc)
#    return ax
