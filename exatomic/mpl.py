# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Custom Axes
###################
"""
#
#import seaborn as sns
#
#from exa.mpl import _plot_contour, _plot_surface
#from exatomic import Energy
#
#
#def _get_minimum(mindf):
#    absmin = mindf[mindf[2] == mindf[2].min()]
#    idxs = mindf[(mindf[0] > 0) & (mindf[1] > 0)].index.values
#    id0, id1 = idxs[:2]
#    cnt = 1
#    try:
#        while np.isclose(id0 + 1, id1):
#            id0, id1 = idxs[cnt:cnt + 2]
#            cnt += 1
#        slc = slice(idxs[0], id0 + 1)
#        amin = mindf.ix[idxs[0]:id0 + 1]
#    except:
#        if absmin.index[0] in idxs:
#            slc = list(idxs) + [idxs[-1] + 1]
#            amin = mindf.ix[idxs]
#        else:
#            slc = list(idxs) + list(absmin.index.values)
#    return mindf.ix[slc]
#
#
#def plot_j2_surface(data, key='j2', method='wireframe', nxlabel=6,
#                    nylabel=6, nzlabel=6, minimum=False, figsize=(8,6),
#                    alpha=0.5, cmap=None, title=None):
#    cmap = sns.mpl.pyplot.cm.get_cmap('coolwarm') if cmap is None else cmap
#    figargs = {'figsize': figsize}
#    axargs = {'alpha': alpha, 'cmap': cmap}
#    fig = _plot_surface(data['alpha'], data['gamma'], data['j2'],
#                        nxlabel, nylabel, nzlabel, method, figargs, axargs)
#    ax = fig.gca()
#    if 'min' in data and minimum:
#        mindf = _get_minimum(data['min'])
#        ax.plot(mindf[0], mindf[1], mindf[2], color='k', zorder=2)
#    ax.set_ylabel(r'$\gamma$')
#    ax.set_xlabel(r'$\\\alpha$')
#    ax.set_zlabel(r'J$^{2}$')
#    if title is not None:
#        ax.set_title(title)
#    return fig
#
#
#def plot_j2_contour(data, vmin=None, vmax=None, key='j2', figsize=(8,6),
#                    nxlabel=6, nylabel=6, method='pcolor', cmap=None, title=None,
#                    minline=False, minpoint=False, legend=False, colorbar=False):
#    vmin = data[key].min() if vmin is None else vmin
#    vmax = data[key].max() if vmax is None else vmax
#    cmap = sns.mpl.pyplot.cm.get_cmap('coolwarm') if cmap is None else cmap
#    figargs = {'figsize': figsize}
#    axargs = {'vmin': vmin, 'vmax': vmax, 'cmap': cmap,
#              'zorder': 1, 'rasterized': True}
#    fig, cbar = _plot_contour(data['alpha'], data['gamma'], data[key],
#                              nxlabel, nylabel, method, colorbar, figargs, axargs)
#    ax = fig.gca()
#    if (minline or minpoint) and 'min' in data:
#        mindf = _get_minimum(data['min'])
#        if minline:
#            ax.plot(mindf[0], mindf[1], label='Min.(J$^{2}$)', color='k', zorder=2)
#        if minpoint:
#            jmin = mindf[2].argmin()
#            labl = '({:.4f},{:.4f})'.format(mindf[0][jmin], mindf[1][jmin])
#            ax.scatter([mindf[0][jmin]], [mindf[1][jmin]], label=labl,
#                       marker='*', color='y', s=200, zorder=3)
#        if legend:
#            hdls, lbls = ax.get_legend_handles_labels()
#            leg = ax.legend(hdls, lbls)
#            leg.get_frame().set_alpha(0.5)
#    ax.set_ylabel(r'$\gamma$')
#    ax.set_xlabel(r'$\\\alpha$')
#    if title is not None:
#        ax.set_title(title)
#    return fig
#
#def photoelectron_spectrum(*unis, filters=None, broaden=0.06, color=None,
#                           stepe=1, units='eV', fontsize=20, peaklabels=True,
#                           xlim=None, extra=None, figsize=(10,10)):
#    """
#    Plot what is essentially a density of states for any number of universes,
#    attempting to associate symmetry labels in order of peak positions.
#
#    Args
#        unis (exatomic.container.Universe): any number of universes with orbitals
#        filters (dict,list): dict or list of dicts for each universe
#            accepted kwargs: 'shift', uni.orbital column names
#            special kwargs: 'shift' shifts energies,
#                ['energy', 'eV', units] must be in the form of [min, max]
#            Note: values can be strings defining relationships like
#                {'occupation': '> 0'}
#        units (str): the units in which to display the spectrum
#        broaden (float): how broad to convolute each orbital energy (FWHM gaussian)
#        color (list): commonly sns.color_palette or ['r', 'g', 'b', ...]
#        stepe (int,float): how far to separate symmetry labels on plot (modify for
#            units other than 'eV')
#        fontsize (int): font size of text on plot (symmetry labels are fontsize - 2)
#        peaklabels (bool): if True and symmetry in uni.orbital, put labels on plots
#        xlim (tuple): (xmin, xmax)
#        extra (dict): Custom plot of additional data on the same figure object
#            accepted kwargs: ['x', 'y', 'color', 'label']
#        figsize (tuple): matplotlib.figure.Figure figuresize keyword arg
#
#    Returns
#        fig (matplotlib.figure.Figure): the plot
#    """
#    pass
##    unis = [unis] if not isinstance(unis, list) else unis
##    if window is None:
##        window = []
##        for i, uni in enumerate(unis):
##            uni.orbital[units] = uni.orbital['energy'] * Energy['Ha', units]
##            window.append([uni.orbital.get_orbital(orb=-15)[units],
##                           uni.orbital.get_orbital()[units]])
##    else:
##        if not isinstance(window, list): window = window * len(unis)
##    if shift or not isinstance(shift, list):
##def photoelectron_spectrum(ax, unis, window=[-10, 0], broaden=0.6,
##                           shift=0, label='', color=None, stepe=1, units='eV',
##                           loc='upper left', fontsize=26, peaks=True,
##                           xlim=None, ylim=None):
##    color = ['r', 'g', 'b', 'c', 'y', 'm', 'k'] if color is None else color
##    arrowprops = {'arrowstyle': '->', 'connectionstyle': 'arc3'}
##    arrowargs = {'xycoords': 'data', 'textcoords': 'data',
##                 'arrowprops': arrowprops, 'fontsize': fontsize}
##    unis = [unis] if not isinstance(unis, list) else unis
##    xmin, xmax = [], []
##    if (len(window) != len(unis) or len(unis) == 2): window = window * len(unis)
##    if not isinstance(shift, list): shift = [shift] * len(unis)
##    if not isinstance(label, list): label = [label] * len(unis)
##    for i, uni in enumerate(unis):
##        height = len(unis) - 1 - i
##        lo, hi = window[i]
##        pes = uni.orbital.convolve(ewin=[lo,hi], broaden=broaden, units=units)[::-1]
##        pes[units] = -pes[units]
##        pes['shifted'] = pes[units] + shift[i]
##        heightened = pes['signal'] + height
##        lab = uni.name if uni.name and not label[i] else label[i]
##        ax.axhline(height, color='k', linewidth=1.2)
##        ax.plot(pes['shifted'], heightened, label=lab, color=color[i % len(color)])
##        o = uni.orbital
##        o = o[(o[units] > lo) & (o[units] < hi) & (o['occupation'] > 0)].drop_duplicates(
##            units).copy().drop_duplicates('vector').sort_values(
##            by=units, ascending=False).reset_index()
##        o[units] = -o[units]
##        leno = len(o)
##        switch = leno // 2
##        nonzero = pes[pes['signal'] > 0.1]['shifted']
##        small = nonzero.min()
##        esmall = small - stepe * switch
##        elarge = nonzero.max()
##        xmin.append(esmall)
##        dh = 1 / (switch + 3)
##        hlo = height + dh
##        hhi = height + (switch + switch % 2) * dh
##        for t in range(-20, 20):
##            ax.plot([t] * 2, [height, height - 0.05], color='k', linewidth=1)
##        if peaks:
##            for c, (sym, en) in enumerate(zip(o['symmetry'], o[units] + shift[i])):
##                ax.plot([en] * 2, [height, height + 0.05], color='k', linewidth=1)
##                astr = r'$' + sym[0].lower() + '_{' + sym[1:].lower() + '}$'
##                e = esmall if c < switch else elarge
##                h = hlo if c < switch else hhi
##                ax.annotate(astr, xy=(en, height + 0.05), xytext=(e, h), **arrowargs)
##                if c < switch:
##                    esmall += stepe
##                    hlo += dh
##                else:
##                    elarge += stepe * 1.5
##                    hhi -= dh
##            xmax.append(elarge)
##    xax = 'E* (' + units + ')' if any((i for i in shift)) else 'E (' + units + ')'
##    xlim = (min(xmin), max(xmax)) if xlim is None else xlim
##    ylim = (0, len(unis)) if ylim is None else ylim
##    ax.set_xlim(xlim)
##    ax.set_ylim(ylim)
##    ax.set_xlabel(xax)
##    ax.legend(loc=loc)
##    return ax
#def new_pes(*unis, filters=None, broaden=0.06, color=None, stepe=0.5, units='eV',
#            fontsize=20, peaklabels=True, xlim=None, extra=None,
#            figsize=(10,10), title=None):
#    """
#    Things
#    """
#    def plot_details(ax, dos, xmin, xmax, peaklabels):
#        switch = len(o) // 2
#        nonzero = dos[dos['signal'] > 0.1]['shifted']
#        small = nonzero.min()
#        esmall = small - stepe * switch
#        elarge = nonzero.max()
#        xmin.append(esmall - 0.5)
#        xmax.append(elarge + 0.5)
#        dh = 1 / (switch + 3)
#        hlo = dh
#        hhi = (switch + switch % 2) * dh
#        for c, (sym, en) in enumerate(zip(o['symmetry'], o['shifted'])):
#            ax.plot([en] * 2, [0, 0.05], color='k', linewidth=1)
#            if peaklabels:
#                if '$' in sym: astr = sym
#                else: astr = r'$\textrm{' + sym[0].lower() + '}_{\\large \\textrm{' + sym[1:].lower() + '}}$'
#                e = esmall if c < switch else elarge
#                h = hlo if c < switch else hhi
#                ax.text(e, h, astr, fontsize=fontsize - 4)
#                if c < switch:
#                    esmall += stepe
#                    hlo += dh
#                else:
#                    elarge += stepe * 1.5
#                    hhi -= dh
#                xmax[-1] = elarge
#        return ax, xmin, xmax
#
#    def plot_extra(ax, extra):
#        for i, stargs in enumerate(zip(extra['x'], extra['y'])):
#            kwargs = {'color': extra['color']}
#            if isinstance(extra['label'], list):
#                kwargs['color'] = extra['color'][i]
#                kwargs['label'] = extra['label'][i]
#            else:
#                if not i: kwargs['label'] = extra['label']
#            ax.plot(*stargs, **kwargs)
#            ax.legend(frameon=False)
#        return ax
#
#    nuni = len(unis)
#    if filters is None:
#        print("filters allows for customization of the plot")
#        filters = [{'eV': [-10, 0]}] * nuni
#    elif isinstance(filters, dict):
#        filters = [filters] * nuni
#    elif len(filters) == 1 and isinstance(filters, list):
#        filters = filters * nuni
#    elif len(filters) != nuni:
#        raise Exception("Provide a list of filter dicts same as number of unis.")
#    nax = nuni + 1 if extra is not None else nuni
#    figargs = {'figsize': figsize}
#    fig = _gen_figure(nxplot=nax, nyplot=1, joinx=True, figargs=figargs)
#    axs = fig.get_axes()
#    color = sns.color_palette('cubehelix', nuni) if color is None else color
#    xmin, xmax = [], []
#    hdls, lbls = [], []
#    for i, (uni, ax, fil) in enumerate(zip(unis, axs, filters)):
#        if 'energy' in fil: lo, hi = fil['energy']
#        elif units in fil: lo, hi = fil[units]
#        else: raise Exception('filters must include an energetic keyword')
#        shift = fil['shift'] if 'shift' in fil else 0
#        lframe = uni.orbital['group'].astype(int).max()
#        dos = uni.orbital.convolve(ewin=[lo,hi], broaden=broaden,
#                                   units=units, frame=lframe)
#        dos['shifted'] = dos[units] + shift
#        lab = uni.name if uni.name is not None \
#              else fil['label'] if 'label' in fil else ''
#        dos[dos['signal'] > 0.01].plot(ax=ax, x='shifted', y='signal',
#                                    label=lab, color=color[i % len(color)])
#        li = uni.orbital['group'].astype(int).max()
#        o = uni.orbital[uni.orbital['group'] == li]
#        o = o[(o[units] > lo) & (o[units] < hi) & (o['occupation'] > 0)]
#        o = o.drop_duplicates(units).copy().drop_duplicates(
#                units).sort_values(by=units).reset_index()
#        o['shifted'] = o[units] + shift
#        ax, xmin, xmax = plot_details(ax, dos, xmin, xmax, peaklabels)
#    if extra:
#        axs[-1] = plot_extra(axs[-1], extra)
#    xlim = (min(xmin), max(xmax)) if xlim is None else xlim
#    if title is not None:
#        axs[0].set_title(title)
#    for i in range(nax):
#        if not (i == nax - 1):
#            sns.despine(bottom=True, trim=True)
#            axs[i].set_xticks([])
#            axs[i].set_xlabel('')
#        axs[i].legend(frameon=False)
#        axs[i].set_xlim(xlim)
#        axs[i].set_yticks([])
#        axs[i].set_yticklabels([])
#    shifted = any(('shift' in fil for fil in filters))
#    xax = 'E* (' + units + ')' if shifted else 'E (' + units + ')'
#    axs[-1].set_xlabel(xax)
#    nx = 2 if abs(xlim[1] - xlim[0]) > 8 else 1
#    axs[-1].set_xticks(np.arange(xlim[0], xlim[1] + 1, nx, dtype=int))
#    return fig
#
## Example filter for the following mo_diagram function
## applied to orbital table
##
##mofilters[key] = [{'eV': [-7, 5],
##                   'occupation': 2,
##                   'symmetry': 'EU'}.copy() for i in range(5)]
##mofilters[key][0]['shift'] = 24.7
##mofilters[key][0]['eV'] = [-30, -10]
##mofilters[key][0]['symmetry'] = '$\pi_{u}$'
##mofilters[key][-1]['eV'] = [0, 10]
##mofilters[key][-1]['shift'] = -11.5
#
#def new_mo_diagram(*unis, filters=None, units='eV', width=0.0625,
#                   pad_degen=0.125, pad_occ=0.03125, scale_occ=1,
#                   fontsize=22, figsize=(10,8), labelpos='right',
#                   ylim=None):
#    """
#    Args
#        unis(exatomic.container.Universe): uni or list of unis
#        filters(dict): dict or list of dicts for each uni
#            accepted kwargs: 'shift', uni.orbital column names
#            special kwargs: 'shift' shifts energies,
#                ['energy', 'eV', units] must be of the form [min, max]
#            Note: values can be strings defining relationships like
#                  {'occupation': '> 0'}
#        units (str): the units in which to display the MO diagram
#        width (float): the width of the line of each energy level
#        pad_degen (float): the spacing between degenerate energy levels
#        pad_occ (float): the spacing between arrows of occupied levels
#        scale_occ (float): scales the size of the occupied arrows
#        fontsize (int): font size for text on the MO diagram
#        figsize (tuple): matplotlib's figure figsize kwarg
#        labelpos (str): ['right', 'bottom'] placement of symmetry labels
#
#    Returns
#        fig (matplotlib.figure.Figure): the plot
#    """
#    def filter_orbs(o, fil):
#        shift = fil['shift'] if 'shift' in fil else 0
#        for key, val in fil.items():
#            if key == 'shift': continue
#            if isinstance(val, str) and \
#            any((i in ['<', '>'] for i in val)):
#                o = eval('o[o["' + key + '"] ' + val + ']')
#                continue
#            val = [val] if not isinstance(val,
#                        (list,tuple)) else val
#            if key in [units, 'energy']:
#                if len(val) != 2:
#                    raise Exception('energy (units) '
#                    'filter arguments must be [min, max]')
#                o = o[(o[key] > val[0]) & (o[key] < val[1])].copy()
#            elif key == 'index':
#                o = o.ix[val].copy()
#            else:
#                o = o[o[key].isin(val)].copy()
#        return o, shift
#
#    def cull_data(o, shift):
#        data = OrderedDict()
#        # Deduplicate manually to count degeneracy
#        for en, sym, occ in zip(o[units], o['symmetry'], o['occupation']):
#            en += shift
#            if '$' in sym: pass
#            else: sym = '${}_{{{}}}$'.format(sym[0].lower(),
#                                             sym[1:].lower())
#            data.setdefault(en, {'degen': 0, 'sym': sym, 'occ': occ})
#            data[en]['degen'] += 1
#        return data
#
#    def offset(degen, pad_degen=pad_degen):
#        start = 0.5 - pad_degen * (degen - 1)
#        return [start + i * 2 * pad_degen for i in range(degen)]
#
#    def occoffset(occ, pad_occ=pad_occ):
#        if not occ: return []
#        if occ <= 1: return [0]
#        if occ <= 2: return [-pad_occ, pad_occ]
#
#    def plot_axis(ax, data):
#        for nrg, vals in data.items():
#            # Format the occupation//symmetry
#            occ = np.round(vals['occ']).astype(int)
#            # Iterate over degeneracy
#            offs = offset(vals['degen'])
#            for x in offs:
#                ax.plot([x - lw, x + lw], [nrg, nrg],
#                        color='k', lw=1.2)
#                # Iterate over occupation
#                for s, ocof in enumerate(occoffset(occ)):
#                    # Down arrow if beta spin else up arrow
#                    pt = -2 * lw * scale_occ if s == 1 else 2 * lw * scale_occ
#                    st = nrg + lw * scale_occ if s == 1 else nrg - lw * scale_occ
#                    ax.arrow(ocof + x, st, 0, pt, **arrows)
#        # Assign symmetry label
#            sym = vals['sym']
#            if labelpos == 'right':
#                ax.text(x + 2 * lw, nrg - lw, sym, fontsize=fontsize - 2)
#            elif labelpos == 'bottom':
#                ax.text(0.5 - 2 * lw, nrg - 4 * lw, sym, fontsize=fontsize - 2)
#        return ax
#
#    if filters is None:
#        print('filters allows for customization of the plot.')
#        filters = {'eV': [-5,5]}
#    nunis = len(unis)
#    filters = [filters] * nunis if isinstance(filters, dict) else filters
#    # Make our figure and axes
#    figargs = {'figsize': figsize}
#    fig = _gen_figure(nxplot=nunis, nyplot=1, joinx=True, sharex=True, figargs=figargs)
#    axs = fig.get_axes()
#    # Some initialization
#    ymin = np.empty(nunis, dtype=np.float64)
#    ymax = np.empty(nunis, dtype=np.float64)
#    ysc = exatomic.Energy['eV', units]
#    lw = width
#    arrows = {'fc': "k", 'ec': "k",
#              'head_width': 0.01,
#              'head_length': 0.05 * ysc}
#    for i, (ax, uni, fil) in enumerate(zip(axs, unis, filters)):
#        if uni.name: ax.set_title(uni.name)
#        o = uni.orbital
#        o[units] = o['energy'] * exatomic.Energy['Ha', units]
#        o, shift = filter_orbs(o, fil)
#        print('Filtered {} eigenvalues from '
#              '{}'.format(o.shape[0], uni.name))
#        ymin[i] = o[units].min() + shift
#        ymax[i] = o[units].max() + shift
#        data = cull_data(o, shift)
#        ax = plot_axis(ax, data)
#    # Go back through axes to set limits
#    for i, ax in enumerate(axs):
#        ax.set_xlim((0,1))
#        ax.xaxis.set_ticklabels([])
#        ylims = (min(ymin[~np.isnan(ymin)]) - 1, max(ymax[~np.isnan(ymax)]) + 1) \
#                if ylim is None else ylim
#        ax.set_ylim(ylims)
#        if not i:
#            ax.set_ylabel('E ({})'.format(units), fontsize=fontsize)
#            diff = ylims[1] - ylims[0]
#            headlen = 0.05 * diff
#            ax.arrow(0.05, ylims[0], 0, diff - headlen, fc="k", ec="k",
#                     head_width=0.05, head_length= headlen)
#    sns.despine(left=True, bottom=True, right=True)
#    return fig
#
##    unis = [unis] if not isinstance(unis, list) else unis
##    if window is None:
##        window = []
##        for i, uni in enumerate(unis):
##            uni.orbital[units] = uni.orbital['energy'] * Energy['Ha', units]
##            window.append([uni.orbital.get_orbital(orb=-15)[units],
##                           uni.orbital.get_orbital()[units]])
##    else:
##        if not isinstance(window, list): window = window * len(unis)
##    if shift or not isinstance(shift, list):
##def photoelectron_spectrum(ax, unis, window=[-10, 0], broaden=0.6,
##                           shift=0, label='', color=None, stepe=1, units='eV',
##                           loc='upper left', fontsize=26, peaks=True,
##                           xlim=None, ylim=None):
##    color = ['r', 'g', 'b', 'c', 'y', 'm', 'k'] if color is None else color
##    arrowprops = {'arrowstyle': '->', 'connectionstyle': 'arc3'}
##    arrowargs = {'xycoords': 'data', 'textcoords': 'data',
##                 'arrowprops': arrowprops, 'fontsize': fontsize}
##    unis = [unis] if not isinstance(unis, list) else unis
##    xmin, xmax = [], []
##    if (len(window) != len(unis) or len(unis) == 2): window = window * len(unis)
##    if not isinstance(shift, list): shift = [shift] * len(unis)
##    if not isinstance(label, list): label = [label] * len(unis)
##    for i, uni in enumerate(unis):
##        height = len(unis) - 1 - i
##        lo, hi = window[i]
##        pes = uni.orbital.convolve(ewin=[lo,hi], broaden=broaden, units=units)[::-1]
##        pes[units] = -pes[units]
##        pes['shifted'] = pes[units] + shift[i]
##        heightened = pes['signal'] + height
##        lab = uni.name if uni.name and not label[i] else label[i]
##        ax.axhline(height, color='k', linewidth=1.2)
##        ax.plot(pes['shifted'], heightened, label=lab, color=color[i % len(color)])
##        o = uni.orbital
##        o = o[(o[units] > lo) & (o[units] < hi) & (o['occupation'] > 0)].drop_duplicates(
##            units).copy().drop_duplicates('vector').sort_values(
##            by=units, ascending=False).reset_index()
##        o[units] = -o[units]
##        leno = len(o)
##        switch = leno // 2
##        nonzero = pes[pes['signal'] > 0.1]['shifted']
##        small = nonzero.min()
##        esmall = small - stepe * switch
##        elarge = nonzero.max()
##        xmin.append(esmall)
##        dh = 1 / (switch + 3)
##        hlo = height + dh
##        hhi = height + (switch + switch % 2) * dh
##        for t in range(-20, 20):
##            ax.plot([t] * 2, [height, height - 0.05], color='k', linewidth=1)
##        if peaks:
##            for c, (sym, en) in enumerate(zip(o['symmetry'], o[units] + shift[i])):
##                ax.plot([en] * 2, [height, height + 0.05], color='k', linewidth=1)
##                astr = r'$' + sym[0].lower() + '_{' + sym[1:].lower() + '}$'
##                e = esmall if c < switch else elarge
##                h = hlo if c < switch else hhi
##                ax.annotate(astr, xy=(en, height + 0.05), xytext=(e, h), **arrowargs)
##                if c < switch:
##                    esmall += stepe
##                    hlo += dh
##                else:
##                    elarge += stepe * 1.5
##                    hhi -= dh
##            xmax.append(elarge)
##    xax = 'E* (' + units + ')' if any((i for i in shift)) else 'E (' + units + ')'
##    xlim = (min(xmin), max(xmax)) if xlim is None else xlim
##    ylim = (0, len(unis)) if ylim is None else ylim
##    ax.set_xlim(xlim)
##    ax.set_ylim(ylim)
##    ax.set_xlabel(xax)
##    ax.legend(loc=loc)
##    return ax
