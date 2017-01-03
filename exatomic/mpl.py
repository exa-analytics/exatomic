# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Custom Axes
###################
"""

#from exa.mpl import ExAxes

#class PESAx(ExAxes):
#
#    def photoelectron_spectrum(self, unis, window=[-10, 0], broaden=0.6,
#                               shift=0, label='', color=None, stepe=1, units='eV',
#                               loc='upper left', fontsize=26, peaks=True):
#        color = ['r', 'g', 'b', 'c', 'y', 'm', 'k', 0.75] if color is None else color
#        arrowprops = {'arrowstyle': '->', 'connectionstyle': 'arc3'}
#        arrowargs = {'xycoords': 'data', 'textcoords': 'data',
#                     'arrowprops': arrowprops, 'fontsize': fontsize}
#        unis = [unis] if not isinstance(unis, list) else unis
#        xmin = []
#        xmax = []
#        if len(window) != len(unis):
#            window = window * len(unis)
#        if not isinstance(shift, list):
#            shift = [shift] * len(unis)
#        if not isinstance(label, list):
#            label = [label] * len(unis)
#        for i, uni in enumerate(unis):
#            height = len(unis) - 1 - i
#            lo, hi = window[i]
#            pes = uni.orbital.convolve(ewin=[lo,hi], broaden=broaden, units=units)[::-1]
#            pes[units] -= pes[units]
#            pes['shifted'] = pes[units] + shift[i]
#            heightened = pes['signal'] + height
#            lab = uni.name if uni.name and not label[i] else label[i]
#            self.axhline(height, color='k', linewidth=1.2)
#            self.plot(pes['shifted'], heightened, label=lab, color=color[i])
#            o = uni.orbital
#            o = o[(o[units] > lo) & (o[units] < hi)].drop_duplicates(
#                units).copy().sort_values(by=units, ascending=False).reset_index()
#            o[units] = -o[units]
#            leno = len(o)
#            switch = leno // 2
#            nonzero = pes[pes['signal'] > 0.01]['shifted']
#            small = nonzero.min()
#            esmall = small - stepe * switch
#            elarge = nonzero.max()
#            xmin.append(esmall)
#            dh = 1 / (switch + 3)
#            hlo = height + dh
#            hhi = height(switch + switch % 2) * dh
#            for c, (sym, en) in enumerate(zip(o['symmetry'], o[units + shift[i])):
#                self.plot([en] * 2, [height, height + 0.05], color='k', linewidth=1)
#                astr = r'$' + sym[0].lower() + '_{' + sym[1:].lower() + '}$'
#                e = esmall if c < switch else elarge
#                h = hlo if c < switch else hhi
#                self.annotate(astr, xy=(en, height + 0.05), xytext=(e, h), **arrowargs)
#                if c < switch:
#                    esmall += stepe
#                    hlo += dh
#                else:
#                    elarge += stepe * 1.5
#                    hhi -= dh
#            xmax.append(elarge)
#        xax = 'E* (' + units + ')' if any((i for i in shift)) else 'E (' + units + ')'
#        self.set_xlim([min(xmin), max(xmax)])
#        self.set_xlabel(xax)
#        self.legend(loc=loc)
