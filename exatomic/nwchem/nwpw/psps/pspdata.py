# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
NWChem PSP Data
#####################################
This module provides a container object that stores output data coming from
NWChem's pseudopotential generation scheme. Note that parsable data is created
only if debug printing is used. This module also provides functions that parse
multiple files simultaneously.
"""
import six
import pandas as pd
import os
from glob import glob
from exa.core import Container
from exa.special import Typed
from exa.mpl import qualitative, sns
from .paw_ae import AEOutput
from .paw_ps import PAWOutput


def parse_psp_data(scratch):
    """
    Given an NWChem scratch directory parse all pseudopotential
    information.
    """
    plts = glob(os.path.join(scratch, "*.dat"))
    symbols = {}
    for plt in plts:
        base = os.path.basename(plt)
        first = base.split(".")[0]
        paw = False
        if "_" in first:
            paw = True
            first = first.split("_")[0]
        symbols[first] = paw
    for symbol, paw in symbols.items():
        if paw:
            return parse_paw_psp(scratch, symbol)
        else:
            raise
            #return parse_nc_psp(scratch, symbol)


def parse_paw_psp(scratch, symbol):
    """
    """
    # Parse AE and PS output data to get nl values
    aeed = AEOutput(os.path.join(scratch, symbol + "_out"))
    psed = PAWOutput(os.path.join(scratch, symbol + "_paw"))
    aenl = (aeed.data['n'].astype(str) + aeed.data['l'].astype(str)).tolist()
    psnl = psed.data['nl'].tolist()
    r = [r"$r$"]
    # Parse AE orbital data
    aeorb = pd.read_csv(os.path.join(scratch, symbol+"_orb.dat"),
                        delim_whitespace=True, names=r+aenl)
    del aeorb[aeorb.columns[0]]
    aeorb.columns = [[r"$\psi_i(r)$"]*len(aenl), aenl]
    # Parse PS orbital data
    ps3nl = [nl for nl in psnl for i in range(3)]
    psorb = pd.read_csv(os.path.join(scratch, symbol + "_paw.dat"),
                        delim_whitespace=True, names=r+ps3nl)
    del psorb[psorb.columns[0]]
    psorb.columns = [[r"$\phi_i(r)$", r"$\tilde{\phi}_i(r)$",
                      r"$\tilde{p}_i(r)$"]*len(psnl), ps3nl]
    # Parse PS potential data
    potnames = [r"$v(r)$", r"$\tilde{v}(r)$", r"$\hat{V}_{PS}$"]
    potnames += [r"$V_{" + nl + "}$" for nl in psnl]
    pspot = pd.read_csv(os.path.join(scratch, symbol + "_pot.dat"),
                        delim_whitespace=True, names=r+potnames)
    del pspot[pspot.columns[0]]
    # Parse PS wfc data
    ps2nl = [nl for nl in psnl for i in range(2)]
    pstest = pd.read_csv(os.path.join(scratch, symbol + "_test.dat"),
                         delim_whitespace=True, names=r+ps2nl)
    del pstest[pstest.columns[0]]
    pstest.columns = [[r"phi-ps0", r"psi-ps"]*len(psnl), ps2nl]
    # Parse logrithmic derivatives tests
    log = []
    for path in glob(os.path.join(scratch, "*_scat_test.dat")):
        angmom = path.split(os.sep)[-1].split("_")[1].upper()
        ae_ = r"$D^{AE}_{" + angmom + "}(E)$"
        ps_ = r"$D^{PS}_{" + angmom + "}(E)$"
        df = pd.read_csv(os.path.join(path), delim_whitespace=True,
                         names=["$E\ (au)$", ae_, ps_])
        if len(log) > 0:
            del df["$E\ (au)$"]
        log.append(df)
    if len(log) > 0:
        log = pd.concat(log, axis=1).set_index("$E\ (au)$")
    data = pd.concat((aeorb, psorb, pspot), axis=1)
    data.index = psed.grid()
    pstest.index = psed.grid()
    return data, log, pstest, psed, aeed


class PSPMeta(Typed):
    """
    Defines the data objects associated with the container
    :class:`~exatomic.nwchem.nwpw.psps.base.NWChemPSPs`.
    """
    pass


class PSPData(six.with_metaclass(PSPMeta, Container)):
    """
    A container for storing discrete pseudopotentials and pseudo-waves
    defined on a radial grid and used in plane wave calculations.

    Note:
        This container stores pseudopotential data for all atoms in the
        calculation of interest.
    """
    def plot_log(self, **kwargs):
        """Plot the logarithmic derivatives."""
        n = len(self.log.columns)//2
        # Get default values
        style = kwargs.pop("style", ["-", "--"])
        ylim = kwargs.pop("ylim", (-5, 5))
        colors = kwargs.pop("colors", None)
        if colors is None:
            colors = qualitative(n)
        elif not isinstance(colors, (list, tuple)):
            colors = [colors]*n
        # Plot the figure
        ax = sns.plt.subplot()
        for i, col in enumerate(self.log.columns[::2]):
            cols = [col, col.replace("AE", "PS")]
            ax = self.log[cols].plot(ax=ax, style=style, c=colors[i], **kwargs)
        ax.set_ylim(*ylim)
        return ax

    def plot_psae(self):
        """Plot AE and PS waves for comparison."""
        raise
        #nls = self.psed.data['nl'].tolist()

    def plot_ps(self, nl, **kwargs):
        """Plot a given pseudo wave, projector, and AE reference."""
        cols = [col for col in self.data.columns if nl.upper() in col]
        vcol = [col for col in cols if "V" in col][0]
        xlim = kwargs.pop("xlim", (0, 3.5))
        ax = self.data[cols].plot(secondary_y=vcol, xlim=xlim, **kwargs)
        return ax

    def log_diff(self):
        """Error in logarithmic differences."""
        self._logs = []
        for l in ("S", "P", "D", "F"):
            cols = [col for col in self.log.columns if "{" + l + "}" in col]
            if len(cols) == 2:
                self.log[l] = self.log[cols[0]] - self.log[cols[1]]
                self._logs.append(l)

    def log_diff_estimate(self):
        """Log diff errore."""
        if "S" not in self.log.columns:
            self.log_diff()
        return self.log.loc[self._logs].abs().sum()

    def __init__(self, path):
        data, log, pstest, psed, aeed = parse_psp_data(path)
        super(PSPData, self).__init__(data=data, log=log, pstest=pstest, psed=psed, aeed=aeed)


#
#
#def parse_nc_psp(paths):
#    """
#    """
#    pass
#
#
