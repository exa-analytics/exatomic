# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Wrappers for Pseudopotential Parsing
#####################################
This module provides the
"""
import six
from exa.core import Container
from exa.special import Typed


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
    def __init__(self, arg, symbol):
        names = ("data", "log", "pstest", "psed", "aeed")
        kwargs = dict(zip(names, parse_paw_psp(arg, symbol)))
        super(PSPData, self).__init__(**kwargs)





#def parse_psp_data(scratch):
#    """
#    Given an NWChem scratch directory parse all pseudopotential
#    information.
#    """
#    plts = glob(os.path.join(scratch, "*.dat"))
#    symbols = {}
#    for plt in plts:
#        base = os.path.basename(plt)
#        first = base.split(".")[0]
#        paw = False
#        if "_" in first:
#            paw = True
#            first = first.split("_")[0]
#        symbols[first] = paw
#    for symbol, paw in symbols.items():
#        if paw:
#            return parse_paw_psp(scratch, symbol)
#        else:
#            return parse_nc_psp(scratch, symbol)
#import six
#from exa.tex import text_value_cleaner
#from exa.special import LazyFunction
#from exa.core import Meta, Parser, DataFrame
#from exatomic.nwchem.nwpw.pseudopotentials.ps import PAWOutput
#
#
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
#
#
#def parse_nc_psp(paths):
#    """
#    """
#    pass
#
#
