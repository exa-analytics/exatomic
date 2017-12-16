#
#def plot_wfc(df, nl, xlim=(0, 5), **kwargs):
#    nl = nl.upper()
#    cols = [col for col in df.columns if nl in col and "psi" in col[0]]
#    ax = df[cols].plot(xlim=xlim, **kwargs)
#    patches, labels = ax.get_legend_handles_labels()
#    labels = [i.replace(r"_i", "_{" + j + "}") for i, j in cols]
#    legend = ax.legend(patches, labels)
#    ax.set_ylabel("Wave Functions (arb.)")
#    ax.set_xlabel(r"$r\ (\text{\AA})$")
#
#
#def plot_phi_p(df, nl, xlim=(0, 5), **kwargs):
#    nl = nl.upper()
#    cols = [col for col in df.columns if nl in col and ("phi" in col[0] or "tilde" in col[0])]
#    ax = df[cols].plot(xlim=xlim, **kwargs)
#    patches, labels = ax.get_legend_handles_labels()
#    labels = [i.replace(r"_i", "_{" + j + "}") for i, j in cols]
#    legend = ax.legend(patches, labels)
#    ax.set_ylabel("Partial Waves (arb.)")
#    ax.set_xlabel(r"$r\ (\text{\AA})$")
#
#
#def plot_pot(df, xlim=(0, 5), **kwargs):
#    cols = [r"$v(r)$", r"$\tilde{v}(r)$", r"$\hat{V}_{PS}$"]
#    ax = df[cols].plot(xlim=xlim, secondary_y=r"$v(r)$", **kwargs)
#    ax.set_ylabel("Potential (a.u.)")
#    ax.set_xlabel(r"$r\ (\text{\AA})$")
#
#
#def plot_paw_pot(df, nl, xlim=(0, 5), **kwargs):
#    if isinstance(nl, str):
#        nl = [nl]
#    for i in range(len(nl)):
#        nl[i] = nl[i].upper()
#    cols = list(set([col for col in df.columns if any(nl_ in col for nl_ in nl) and "V" in col]))
#    ax = df[cols].plot(xlim=xlim, **kwargs)
#    patches, labels = ax.get_legend_handles_labels()
#    #labels = [i.replace(r"\alpha", "{" + j + "}") for i, j in cols]
#    legend = ax.legend(patches, labels)
#    ax.set_ylabel("Potential (a.u.)")
#    ax.set_xlabel(r"$r\ (\text{\AA})$")
#
#
#def plot_log(df, l=["S", "P", "D", "F"], ylim=(-5, 5), **kwargs):
#    if isinstance(l, str):
#        l = [l]
#    for i in range(len(l)):
#        l[i] = l[i].upper()
#    cols = [col for col in df.columns if any("_{" + l_ + "}" in col for l_ in l)]
#    style = ["-", "--"]*(len(cols)//2)
#    ax = df[cols].plot(ylim=ylim, style=style, **kwargs)
#    ax.set_ylabel("$D(E)$")
#    ax.set_xlabel(r"$E\ (a.u.)$")
#    return ax
