
from collections import OrderedDict

def compute_nodes(df):
    rnodepos = OrderedDict()
    r = (df.index[1:] + df.index[:-1])/2
    for col in df.columns:
        series = pd.Series(df[col].values[1:]*df[col].values[:-1], index=r)
        rnodepos[col] = series[series < 0].index.values
    return rnodepos


def compute_log_diff(df):
    l = ["S", "P", "D", "F"]
    d = []
    for l_ in l:
        cols = [col for col in df.columns if "_{" + l_ + "}" in col]
        if len(cols) != 2:
            continue
        s = (df[cols[0]] - df[cols[1]]).abs()
        s.name = l_
        d.append(s)
    d = pd.concat(d, axis=1)
    return d
