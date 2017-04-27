
class PAWPSPMeta(exa.core.editor.SectionsMeta):
    """Defines data objects for PAWOutput."""
    fields = pd.DataFrame
    data = pd.DataFrame
    info = dict
    grid = LazyEval
    _descriptions = {'data': "Pseudized channel data",
                     'info': "Atom, charge, and core info",
                     'grid': "Grid information",
                     'fields': "Stuff"}

class PAWPSP(six.with_metaclass(PAWPSPMeta, exa.Parser)):
    """
    """
    _key_info_names = ["psp_type", "atom_name", "valence_z", "rmin",
                       "rmax", "nr", "nbasis", "rcuts", "max_i_r",
                       "comment", "core_kinetic_energy"]
    def _parse(self):
        info = {}
        for i, name in enumerate(self._key_info_names):
            info[name] = text_value_cleaner(str(self[i]))
        self.info = info
        n = self.info['nbasis']
        nr = self.info['nr']
        self.grid = LazyEval(scaled_logspace, self.info['rmin'], 1.005, nr-1)
        i += 1
        start = i + self.info['nbasis']
        self.data = pd.read_csv(self[i:start].to_stream(), delim_whitespace=True,
                                names=('n', 'eigenvalue', 'nps', 'l'))
        self.data['rcut'] = self.info['rcuts'].split()
        self.data['rcut'] = self.data['rcut'].astype(np.float64)
        nls = (self.data['n'].astype(str) + self.data['l'].map(l0)).tolist()
        fields = pd.DataFrame()
        for name in [r"$\psi_nl(r)$", r"$\psi^{'}_nl(r)$", r"$\tilde{\psi}_nl(r)$",
                        r"$\tilde{\psi}^{'}_nl(r)$", r"$\tilde{p}_nl(r)$"]:
            for nl in nls:
                end = start + nr
                wave = pd.read_csv(self[start:end].to_stream(), delim_whitespace=True)
                wave_name = name.replace('nl', "{" + nl + "}")
                fields[wave_name] = wave
                start += nr
        for name in [r"$\rho_{\text{core}}(r)/(4\pi)$",
                     r"$\rho_{\text{core, ps}}(r)/(4\pi)$",
                     r"$V_{ps}(r)$"]:
            end = start + nr
            rho = pd.read_csv(self[start:end].to_stream(), delim_whitespace=True)
            fields[name] = rho
            start += nr
        self.info[r"$\sigma_{comp}$"] = text_value_cleaner(str(self[start]))
        start += 1
        self.info["Z"] = text_value_cleaner(str(self[start]))
        start += 1
        for nl in nls:
            end = start + nr
            wave = pd.read_csv(self[start:end].to_stream(), delim_whitespace=True)
            wave_name = r"$\tilde{p}_{" + nl + ",0}(r)$"
            fields[wave_name] = wave
            start += nr
        fields.index = self.grid()
        self.fields = fields
