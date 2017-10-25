from exa import Parser, Typed, DataFrame
import exatomic.base
from exatomic.core import Atom, GaussianBasisSet, Coefficient
from exatomic.core.orbital import _gen_mo_index
from exatomic.core.basis import str2l
import re

# Auxiliary section parsers
class Header(Parser):
    """
    Parser for the header of an NWChem calculation.

    Constructs a dictionary of 'metadata' related to calculation
    being performed such as code version, time and date, and processor count.
    Additionally, the input deck is occasionally printed here.
    """
    _ek = "   NWChem Input Module"
    _key0 = " = "
    info = Typed(dict, doc="Metadata about the calculation as a dict")

    def _parse_both(self):
        """Only parse the end; the start is the beginning of the file."""
        end = self.find_next(self._ek, cursor=0)[0]
        return [(0, self.lines[0])], [(end, self.lines[end])]

    def parse_meta(self):
        """
        """
        meta = {}
        for line in self:
            cnt = line.count(self._key0)
            if cnt > 0:
                split = line.split(self._key0)
                key = split[0].strip()
                value = split[-1].strip()
                meta[key] = value
        self.info = meta


        class Geometry(Parser):
    """
    Parser for the 'Geometry' section of NWChem output files.
    """
    _start = re.compile("^\s*Geometry \".*\" -> \".*\"")
    _end = re.compile("^\s*Atomic Mass")
    _int0 = 7
    _int1 = -2
    _cols = ("number", "tag", "charge", "x", "y", "z")
    _wids = (6, 17, 10, 15, 15, 15)
    atom = Typed(Atom, doc="Table of nuclear coordinates.")

    def parse_atom(self):
        """
        Create an :class:`~exatomic.core.atom.Atom` object.
        """
        atom = pd.read_fwf(self[self._int0:self._int1].to_stream(),
                           widths=self._wids, names=self._cols)
        atom[self._cols[0]] = atom[self._cols[0]].astype(int)
        atom['symbol'] = atom['tag'].apply(lambda x: "".join([a for a in x if a.isalpha()]))
        self.atom = Atom.from_xyz(atom)


        class BasisSet(Parser):
    """
    Parser for the NWChem's printing of basis sets.
    """
    _start = re.compile("^\s*Basis \"")
    _ek = re.compile("^ Summary of \"")
    _k = "spherical"
    _k0 = "Exponent"
    _k1 = "("
    _i0 = 2
    _i1 = 4
    _cols = ("function", "l", "a", "d", "tag")
    basis_set = Typed(GaussianBasisSet, doc="Gaussian basis set description.")

    def _parse_end(self, starts):
        return [self.regex_next(self._ek, cursor=i[0]) for i in starts]

    def parse_basis_set(self):
        """
        """
        data = []
        tag = None
        spherical = True if self._k in self.lines[0] else False
        for line in self:
            if self._k0 not in line:
                ls = line.split()
                if len(ls) == self._i0 and self._k1 in line:
                    tag = ls[0]
                elif len(ls) == self._i1:
                    data.append(ls+[tag])
        basis_set = DataFrame(data, columns=self._cols)
        basis_set['l'] = basis_set['l'].str.lower().map(str2l)
        self.basis_set = GaussianBasisSet(basis_set, spherical=spherical, order=lambda x: x)



        class MOVectors(Parser):
    """
    Parser for NWChem's molecular orbital coefficient matrix output.
    """
    _start = "Final MO vectors"
    _end = "center of mass"
    _i0 = 6
    _i1 = -1
    _cols = (0, 1, 2, 3, 4, 5, 6)
    _wids = (6, 12, 12, 12, 12, 12, 12)
    coefficient = Typed(Coefficient, doc="Molecular orbital coefficients.")

    def parse_coefficient(self):
        """
        Parse molecular orbital coefficient matrix.
        """
        # Read in the mangled table
        c = pd.read_fwf(self[self._i0:self._i1].to_stream(),
                        names=self._cols, widths=self._wids)
        # Remove null lines
        idx = c[c[0].isnull()].index.values
        c = c[~c.index.isin(idx)]
        # The size of the basis is given by the informational numbers
        nbas = c[0].unique().shape[0]
        # Remove that column; it doesn't contain coefficients only sequential integers
        del c[0]
        n = c.shape[0]//nbas
        coefs = []
        # The loop below is like numpy.array_split (same speed, but
        # fully compatible with pandas)
        for i in range(n):
            coefs.append(c.iloc[i*nbas:(i+1)*nbas, :].astype(float).dropna(axis=1).values.ravel("F"))
        # Concatenate coefficients
        c = np.concatenate(coefs)
        orbital, chi = _gen_mo_index(nbas)
        df = pd.DataFrame.from_dict({'orbital': orbital, 'chi': chi, 'c': c})
        df['frame'] = 0
        self.coefficient = df


        # nwchem.scf.output.Output
class Output(Parser):
    """
    Base parser for NWChem output files

    In some cases the parser cannot correctly assign frame indexes for the
    atom table. The **frame_map** attributed can be used to fix this problem.

    Attributes:
        frame_map (iterable): Correctly ordered frame indexes
    """
    atom = Typed(Atom, doc="Full atom table from all 'frames'.")
    basis_set = Typed(GaussianBasisSet, doc="Gaussian basis set description.")
    coefficient = Typed(Coefficient, doc="Full molecular orbital coefficient table.")

    def parse_atom(self):
        """
        Generate the complete :class:`~exatomic.core.atom.Atom` table
        for the entire calculation.

        Warning:
            The frame index is arbitrary! Map it to the correct
            values as required!
        """
        atoms = []
        for i, sec in enumerate(self.get_sections(Geometry)):
            atom = sec.atom
            fdx = i if self.frame_map is None else self.frame_map[i]
            atom['frame'] = fdx
            atoms.append(atom)
        self.atom = pd.concat(atoms, ignore_index=True)

    def parse_basis_set(self):
        """
        """
        key = self.sections[self.sections['parser'] == BasisSet].index[0]
        self.basis_set = self.get_section(key).basis_set

    def parse_coefficient(self):
        """
        Complete :class:`~exatomic.core.orbital.Coefficient` table.

        Warning:
            The frame index is arbitrary! Map it to the correct
            values as required!
        """
        coefs = []
        frames = sorted(self.atom['frame'].unique())
        for i, sec in enumerate(self.get_sections(MOVectors)):
            c = sec.coefficient
            fdx = frames[i]
            c['frame'] = fdx
            coefs.append(c)
        self.coefficient = pd.concat(coefs, ignore_index=True)

    def __init__(self, *args, **kwargs):
        frame_map = kwargs.pop("frame_map", None)
        super(Output, self).__init__(*args, **kwargs)
        self.frame_map = frame_map

Output.add_parsers(Header, Geometry, BasisSet, MOVectors)
