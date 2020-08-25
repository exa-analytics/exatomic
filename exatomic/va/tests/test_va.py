import numpy as np
from unittest import TestCase
from os import sep, remove, rmdir
from tempfile import mkdtemp
import tarfile
from glob import glob
from exatomic.base import resource
from exatomic.va import VA, get_data, gen_delta
from exatomic.gaussian import Fchk, Output as gOutput
from exatomic.nwchem import Output


TMPDIR = mkdtemp()
h2o2_freq = Fchk(resource('g16-h2o2-def2tzvp-freq.fchk'))
methyloxirane_freq = Fchk(resource('g16-methyloxirane-def2tzvp-freq.fchk'))
tar = tarfile.open(resource('va-vroa-h2o2.tar.bz'), mode='r')
tar.extractall(TMPDIR)
tar.close()
tar = tarfile.open(resource('va-vroa-methyloxirane.tar.bz'), mode='r')
tar.extractall(TMPDIR)
tar.close()
nitro_freq = gOutput(resource('g09-nitromalonamide-6-31++g-freq.out'))
tar = tarfile.open(resource('va-zpvc-nitro_nmr.tar.bz'), mode='r')
tar.extractall(TMPDIR)
tar.close()


class TestGetData(TestCase):
    def test_getter_small(self):
        path = sep.join([TMPDIR, 'h2o2', '*'])
        df = get_data(path=path, attr='roa', soft=Output, f_start='va-roa-h2o2-def2tzvp-514.5-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 130)
        df = get_data(path=path, attr='gradient', soft=Output, f_start='va-roa-h2o2-def2tzvp-514.5-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 52)

    def test_getter_large(self):
        path = sep.join([TMPDIR, 'methyloxirane', '*'])
        df = get_data(path=path, attr='roa', soft=Output, f_start='va-roa-methyloxirane-def2tzvp-488.9-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 160)
        df = get_data(path=path, attr='gradient', soft=Output, f_start='va-roa-methyloxirane-def2tzvp-488.9-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 160)


class TestVROA(TestCase):
    def test_vroa(self):
        h2o2_freq.parse_frequency()
        h2o2_freq.parse_frequency_ext()
        delta = gen_delta(delta_type=2, freq=h2o2_freq.frequency.copy())
        va_corr = VA()
        path = sep.join([TMPDIR, 'h2o2', '*'])
        va_corr.roa = get_data(path=path, attr='roa', soft=Output, f_start='va-roa-h2o2-def2tzvp-514.5-',
                               f_end='.out')
        va_corr.roa['exc_freq'] = np.tile(514.5, len(va_corr.roa))
        va_corr.gradient = get_data(path=path, attr='gradient', soft=Output,
                                    f_start='va-roa-h2o2-def2tzvp-514.5-', f_end='.out')
        va_corr.gradient['exc_freq'] = np.tile(514.5, len(va_corr.gradient))
        va_corr.vroa(uni=h2o2_freq, delta=delta['delta'].values)
        scatter_data = np.array([[ 3.47311779e+02,  0.00000000e+00, -3.27390198e+02,
                                  -8.44921542e+01, -4.22102267e-02, -3.41332079e-02,
                                  -3.91676006e-03,  5.14500000e+02],
                                 [ 8.60534577e+02,  1.00000000e+00,  1.75268228e+02,
                                  -8.09603043e+00, -8.26286589e+00,  1.65666769e-02,
                                  -3.01543530e-03,  5.14500000e+02],
                                 [ 1.24319010e+03,  2.00000000e+00, -3.35605422e+02,
                                  -7.26516978e+01,  1.06370293e-06, -3.45429748e-02,
                                  -4.20725882e-03,  5.14500000e+02],
                                 [ 1.37182002e+03,  3.00000000e+00,  2.20210484e+02,
                                   5.78999940e+01, -4.46257676e+00,  2.29930063e-02,
                                  -6.16087427e-04,  5.14500000e+02],
                                 [ 3.59750268e+03,  4.00000000e+00, -3.50819253e+03,
                                  -3.90826518e+02,  5.90325028e-02, -3.49292932e-01,
                                  -4.98353528e-02,  5.14500000e+02],
                                 [ 3.59821746e+03,  5.00000000e+00,  5.05236006e+03,
                                   4.00023286e+02,  6.60389814e+00,  4.97827311e-01,
                                   7.91921951e-02,  5.14500000e+02]])
        raman_data = np.array([[3.47311779e+02, 0.00000000e+00, 3.42307581e-05, 6.38955258e-01,
                                2.04527298e+01, 5.14500000e+02],
                               [8.60534577e+02, 1.00000000e+00, 1.90190339e-01, 1.07090867e+00,
                                6.85033386e+01, 5.14500000e+02],
                               [1.24319010e+03, 2.00000000e+00, 1.27606294e-08, 3.46909710e-01,
                                1.11011130e+01, 5.14500000e+02],
                               [1.37182002e+03, 3.00000000e+00, 3.94139588e-02, 1.19286864e+00,
                                4.52663091e+01, 5.14500000e+02],
                               [3.59750268e+03, 4.00000000e+00, 1.52822910e-02, 6.21166773e+00,
                                2.01524180e+02, 5.14500000e+02],
                               [3.59821746e+03, 5.00000000e+00, 1.60412161e+00, 5.19841596e+00,
                                4.55091201e+02, 5.14500000e+02]])
        scatter_data = scatter_data.T.copy()
        raman_data = raman_data.T.copy()
        # test all columns of the respective dataframe to get a better sense of what is broken
        self.assertTrue(np.allclose(va_corr.scatter['freq'].values,           scatter_data[0], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['freqdx'].values,         scatter_data[1], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['beta_g*1e6'].values,     scatter_data[2], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['beta_A*1e6'].values,     scatter_data[3], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['alpha_g*1e6'].values,    scatter_data[4], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['backscatter'].values,    scatter_data[5], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['forwardscatter'].values, scatter_data[6], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['exc_freq'].values,       scatter_data[7], rtol=5e-4))

        self.assertTrue(np.allclose(va_corr.raman['freq'].values,          raman_data[0], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.raman['freqdx'].values,        raman_data[1], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.raman['alpha_squared'].values, raman_data[2], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.raman['beta_alpha'].values,    raman_data[3], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.raman['raman_int'].values,     raman_data[4], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.raman['exc_freq'].values,      raman_data[5], rtol=5e-4))

    def test_select_freq(self):
        methyloxirane_freq.parse_frequency()
        methyloxirane_freq.parse_frequency_ext()
        delta = gen_delta(delta_type=2, freq=methyloxirane_freq.frequency.copy())
        va_corr = VA()
        path = sep.join([TMPDIR, 'methyloxirane', '*'])
        va_corr.roa = get_data(path=path, attr='roa', soft=Output,
                               f_start='va-roa-methyloxirane-def2tzvp-488.9-', f_end='.out')
        va_corr.roa['exc_freq'] = np.tile(488.9, len(va_corr.roa))
        va_corr.gradient = get_data(path=path, attr='gradient', soft=Output,
                                    f_start='va-roa-methyloxirane-def2tzvp-488.9-', f_end='.out')
        va_corr.gradient['exc_freq'] = np.tile(488.9, len(va_corr.gradient))
        va_corr.vroa(uni=methyloxirane_freq, delta=delta['delta'].values)
        scatter_data = np.array([[ 1.12639199e+03,  1.00000000e+01, -6.15736884e+01,
                                  -1.53103521e+01, -1.68892383e-01, -6.40100535e-03,
                                  -8.61815897e-04,  4.88900000e+02],
                                 [ 1.15100631e+03,  1.10000000e+01, -1.06898371e+02,
                                  -2.76794343e+01,  3.53857297e+00, -1.11479855e-02,
                                   1.28026956e-03,  4.88900000e+02],
                                 [ 1.24937064e+03,  1.20000000e+01,  5.23431984e+01,
                                   5.17874012e+00, -8.24615516e+00,  5.19066673e-03,
                                  -5.18260038e-03,  4.88900000e+02],
                                 [ 1.37094149e+03,  1.30000000e+01,  3.49746537e+01,
                                  -9.43653998e+00, -8.72689935e-02,  3.05559747e-03,
                                   6.47745423e-04,  4.88900000e+02],
                                 [ 1.39064221e+03,  1.40000000e+01, -5.31532967e+01,
                                  -5.65151249e+00, -6.11336634e+00, -5.28356488e-03,
                                  -5.16165231e-03,  4.88900000e+02],
                                 [ 1.44754882e+03,  1.50000000e+01, -1.25064010e+02,
                                  -2.49033712e+01, -8.12931706e-02, -1.28030529e-02,
                                  -1.66110131e-03,  4.88900000e+02]])
        raman_data = np.array([[1.12639199e+03, 1.00000000e+01, 3.61528920e-04, 5.94192417e-01,
                                1.90792325e+01, 4.88900000e+02],
                               [1.15100631e+03, 1.10000000e+01, 3.36862711e-02, 1.62723253e-01,
                                1.12706729e+01, 4.88900000e+02],
                               [1.24937064e+03, 1.20000000e+01, 3.10356963e-01, 1.61670230e+00,
                                1.07598727e+02, 4.88900000e+02],
                               [1.37094149e+03, 1.30000000e+01, 6.63217766e-03, 2.37302109e-01,
                                8.78745947e+00, 4.88900000e+02],
                               [1.39064221e+03, 1.40000000e+01, 6.30361373e-02, 8.04145907e-01,
                                3.70791737e+01, 4.88900000e+02],
                               [1.44754882e+03, 1.50000000e+01, 5.36564516e-05, 1.07944901e+00,
                                3.45520265e+01, 4.88900000e+02]])
        scatter_data = scatter_data.T.copy()
        raman_data = raman_data.T.copy()

        # test all columns of the respective dataframe to get a better sense of what is broken
        self.assertTrue(np.allclose(va_corr.scatter['freq'].values,           scatter_data[0], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['freqdx'].values,         scatter_data[1], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['beta_g*1e6'].values,     scatter_data[2], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['beta_A*1e6'].values,     scatter_data[3], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['alpha_g*1e6'].values,    scatter_data[4], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['backscatter'].values,    scatter_data[5], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['forwardscatter'].values, scatter_data[6], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.scatter['exc_freq'].values,       scatter_data[7], rtol=5e-4))

        self.assertTrue(np.allclose(va_corr.raman['freq'].values,          raman_data[0], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.raman['freqdx'].values,        raman_data[1], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.raman['alpha_squared'].values, raman_data[2], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.raman['beta_alpha'].values,    raman_data[3], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.raman['raman_int'].values,     raman_data[4], rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.raman['exc_freq'].values,      raman_data[5], rtol=5e-4))

#class TestZPVC(TestCase):
    def test_zpvc(self):
        nitro_freq.parse_frequency()
        nitro_freq.parse_frequency_ext()
        path = sep.join([TMPDIR, 'nitromalonamide_nmr', '*'])
        va_corr = VA()
        va_corr.gradient = get_data(path=path, attr='gradient', soft=gOutput, f_start='nitromal_grad_',
                                    f_end='.out')
        va_corr.property = get_data(path=path, attr='nmr_shielding', soft=gOutput,
                                    f_start='nitromal_prop', f_end='.out').groupby(
                                    'atom').get_group(0)[['isotropic', 'file']].reset_index(drop=True)
        delta = gen_delta(delta_type=2, freq=nitro_freq.frequency.copy())

        va_corr.zpvc(uni=nitro_freq, delta=delta['delta'].values, temperature=[0, 200])
        zpvc_results = np.array([[ 13.9329    ,  -1.80706136,  12.12583864,  -2.65173195,
          0.84467059,   0.        ],
       [ 13.9329    ,  -1.48913965,  12.44376035,  -2.39264653,
          0.90350687, 200.        ]])
        eff_coord = np.array([[ 1.        ,  0.43933078, -4.1685104 ,  0.        ],
       [ 1.        , -5.89314484, -2.18542914,  0.        ],
       [ 1.        , -4.92050271,  1.02704248,  0.        ],
       [ 1.        ,  6.27207499, -0.61188352,  0.        ],
       [ 1.        ,  4.51177692,  2.23712277,  0.        ],
       [ 6.        ,  2.50712078, -1.03455595,  0.        ],
       [ 8.        ,  2.68478934, -3.43995936,  0.        ],
       [ 7.        ,  4.63712862,  0.33562593,  0.        ],
       [ 6.        , -0.00921467,  0.10958486,  0.        ],
       [ 6.        , -2.14821065, -1.60648521,  0.        ],
       [ 8.        , -1.72316811, -4.00875859,  0.        ],
       [ 7.        , -0.356593  ,  2.76254918,  0.        ],
       [ 8.        ,  1.51924782,  4.1859399 ,  0.        ],
       [ 8.        , -2.53953864,  3.65511737,  0.        ],
       [ 7.        , -4.55751742, -0.8483932 ,  0.        ],
       [ 1.        ,  0.42984441, -4.17018127,  0.        ],
       [ 1.        , -5.88480085, -2.18035498,  0.        ],
       [ 1.        , -4.91515264,  1.02280195,  0.        ],
       [ 1.        ,  6.26195233, -0.60839022,  0.        ],
       [ 1.        ,  4.50575843,  2.22964105,  0.        ],
       [ 6.        ,  2.50646903, -1.03456634,  0.        ],
       [ 8.        ,  2.68667123, -3.43354071,  0.        ],
       [ 7.        ,  4.62935451,  0.33263776,  0.        ],
       [ 6.        , -0.00919927,  0.1084463 ,  0.        ],
       [ 6.        , -2.14622625, -1.60477806,  0.        ],
       [ 8.        , -1.72498822, -4.00524579,  0.        ],
       [ 7.        , -0.35655368,  2.76183552,  0.        ],
       [ 8.        ,  1.50986822,  4.18058144,  0.        ],
       [ 8.        , -2.52931977,  3.65383296,  0.        ],
       [ 7.        , -4.55110836, -0.84857983,  0.        ]])
        vib_average = np.array([[ 8.50080000e+01,  0.00000000e+00, -0.00000000e+00,
         7.20197387e-03,  7.20197387e-03,  0.00000000e+00],
       [ 8.92980000e+01,  1.00000000e+00, -0.00000000e+00,
        -1.92131709e-03, -1.92131709e-03,  0.00000000e+00],
       [ 1.46093800e+02,  2.00000000e+00, -0.00000000e+00,
         2.20218342e-02,  2.20218342e-02,  0.00000000e+00],
       [ 2.17704400e+02,  3.00000000e+00, -0.00000000e+00,
         9.06126467e-04,  9.06126467e-04,  0.00000000e+00],
       [ 3.21218700e+02,  4.00000000e+00, -1.04570441e+00,
         7.43587697e-02, -9.71345643e-01,  0.00000000e+00],
       [ 3.54578000e+02,  5.00000000e+00, -1.69193968e-01,
         1.02299102e-02, -1.58964058e-01,  0.00000000e+00],
       [ 4.01846100e+02,  6.00000000e+00, -9.95359531e-04,
         3.02869005e-03,  2.03333052e-03,  0.00000000e+00],
       [ 4.18516500e+02,  7.00000000e+00, -0.00000000e+00,
        -1.59091073e-02, -1.59091073e-02,  0.00000000e+00],
       [ 4.25136100e+02,  8.00000000e+00,  9.96865969e-02,
         9.60161049e-03,  1.09288207e-01,  0.00000000e+00],
       [ 4.33864900e+02,  9.00000000e+00, -0.00000000e+00,
        -2.24181573e-02, -2.24181573e-02,  0.00000000e+00],
       [ 4.61383400e+02,  1.00000000e+01,  3.02755804e-01,
         7.35128407e-02,  3.76268645e-01,  0.00000000e+00],
       [ 4.85255900e+02,  1.10000000e+01,  5.73683252e-04,
         3.43416159e-03,  4.00784484e-03,  0.00000000e+00],
       [ 6.09549200e+02,  1.20000000e+01,  8.95579596e-02,
         9.59664081e-03,  9.91546004e-02,  0.00000000e+00],
       [ 6.66622400e+02,  1.30000000e+01, -0.00000000e+00,
        -2.76336851e-03, -2.76336851e-03,  0.00000000e+00],
       [ 6.85145800e+02,  1.40000000e+01, -0.00000000e+00,
        -7.69008204e-03, -7.69008204e-03,  0.00000000e+00],
       [ 7.03986700e+02,  1.50000000e+01, -3.59227410e-01,
         2.76777550e-02, -3.31549655e-01,  0.00000000e+00],
       [ 7.14892300e+02,  1.60000000e+01, -0.00000000e+00,
         2.59325707e-03,  2.59325707e-03,  0.00000000e+00],
       [ 7.25846000e+02,  1.70000000e+01, -0.00000000e+00,
        -6.41058056e-03, -6.41058056e-03,  0.00000000e+00],
       [ 7.62762300e+02,  1.80000000e+01, -0.00000000e+00,
        -5.35324542e-03, -5.35324542e-03,  0.00000000e+00],
       [ 8.46200900e+02,  1.90000000e+01, -3.55846141e-03,
        -8.75803015e-04, -4.43426443e-03,  0.00000000e+00],
       [ 1.07527990e+03,  2.00000000e+01, -1.84207554e-02,
         4.72086557e-03, -1.36998899e-02,  0.00000000e+00],
       [ 1.09465730e+03,  2.10000000e+01, -3.15295434e-02,
         5.46069862e-03, -2.60688448e-02,  0.00000000e+00],
       [ 1.10619190e+03,  2.20000000e+01, -0.00000000e+00,
         4.49794220e-02,  4.49794220e-02,  0.00000000e+00],
       [ 1.16155690e+03,  2.30000000e+01, -1.07394061e-02,
         2.19152295e-03, -8.54788314e-03,  0.00000000e+00],
       [ 1.17408590e+03,  2.40000000e+01,  2.00299542e-02,
         9.48213114e-03,  2.95120853e-02,  0.00000000e+00],
       [ 1.26700700e+03,  2.50000000e+01, -5.20609316e-01,
         1.81355739e-01, -3.39253577e-01,  0.00000000e+00],
       [ 1.31668580e+03,  2.60000000e+01, -3.86074967e-03,
         2.99473191e-03, -8.66017757e-04,  0.00000000e+00],
       [ 1.39527270e+03,  2.70000000e+01, -1.48603969e-03,
         3.70156572e-03,  2.21552603e-03,  0.00000000e+00],
       [ 1.45205880e+03,  2.80000000e+01, -2.50446596e-03,
        -1.17089006e-03, -3.67535602e-03,  0.00000000e+00],
       [ 1.55570980e+03,  2.90000000e+01, -6.91414901e-04,
        -9.62715516e-04, -1.65413042e-03,  0.00000000e+00],
       [ 1.57585090e+03,  3.00000000e+01,  4.04075730e-03,
        -1.72319285e-03,  2.31756445e-03,  0.00000000e+00],
       [ 1.59816940e+03,  3.10000000e+01, -3.02221180e-03,
        -1.26461431e-02, -1.56683549e-02,  0.00000000e+00],
       [ 1.63134120e+03,  3.20000000e+01, -3.85968122e-02,
         1.18178497e-02, -2.67789625e-02,  0.00000000e+00],
       [ 1.71103720e+03,  3.30000000e+01, -5.30081127e-02,
         9.07966487e-03, -4.39284478e-02,  0.00000000e+00],
       [ 2.26025580e+03,  3.40000000e+01, -9.00414250e-01,
         4.04124232e-01, -4.96290017e-01,  0.00000000e+00],
       [ 3.52007010e+03,  3.50000000e+01, -3.79625326e-04,
         7.67211701e-04,  3.87586375e-04,  0.00000000e+00],
       [ 3.54188550e+03,  3.60000000e+01, -2.40654106e-03,
        -3.52958626e-04, -2.75949969e-03,  0.00000000e+00],
       [ 3.68688910e+03,  3.70000000e+01, -1.40898432e-03,
         3.54638132e-04, -1.05434619e-03,  0.00000000e+00],
       [ 3.69638850e+03,  3.80000000e+01, -6.18866036e-04,
        -3.25687908e-04, -9.44553944e-04,  0.00000000e+00],
       [ 8.50080000e+01,  0.00000000e+00, -0.00000000e+00,
         2.42846696e-02,  2.42846696e-02,  2.00000000e+02],
       [ 8.92980000e+01,  1.00000000e+00, -0.00000000e+00,
        -6.18637794e-03, -6.18637794e-03,  2.00000000e+02],
       [ 1.46093800e+02,  2.00000000e+00, -0.00000000e+00,
         4.56979114e-02,  4.56979114e-02,  2.00000000e+02],
       [ 2.17704400e+02,  3.00000000e+00, -0.00000000e+00,
         1.38459170e-03,  1.38459170e-03,  2.00000000e+02],
       [ 3.21218700e+02,  4.00000000e+00, -1.01880224e+00,
         9.07354499e-02, -9.28066794e-01,  2.00000000e+02],
       [ 3.54578000e+02,  5.00000000e+00, -1.47884125e-01,
         1.19615757e-02, -1.35922549e-01,  2.00000000e+02],
       [ 4.01846100e+02,  6.00000000e+00, -1.06537672e-03,
         3.38490386e-03,  2.31952714e-03,  2.00000000e+02],
       [ 4.18516500e+02,  7.00000000e+00, -0.00000000e+00,
        -1.75578234e-02, -1.75578234e-02,  2.00000000e+02],
       [ 4.25136100e+02,  8.00000000e+00,  1.13889092e-01,
         1.05481068e-02,  1.24437199e-01,  2.00000000e+02],
       [ 4.33864900e+02,  9.00000000e+00, -0.00000000e+00,
        -2.44873679e-02, -2.44873679e-02,  2.00000000e+02],
       [ 4.61383400e+02,  1.00000000e+01,  3.50247546e-01,
         7.90337919e-02,  4.29281338e-01,  2.00000000e+02],
       [ 4.85255900e+02,  1.10000000e+01,  1.71633606e-04,
         3.65009840e-03,  3.82173201e-03,  2.00000000e+02],
       [ 6.09549200e+02,  1.20000000e+01,  7.81791456e-02,
         9.83892770e-03,  8.80180732e-02,  2.00000000e+02],
       [ 6.66622400e+02,  1.30000000e+01, -0.00000000e+00,
        -2.80944797e-03, -2.80944797e-03,  2.00000000e+02],
       [ 6.85145800e+02,  1.40000000e+01, -0.00000000e+00,
        -7.80220122e-03, -7.80220122e-03,  2.00000000e+02],
       [ 7.03986700e+02,  1.50000000e+01, -2.69187645e-01,
         2.80298166e-02, -2.41157829e-01,  2.00000000e+02],
       [ 7.14892300e+02,  1.60000000e+01, -0.00000000e+00,
         2.62373990e-03,  2.62373990e-03,  2.00000000e+02],
       [ 7.25846000e+02,  1.70000000e+01, -0.00000000e+00,
        -6.48019409e-03, -6.48019409e-03,  2.00000000e+02],
       [ 7.62762300e+02,  1.80000000e+01, -0.00000000e+00,
        -5.39776338e-03, -5.39776338e-03,  2.00000000e+02],
       [ 8.46200900e+02,  1.90000000e+01,  2.39979858e-03,
        -8.79791824e-04,  1.52000675e-03,  2.00000000e+02],
       [ 1.07527990e+03,  2.00000000e+01, -1.08417544e-02,
         4.72499608e-03, -6.11675836e-03,  2.00000000e+02],
       [ 1.09465730e+03,  2.10000000e+01, -2.64528256e-03,
         5.46485456e-03,  2.81957200e-03,  2.00000000e+02],
       [ 1.10619190e+03,  2.20000000e+01, -0.00000000e+00,
         4.50109276e-02,  4.50109276e-02,  2.00000000e+02],
       [ 1.16155690e+03,  2.30000000e+01, -6.76838012e-03,
         2.19255359e-03, -4.57582653e-03,  2.00000000e+02],
       [ 1.17408590e+03,  2.40000000e+01,  9.29716984e-03,
         9.48620604e-03,  1.87833759e-02,  2.00000000e+02],
       [ 1.26700700e+03,  2.50000000e+01, -5.14657884e-01,
         1.81395679e-01, -3.33262205e-01,  2.00000000e+02],
       [ 1.31668580e+03,  2.60000000e+01, -2.24279961e-02,
         2.99519325e-03, -1.94328029e-02,  2.00000000e+02],
       [ 1.39527270e+03,  2.70000000e+01, -9.34932877e-04,
         3.70188971e-03,  2.76695683e-03,  2.00000000e+02],
       [ 1.45205880e+03,  2.80000000e+01,  3.53202807e-03,
        -1.17095817e-03,  2.36106990e-03,  2.00000000e+02],
       [ 1.55570980e+03,  2.90000000e+01, -2.49842213e-04,
        -9.62742087e-04, -1.21258430e-03,  2.00000000e+02],
       [ 1.57585090e+03,  3.00000000e+01,  2.39535538e-03,
        -1.72323400e-03,  6.72121388e-04,  2.00000000e+02],
       [ 1.59816940e+03,  3.10000000e+01,  4.90707509e-03,
        -1.26464003e-02, -7.73932517e-03,  2.00000000e+02],
       [ 1.63134120e+03,  3.20000000e+01, -3.00698127e-02,
         1.18180390e-02, -1.82517738e-02,  2.00000000e+02],
       [ 1.71103720e+03,  3.30000000e+01, -3.95916986e-02,
         9.07974685e-03, -3.05119518e-02,  2.00000000e+02],
       [ 2.26025580e+03,  3.40000000e+01, -8.86090407e-01,
         4.04124303e-01, -4.81966104e-01,  2.00000000e+02],
       [ 3.52007010e+03,  3.50000000e+01, -5.63472032e-04,
         7.67211701e-04,  2.03739669e-04,  2.00000000e+02],
       [ 3.54188550e+03,  3.60000000e+01, -3.39925478e-03,
        -3.52958626e-04, -3.75221341e-03,  2.00000000e+02],
       [ 3.68688910e+03,  3.70000000e+01, -1.73391855e-03,
         3.54638132e-04, -1.37928042e-03,  2.00000000e+02],
       [ 3.69638850e+03,  3.80000000e+01, -7.51344750e-04,
        -3.25687908e-04, -1.07703266e-03,  2.00000000e+02]])
        cols = ['property', 'zpvc', 'zpva', 'tot_anharm', 'tot_curva', 'temp']
        self.assertTrue(np.allclose(va_corr.zpvc_results[cols].values, zpvc_results, rtol=5e-4))
        va_corr.eff_coord['Z'] = va_corr.eff_coord['Z'].astype(int)
        self.assertTrue(np.allclose(va_corr.eff_coord[['Z','x','y','z']].values, eff_coord, atol=5e-5))
        cols = ['freq', 'freqdx', 'anharm', 'curva', 'sum', 'temp']
        self.assertTrue(np.allclose(va_corr.vib_average[cols].values, vib_average, rtol=5e-4))
