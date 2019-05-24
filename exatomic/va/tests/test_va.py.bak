import numpy as np
from unittest import TestCase
from os import sep, remove, rmdir
from tarfile import open
from glob import glob
from inspect import getfile

#from exatomic import va
from exatomic.base import resource
from exatomic.va import VA, get_data, gen_delta
from exatomic.gaussian import Fchk, Output as gOutput
from exatomic.nwchem import Output

class TestGetData(TestCase):
    def setUp(self):
        tar = open(resource('va-vroa-h2o2.tar.bz'), mode='r')
        tar.extractall()
        tar.close()
        tar = open(resource('va-vroa-methyloxirane.tar.bz'), mode='r')
        tar.extractall()
        tar.close()

    def tearDown(self):
        dirs = ['h2o2', 'methyloxirane']
        for dir in dirs:
            path = sep.join([dir, '*'])
            for i in glob(path):
                remove(i)
            rmdir(dir)

    def test_getter_small(self):
        path = sep.join(['h2o2', '*'])
        df = get_data(path=path, attr='roa', soft=Output, f_start='va-roa-h2o2-def2tzvp-514.5-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 130)
        df = get_data(path=path, attr='gradient', soft=Output, f_start='va-roa-h2o2-def2tzvp-514.5-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 52)

    def test_getter_large(self):
        path = sep.join(['methyloxirane', '*'])
        df = get_data(path=path, attr='roa', soft=Output, f_start='va-roa-methyloxirane-def2tzvp-488.9-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 160)
        df = get_data(path=path, attr='gradient', soft=Output, f_start='va-roa-methyloxirane-def2tzvp-488.9-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 160)

class TestVROA(TestCase):
    def setUp(self):
        self.h2o2_freq = Fchk(resource('g16-h2o2-def2tzvp-freq.fchk'))
        self.methyloxirane_freq = Fchk(resource('g16-methyloxirane-def2tzvp-freq.fchk'))
        tar = open(resource('va-vroa-h2o2.tar.bz'), mode='r')
        tar.extractall()
        tar.close()
        tar = open(resource('va-vroa-methyloxirane.tar.bz'), mode='r')
        tar.extractall()
        tar.close()

    def tearDown(self):
        dirs = ['h2o2', 'methyloxirane']
        for dir in dirs:
            path = sep.join([dir, '*'])
            for i in glob(path):
                remove(i)
            rmdir(dir)

    def test_vroa(self):
        self.h2o2_freq.parse_frequency()
        self.h2o2_freq.parse_frequency_ext()
        delta = gen_delta(delta_type=2, freq=self.h2o2_freq.frequency.copy())
        va_corr = VA()
        path = sep.join(['h2o2', '*'])
        va_corr.roa = get_data(path=path, attr='roa', soft=Output, f_start='va-roa-h2o2-def2tzvp-514.5-',
                               f_end='.out')
        va_corr.roa['exc_freq'] = np.tile(514.5, len(va_corr.roa))
        va_corr.gradient = get_data(path=path, attr='gradient', soft=Output,
                                    f_start='va-roa-h2o2-def2tzvp-514.5-', f_end='.out')
        va_corr.gradient['exc_freq'] = np.tile(514.5, len(va_corr.gradient))
        va_corr.vroa(uni=self.h2o2_freq, delta=delta['delta'].values)
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
        scatter_data = scatter_data.T
        raman_data = raman_data.T
        print(va_corr.scatter.to_string())
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
        self.methyloxirane_freq.parse_frequency()
        self.methyloxirane_freq.parse_frequency_ext()
        delta = gen_delta(delta_type=2, freq=self.methyloxirane_freq.frequency.copy())
        va_corr = VA()
        path = sep.join(['methyloxirane', '*'])
        va_corr.roa = get_data(path=path, attr='roa', soft=Output,
                               f_start='va-roa-methyloxirane-def2tzvp-488.9-', f_end='.out')
        va_corr.roa['exc_freq'] = np.tile(488.9, len(va_corr.roa))
        va_corr.gradient = get_data(path=path, attr='gradient', soft=Output,
                                    f_start='va-roa-methyloxirane-def2tzvp-488.9-', f_end='.out')
        va_corr.gradient['exc_freq'] = np.tile(488.9, len(va_corr.gradient))
        va_corr.vroa(uni=self.methyloxirane_freq, delta=delta['delta'].values)
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
        scatter_data = scatter_data.T
        raman_data = raman_data.T

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

class TestZPVC(TestCase):
    def setUp(self):
        self.nitro_freq = gOutput(resource('g09-nitromalonamide-6-31++g-freq.out'))
        tar = open(resource('va-zpvc-nitro_nmr.tar.bz'), mode='r')
        tar.extractall()
        tar.close()

    def tearDown(self):
        dirs = ['nitromalonamide_nmr']
        for dir in dirs:
            path = sep.join([dir, '*'])
            for i in glob(path):
                remove(i)
            rmdir(dir)

    def test_zpvc(self):
        self.nitro_freq.parse_frequency()
        self.nitro_freq.parse_frequency_ext()
        path = sep.join(['nitromalonamide_nmr', '*'])
        va_corr = VA()
        va_corr.gradient = get_data(path=path, attr='gradient', soft=Fchk, f_start='nitromal_grad_',
                                    f_end='.fchk')
        va_corr.property = get_data(path=path, attr='nmr_shielding', soft=gOutput,
                                    f_start='nitromal_prop', f_end='.out').groupby(
                                    'atom').get_group(0)[['isotropic', 'file']].reset_index(drop=True)
        delta = gen_delta(delta_type=2, freq=self.nitro_freq.frequency.copy())

        va_corr.zpvc(uni=self.nitro_freq, delta=delta['delta'].values, temperature=[0, 200])
        zpvc_results = np.array([[ 13.9329    ,  -1.80695969,  12.12594031,  -2.65163028,
          0.84467059,   0.        ],
       [ 13.9329    ,  -1.48901879,  12.44388121,  -2.39252567,
          0.90350687, 200.        ]])
        eff_coord = np.array([[ 1.00000000e+00,  4.39326428e-01, -4.16851446e+00,
         2.15163176e-09],
       [ 1.00000000e+00, -5.89314402e+00, -2.18542497e+00,
        -3.03691475e-09],
       [ 1.00000000e+00, -4.92050048e+00,  1.02704577e+00,
        -5.08064859e-09],
       [ 1.00000000e+00,  6.27207613e+00, -6.11878215e-01,
        -3.73777367e-09],
       [ 1.00000000e+00,  4.51176919e+00,  2.23712394e+00,
        -2.36915085e-09],
       [ 6.00000000e+00,  2.50712167e+00, -1.03455740e+00,
        -2.23346866e-10],
       [ 8.00000000e+00,  2.68478990e+00, -3.43996038e+00,
         8.61699300e-10],
       [ 7.00000000e+00,  4.63712731e+00,  3.35627359e-01,
        -3.36321067e-09],
       [ 6.00000000e+00, -9.21374685e-03,  1.09584266e-01,
         1.11480515e-09],
       [ 6.00000000e+00, -2.14820942e+00, -1.60648467e+00,
         7.65078468e-11],
       [ 8.00000000e+00, -1.72316937e+00, -4.00875887e+00,
         1.81109195e-09],
       [ 7.00000000e+00, -3.56592976e-01,  2.76254803e+00,
         7.25235472e-10],
       [ 8.00000000e+00,  1.51924653e+00,  4.18594015e+00,
         2.32593464e-09],
       [ 8.00000000e+00, -2.53953909e+00,  3.65511628e+00,
        -4.23642657e-10],
       [ 7.00000000e+00, -4.55751540e+00, -8.48390449e-01,
        -2.54843078e-09],
       [ 1.00000000e+00,  4.29838602e-01, -4.17018648e+00,
         2.35499795e-09],
       [ 1.00000000e+00, -5.88479973e+00, -2.18035033e+00,
        -3.46231206e-09],
       [ 1.00000000e+00, -4.91515030e+00,  1.02280567e+00,
        -5.55634059e-09],
       [ 1.00000000e+00,  6.26195355e+00, -6.08383910e-01,
        -4.07318710e-09],
       [ 1.00000000e+00,  4.50574935e+00,  2.22964259e+00,
        -2.67270070e-09],
       [ 6.00000000e+00,  2.50647012e+00, -1.03456802e+00,
        -2.04223970e-10],
       [ 8.00000000e+00,  2.68667199e+00, -3.43354183e+00,
         9.90230813e-10],
       [ 7.00000000e+00,  4.62935294e+00,  3.32639600e-01,
        -3.60787768e-09],
       [ 6.00000000e+00, -9.19816092e-03,  1.08445548e-01,
         1.24919787e-09],
       [ 6.00000000e+00, -2.14622477e+00, -1.60477739e+00,
         5.97145182e-11],
       [ 8.00000000e+00, -1.72498997e+00, -4.00524621e+00,
         1.90745879e-09],
       [ 7.00000000e+00, -3.56553610e-01,  2.76183415e+00,
         8.04926775e-10],
       [ 8.00000000e+00,  1.50986674e+00,  4.18058168e+00,
         2.30950854e-09],
       [ 8.00000000e+00, -2.52932020e+00,  3.65383172e+00,
        -2.54575731e-10],
       [ 7.00000000e+00, -4.55110597e+00, -8.48576674e-01,
        -2.83565496e-09]])
        vib_average = np.array([[ 8.50080000e+01,  0.00000000e+00, -0.00000000e+00,
         7.20197387e-03,  7.20197387e-03,  0.00000000e+00],
       [ 8.92980000e+01,  1.00000000e+00,  0.00000000e+00,
        -1.92131709e-03, -1.92131709e-03,  0.00000000e+00],
       [ 1.46093800e+02,  2.00000000e+00, -0.00000000e+00,
         2.20218342e-02,  2.20218342e-02,  0.00000000e+00],
       [ 2.17704400e+02,  3.00000000e+00,  0.00000000e+00,
         9.06126467e-04,  9.06126467e-04,  0.00000000e+00],
       [ 3.21218700e+02,  4.00000000e+00, -1.04564575e+00,
         7.43587697e-02, -9.71286985e-01,  0.00000000e+00],
       [ 3.54578000e+02,  5.00000000e+00, -1.69185949e-01,
         1.02299102e-02, -1.58956039e-01,  0.00000000e+00],
       [ 4.01846100e+02,  6.00000000e+00, -9.95270482e-04,
         3.02869005e-03,  2.03341957e-03,  0.00000000e+00],
       [ 4.18516500e+02,  7.00000000e+00,  0.00000000e+00,
        -1.59091073e-02, -1.59091073e-02,  0.00000000e+00],
       [ 4.25136100e+02,  8.00000000e+00,  9.97024883e-02,
         9.60161049e-03,  1.09304099e-01,  0.00000000e+00],
       [ 4.33864900e+02,  9.00000000e+00, -0.00000000e+00,
        -2.24181573e-02, -2.24181573e-02,  0.00000000e+00],
       [ 4.61383400e+02,  1.00000000e+01,  3.02764368e-01,
         7.35128407e-02,  3.76277209e-01,  0.00000000e+00],
       [ 4.85255900e+02,  1.10000000e+01,  5.73569581e-04,
         3.43416159e-03,  4.00773117e-03,  0.00000000e+00],
       [ 6.09549200e+02,  1.20000000e+01,  8.95529534e-02,
         9.59664081e-03,  9.91495942e-02,  0.00000000e+00],
       [ 6.66622400e+02,  1.30000000e+01, -0.00000000e+00,
        -2.76336851e-03, -2.76336851e-03,  0.00000000e+00],
       [ 6.85145800e+02,  1.40000000e+01, -0.00000000e+00,
        -7.69008204e-03, -7.69008204e-03,  0.00000000e+00],
       [ 7.03986700e+02,  1.50000000e+01, -3.59229024e-01,
         2.76777550e-02, -3.31551269e-01,  0.00000000e+00],
       [ 7.14892300e+02,  1.60000000e+01, -0.00000000e+00,
         2.59325707e-03,  2.59325707e-03,  0.00000000e+00],
       [ 7.25846000e+02,  1.70000000e+01,  0.00000000e+00,
        -6.41058056e-03, -6.41058056e-03,  0.00000000e+00],
       [ 7.62762300e+02,  1.80000000e+01,  0.00000000e+00,
        -5.35324542e-03, -5.35324542e-03,  0.00000000e+00],
       [ 8.46200900e+02,  1.90000000e+01, -3.55838970e-03,
        -8.75803015e-04, -4.43419272e-03,  0.00000000e+00],
       [ 1.07527990e+03,  2.00000000e+01, -1.84221378e-02,
         4.72086557e-03, -1.37012722e-02,  0.00000000e+00],
       [ 1.09465730e+03,  2.10000000e+01, -3.15270150e-02,
         5.46069862e-03, -2.60663164e-02,  0.00000000e+00],
       [ 1.10619190e+03,  2.20000000e+01, -0.00000000e+00,
         4.49794220e-02,  4.49794220e-02,  0.00000000e+00],
       [ 1.16155690e+03,  2.30000000e+01, -1.07390990e-02,
         2.19152295e-03, -8.54757604e-03,  0.00000000e+00],
       [ 1.17408590e+03,  2.40000000e+01,  2.00297071e-02,
         9.48213114e-03,  2.95118383e-02,  0.00000000e+00],
       [ 1.26700700e+03,  2.50000000e+01, -5.20601430e-01,
         1.81355739e-01, -3.39245691e-01,  0.00000000e+00],
       [ 1.31668580e+03,  2.60000000e+01, -3.86009939e-03,
         2.99473191e-03, -8.65367472e-04,  0.00000000e+00],
       [ 1.39527270e+03,  2.70000000e+01, -1.48591263e-03,
         3.70156572e-03,  2.21565310e-03,  0.00000000e+00],
       [ 1.45205880e+03,  2.80000000e+01, -2.50474550e-03,
        -1.17089006e-03, -3.67563555e-03,  0.00000000e+00],
       [ 1.55570980e+03,  2.90000000e+01, -6.91328870e-04,
        -9.62715516e-04, -1.65404439e-03,  0.00000000e+00],
       [ 1.57585090e+03,  3.00000000e+01,  4.04104497e-03,
        -1.72319285e-03,  2.31785212e-03,  0.00000000e+00],
       [ 1.59816940e+03,  3.10000000e+01, -3.02149365e-03,
        -1.26461431e-02, -1.56676367e-02,  0.00000000e+00],
       [ 1.63134120e+03,  3.20000000e+01, -3.85952628e-02,
         1.18178497e-02, -2.67774131e-02,  0.00000000e+00],
       [ 1.71103720e+03,  3.30000000e+01, -5.30059973e-02,
         9.07966487e-03, -4.39263325e-02,  0.00000000e+00],
       [ 2.26025580e+03,  3.40000000e+01, -9.00411487e-01,
         4.04124232e-01, -4.96287255e-01,  0.00000000e+00],
       [ 3.52007010e+03,  3.50000000e+01, -3.79611761e-04,
         7.67211701e-04,  3.87599940e-04,  0.00000000e+00],
       [ 3.54188550e+03,  3.60000000e+01, -2.40652008e-03,
        -3.52958626e-04, -2.75947871e-03,  0.00000000e+00],
       [ 3.68688910e+03,  3.70000000e+01, -1.40903434e-03,
         3.54638132e-04, -1.05439621e-03,  0.00000000e+00],
       [ 3.69638850e+03,  3.80000000e+01, -6.18852026e-04,
        -3.25687908e-04, -9.44539934e-04,  0.00000000e+00],
       [ 8.50080000e+01,  0.00000000e+00, -0.00000000e+00,
         2.42846696e-02,  2.42846696e-02,  2.00000000e+02],
       [ 8.92980000e+01,  1.00000000e+00,  0.00000000e+00,
        -6.18637794e-03, -6.18637794e-03,  2.00000000e+02],
       [ 1.46093800e+02,  2.00000000e+00, -0.00000000e+00,
         4.56979114e-02,  4.56979114e-02,  2.00000000e+02],
       [ 2.17704400e+02,  3.00000000e+00,  0.00000000e+00,
         1.38459170e-03,  1.38459170e-03,  2.00000000e+02],
       [ 3.21218700e+02,  4.00000000e+00, -1.01873199e+00,
         9.07354499e-02, -9.27996541e-01,  2.00000000e+02],
       [ 3.54578000e+02,  5.00000000e+00, -1.47873118e-01,
         1.19615757e-02, -1.35911542e-01,  2.00000000e+02],
       [ 4.01846100e+02,  6.00000000e+00, -1.06526746e-03,
         3.38490386e-03,  2.31963640e-03,  2.00000000e+02],
       [ 4.18516500e+02,  7.00000000e+00,  0.00000000e+00,
        -1.75578234e-02, -1.75578234e-02,  2.00000000e+02],
       [ 4.25136100e+02,  8.00000000e+00,  1.13907131e-01,
         1.05481068e-02,  1.24455238e-01,  2.00000000e+02],
       [ 4.33864900e+02,  9.00000000e+00, -0.00000000e+00,
        -2.44873679e-02, -2.44873679e-02,  2.00000000e+02],
       [ 4.61383400e+02,  1.00000000e+01,  3.50253963e-01,
         7.90337919e-02,  4.29287755e-01,  2.00000000e+02],
       [ 4.85255900e+02,  1.10000000e+01,  1.71385084e-04,
         3.65009840e-03,  3.82148349e-03,  2.00000000e+02],
       [ 6.09549200e+02,  1.20000000e+01,  7.81729357e-02,
         9.83892770e-03,  8.80118634e-02,  2.00000000e+02],
       [ 6.66622400e+02,  1.30000000e+01, -0.00000000e+00,
        -2.80944797e-03, -2.80944797e-03,  2.00000000e+02],
       [ 6.85145800e+02,  1.40000000e+01, -0.00000000e+00,
        -7.80220122e-03, -7.80220122e-03,  2.00000000e+02],
       [ 7.03986700e+02,  1.50000000e+01, -2.69188069e-01,
         2.80298166e-02, -2.41158252e-01,  2.00000000e+02],
       [ 7.14892300e+02,  1.60000000e+01, -0.00000000e+00,
         2.62373990e-03,  2.62373990e-03,  2.00000000e+02],
       [ 7.25846000e+02,  1.70000000e+01,  0.00000000e+00,
        -6.48019409e-03, -6.48019409e-03,  2.00000000e+02],
       [ 7.62762300e+02,  1.80000000e+01,  0.00000000e+00,
        -5.39776338e-03, -5.39776338e-03,  2.00000000e+02],
       [ 8.46200900e+02,  1.90000000e+01,  2.39991549e-03,
        -8.79791824e-04,  1.52012367e-03,  2.00000000e+02],
       [ 1.07527990e+03,  2.00000000e+01, -1.08435067e-02,
         4.72499608e-03, -6.11851066e-03,  2.00000000e+02],
       [ 1.09465730e+03,  2.10000000e+01, -2.64240457e-03,
         5.46485456e-03,  2.82244999e-03,  2.00000000e+02],
       [ 1.10619190e+03,  2.20000000e+01, -0.00000000e+00,
         4.50109276e-02,  4.50109276e-02,  2.00000000e+02],
       [ 1.16155690e+03,  2.30000000e+01, -6.76804462e-03,
         2.19255359e-03, -4.57549103e-03,  2.00000000e+02],
       [ 1.17408590e+03,  2.40000000e+01,  9.29681629e-03,
         9.48620604e-03,  1.87830223e-02,  2.00000000e+02],
       [ 1.26700700e+03,  2.50000000e+01, -5.14647936e-01,
         1.81395679e-01, -3.33252258e-01,  2.00000000e+02],
       [ 1.31668580e+03,  2.60000000e+01, -2.24272013e-02,
         2.99519325e-03, -1.94320081e-02,  2.00000000e+02],
       [ 1.39527270e+03,  2.70000000e+01, -9.34763385e-04,
         3.70188971e-03,  2.76712632e-03,  2.00000000e+02],
       [ 1.45205880e+03,  2.80000000e+01,  3.53169705e-03,
        -1.17095817e-03,  2.36073888e-03,  2.00000000e+02],
       [ 1.55570980e+03,  2.90000000e+01, -2.49739905e-04,
        -9.62742087e-04, -1.21248199e-03,  2.00000000e+02],
       [ 1.57585090e+03,  3.00000000e+01,  2.39570050e-03,
        -1.72323400e-03,  6.72466499e-04,  2.00000000e+02],
       [ 1.59816940e+03,  3.10000000e+01,  4.90804444e-03,
        -1.26464003e-02, -7.73835581e-03,  2.00000000e+02],
       [ 1.63134120e+03,  3.20000000e+01, -3.00679005e-02,
         1.18180390e-02, -1.82498615e-02,  2.00000000e+02],
       [ 1.71103720e+03,  3.30000000e+01, -3.95890309e-02,
         9.07974685e-03, -3.05092840e-02,  2.00000000e+02],
       [ 2.26025580e+03,  3.40000000e+01, -8.86086292e-01,
         4.04124303e-01, -4.81961990e-01,  2.00000000e+02],
       [ 3.52007010e+03,  3.50000000e+01, -5.63455564e-04,
         7.67211701e-04,  2.03756137e-04,  2.00000000e+02],
       [ 3.54188550e+03,  3.60000000e+01, -3.39923100e-03,
        -3.52958626e-04, -3.75218963e-03,  2.00000000e+02],
       [ 3.68688910e+03,  3.70000000e+01, -1.73397372e-03,
         3.54638132e-04, -1.37933559e-03,  2.00000000e+02],
       [ 3.69638850e+03,  3.80000000e+01, -7.51329849e-04,
        -3.25687908e-04, -1.07701776e-03,  2.00000000e+02]])
        self.assertTrue(np.allclose(va_corr.zpvc_results.values, zpvc_results, rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.eff_coord[['Z','x','y','z']].values, eff_coord, rtol=5e-4))
        self.assertTrue(np.allclose(va_corr.vib_average.values, vib_average, rtol=5e-4))

