import numpy as np
import pandas as pd
from unittest import TestCase
from os import sep

#from exatomic import va
from exatomic.base import resource
from exatomic.va import VA, get_data, gen_delta
from exatomic.gaussian import Fchk
from exatomic.nwchem import Output

class TestGetData(TestCase):
    def setUp(self):
        pass

    def test_getter_small(self):
        path = sep.join(resource('va-roa-h2o2-def2tzvp-514.5-00.out').split(sep)[:-1])+sep+'*'
        df = get_data(path=path, attr='roa', soft=Output, f_start='va-roa-h2o2-def2tzvp-514.5-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 130)
        df = get_data(path=path, attr='gradient', soft=Output, f_start='va-roa-h2o2-def2tzvp-514.5-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 52)

    def test_getter_large(self):
        path = sep.join(resource('va-roa-methyloxirane-def2tzvp-488.9-00.out').split(sep)[:-1])+sep+'*'
        df = get_data(path=path, attr='roa', soft=Output, f_start='va-roa-methyloxirane-def2tzvp-488.9-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 490)
        df = get_data(path=path, attr='gradient', soft=Output, f_start='va-roa-methyloxirane-def2tzvp-488.9-',
                      f_end='.out')
        self.assertEqual(df.shape[0], 490)

class TestVROA(TestCase):
    def setUp(self):
        self.h2o2_freq = Fchk(resource('g16-h2o2-def2tzvp-freq.fchk'))
        self.methyloxirane_freq = Fchk(resource('g16-methyloxirane-def2tzvp-freq.fchk'))
        self.naproxen_freq = Fchk(resource('g16-naproxen-def2tzvp-freq.fchk'))

    def test_vroa_small(self):
        self.h2o2_freq.parse_frequency()
        self.h2o2_freq.parse_frequency_ext()
        delta = gen_delta(delta_type=2, freq=self.h2o2_freq.frequency.copy())
        va_corr = VA()
        path = sep.join(resource('va-roa-h2o2-def2tzvp-514.5-00.out').split(sep)[:-1])+sep+'*'
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
        self.assertTrue(np.allclose(va_corr.scatter['freq'].values,           scatter_data[0]))
        self.assertTrue(np.allclose(va_corr.scatter['freqdx'].values,         scatter_data[1]))
        self.assertTrue(np.allclose(va_corr.scatter['beta_g*1e6'].values,     scatter_data[2]))
        self.assertTrue(np.allclose(va_corr.scatter['beta_A*1e6'].values,     scatter_data[3]))
        self.assertTrue(np.allclose(va_corr.scatter['alpha_g*1e6'].values,    scatter_data[4]))
        self.assertTrue(np.allclose(va_corr.scatter['backscatter'].values,    scatter_data[5]))
        self.assertTrue(np.allclose(va_corr.scatter['forwardscatter'].values, scatter_data[6]))
        self.assertTrue(np.allclose(va_corr.scatter['exc_freq'].values,       scatter_data[7]))

        self.assertTrue(np.allclose(va_corr.raman['freq'].values,          raman_data[0]))
        self.assertTrue(np.allclose(va_corr.raman['freqdx'].values,        raman_data[1]))
        self.assertTrue(np.allclose(va_corr.raman['alpha_squared'].values, raman_data[2]))
        self.assertTrue(np.allclose(va_corr.raman['beta_alpha'].values,    raman_data[3]))
        self.assertTrue(np.allclose(va_corr.raman['raman_int'].values,     raman_data[4]))
        self.assertTrue(np.allclose(va_corr.raman['exc_freq'].values,      raman_data[5]))

    def test_vroa_large(self):
        self.methyloxirane_freq.parse_frequency()
        self.methyloxirane_freq.parse_frequency_ext()
        delta = gen_delta(delta_type=2, freq=self.methyloxirane_freq.frequency.copy())
        va_corr = VA()
        path = sep.join(resource('va-roa-methyloxirane-def2tzvp-488.9-00.out').split(sep)[:-1])+sep+'*'
        va_corr.roa = get_data(path=path, attr='roa', soft=Output, 
                               f_start='va-roa-methyloxirane-def2tzvp-488.9-', f_end='.out')
        va_corr.roa['exc_freq'] = np.tile(488.9, len(va_corr.roa))
        va_corr.gradient = get_data(path=path, attr='gradient', soft=Output, 
                                    f_start='va-roa-methyloxirane-def2tzvp-488.9-', f_end='.out')
        va_corr.gradient['exc_freq'] = np.tile(488.9, len(va_corr.gradient))
        va_corr.vroa(uni=self.methyloxirane_freq, delta=delta['delta'].values)
        scatter_data = np.array([[ 2.05459565e+02,  0.00000000e+00,  1.00172076e+01,
                                  -4.84850486e-01, -7.65869047e-02,  9.46136710e-04,
                                   1.12890357e-04,  4.88900000e+02],
                                 [ 3.55662940e+02,  1.00000000e+00, -3.44953930e+01,
                                   1.45417970e+00,  8.58321820e-02, -3.26502398e-03,
                                  -5.13393993e-04,  4.88900000e+02],
                                 [ 3.99667232e+02,  2.00000000e+00,  6.39931533e+00,
                                   6.29944687e+00, -1.61250429e+00,  8.15916572e-04,
                                  -1.15940520e-03,  4.88900000e+02],
                                 [ 7.20214472e+02,  3.00000000e+00,  1.56891938e+02,
                                   6.22441044e+01,  2.82125161e+00,  1.70534374e-02,
                                   3.54566649e-03,  4.88900000e+02],
                                 [ 7.98848893e+02,  4.00000000e+00, -1.70095872e+02,
                                  -3.28563386e+01, -6.15105350e-01, -1.73806065e-02,
                                  -2.63870838e-03,  4.88900000e+02],
                                 [ 8.77034566e+02,  5.00000000e+00, -2.04265067e+02,
                                  -1.05678239e+02,  1.62956575e+00, -2.29911500e-02,
                                  -4.04101912e-04,  4.88900000e+02],
                                 [ 9.31237454e+02,  6.00000000e+00,  3.90319374e+01,
                                   8.42046844e+00,  3.80748182e+00,  4.01652098e-03,
                                   3.23117041e-03,  4.88900000e+02],
                                 [ 1.00859012e+03,  7.00000000e+00,  6.12759549e+01,
                                   3.27702794e+01,  1.71350157e+00,  6.93114061e-03,
                                   1.68981194e-03,  4.88900000e+02],
                                 [ 1.09121407e+03,  8.00000000e+00, -3.31548180e+01,
                                  -1.90146042e+01, -2.29002434e+00, -3.79132986e-03,
                                  -1.87506094e-03,  4.88900000e+02],
                                 [ 1.11502472e+03,  9.00000000e+00,  1.76435446e+02,
                                   2.06030018e+01,  1.71120213e-02,  1.75970989e-02,
                                   2.50563976e-03,  4.88900000e+02],
                                 [ 1.12639147e+03,  1.00000000e+01, -6.15736884e+01,
                                  -1.53103591e+01, -1.68892383e-01, -6.40100557e-03,
                                  -8.61815785e-04,  4.88900000e+02],
                                 [ 1.15100578e+03,  1.10000000e+01, -1.06898371e+02,
                                  -2.76794469e+01,  3.53857297e+00, -1.11479859e-02,
                                   1.28026976e-03,  4.88900000e+02],
                                 [ 1.24937007e+03,  1.20000000e+01,  5.23431984e+01,
                                   5.17874250e+00, -8.24615516e+00,  5.19066681e-03,
                                  -5.18260042e-03,  4.88900000e+02],
                                 [ 1.37094086e+03,  1.30000000e+01,  3.49746537e+01,
                                  -9.43654431e+00, -8.72689935e-02,  3.05559733e-03,
                                   6.47745492e-04,  4.88900000e+02],
                                 [ 1.39064157e+03,  1.40000000e+01, -5.31532967e+01,
                                  -5.65151508e+00, -6.11336634e+00, -5.28356496e-03,
                                  -5.16165227e-03,  4.88900000e+02],
                                 [ 1.44754816e+03,  1.50000000e+01, -1.25064010e+02,
                                  -2.49033826e+01, -8.12931706e-02, -1.28030532e-02,
                                  -1.66110113e-03,  4.88900000e+02],
                                 [ 1.46057174e+03,  1.60000000e+01,  2.84441235e+02,
                                   1.43685285e+02, -3.04958576e-01,  3.19042876e-02,
                                   2.03252503e-03,  4.88900000e+02],
                                 [ 1.48502219e+03,  1.70000000e+01, -1.01819169e+02,
                                  -3.42973178e+01, -2.05616545e-02, -1.08721544e-02,
                                  -1.09515401e-03,  4.88900000e+02],
                                 [ 2.96096211e+03,  1.80000000e+01, -1.35290755e+02,
                                   3.21865414e+01, -3.31715316e+01, -1.19579431e-02,
                                  -2.65631395e-02,  4.88900000e+02],
                                 [ 3.00376807e+03,  1.90000000e+01,  9.70860850e+02,
                                  -3.36381589e+01,  4.91997388e+01,  9.21262205e-02,
                                   5.14957961e-02,  4.88900000e+02],
                                 [ 3.01002817e+03,  2.00000000e+01, -1.76293561e+03,
                                  -2.29728801e+02, -5.81262246e+01, -1.76593141e-01,
                                  -6.63821907e-02,  4.88900000e+02],
                                 [ 3.01367481e+03,  2.10000000e+01, -3.19566339e+02,
                                  -5.79348975e+02,  1.37233534e+00, -4.92175358e-02,
                                   5.14460361e-03,  4.88900000e+02],
                                 [ 3.03860542e+03,  2.20000000e+01,  7.10230696e+02,
                                   3.37812640e+02, -1.97402680e+01,  7.89921513e-02,
                                  -8.25430403e-03,  4.88900000e+02],
                                 [ 3.08355477e+03,  2.30000000e+01,  4.62044916e+02,
                                   2.61514664e+02,  3.67072678e+01,  5.27247812e-02,
                                   2.96377168e-02,  4.88900000e+02]])
        raman_data = np.array([[2.05459565e+02, 0.00000000e+00, 2.10098597e-04, 2.20562506e-02,
                                7.43617766e-01, 4.88900000e+02],
                               [3.55662940e+02, 1.00000000e+00, 2.93751735e-03, 1.40241476e-01,
                                5.01648035e+00, 4.88900000e+02],
                               [3.99667232e+02, 2.00000000e+00, 1.54918754e-02, 7.04053294e-02,
                                5.04150812e+00, 4.88900000e+02],
                               [7.20214472e+02, 3.00000000e+00, 1.75562149e-02, 9.53021435e-01,
                                3.36568046e+01, 4.88900000e+02],
                               [7.98848893e+02, 4.00000000e+00, 1.97514347e-02, 1.34115770e+00,
                                4.64723046e+01, 4.88900000e+02],
                               [8.77034566e+02, 5.00000000e+00, 1.04992888e-02, 8.67314327e-01,
                                2.96439305e+01, 4.88900000e+02],
                               [9.31237454e+02, 6.00000000e+00, 7.69920991e-02, 1.73225648e-01,
                                1.94017986e+01, 4.88900000e+02],
                               [1.00859012e+03, 7.00000000e+00, 1.10923863e-02, 1.07637928e-01,
                                5.44104322e+00, 4.88900000e+02],
                               [1.09121407e+03, 8.00000000e+00, 1.07356764e-02, 3.67604101e-01,
                                1.36957530e+01, 4.88900000e+02],
                               [1.11502472e+03, 9.00000000e+00, 4.74311823e-06, 5.84222780e-01,
                                1.86959827e+01, 4.88900000e+02],
                               [1.12639147e+03, 1.00000000e+01, 3.61528920e-04, 5.94192417e-01,
                                1.90792325e+01, 4.88900000e+02],
                               [1.15100578e+03, 1.10000000e+01, 3.36862711e-02, 1.62723253e-01,
                                1.12706729e+01, 4.88900000e+02],
                               [1.24937007e+03, 1.20000000e+01, 3.10356963e-01, 1.61670230e+00,
                                1.07598727e+02, 4.88900000e+02],
                               [1.37094086e+03, 1.30000000e+01, 6.63217766e-03, 2.37302109e-01,
                                8.78745947e+00, 4.88900000e+02],
                               [1.39064157e+03, 1.40000000e+01, 6.30361373e-02, 8.04145907e-01,
                                3.70791737e+01, 4.88900000e+02],
                               [1.44754816e+03, 1.50000000e+01, 5.36564516e-05, 1.07944901e+00,
                                3.45520265e+01, 4.88900000e+02],
                               [1.46057174e+03, 1.60000000e+01, 2.60248717e-04, 1.30442400e+00,
                                4.17884127e+01, 4.88900000e+02],
                               [1.48502219e+03, 1.70000000e+01, 9.64922482e-04, 5.73382145e-01,
                                1.85219147e+01, 4.88900000e+02],
                               [2.96096211e+03, 1.80000000e+01, 3.85689581e+00, 1.46006131e+00,
                                7.40963207e+02, 4.88900000e+02],
                               [3.00376807e+03, 1.90000000e+01, 2.08975720e+00, 4.62106278e+00,
                                5.24030304e+02, 4.88900000e+02],
                               [3.01002817e+03, 2.00000000e+01, 2.36288489e+00, 1.74433955e+01,
                                9.83507935e+02, 4.88900000e+02],
                               [3.01367481e+03, 2.10000000e+01, 2.62421130e-04, 1.32487075e+01,
                                4.24005876e+02, 4.88900000e+02],
                               [3.03860542e+03, 2.20000000e+01, 1.58320259e-01, 3.28628068e+00,
                                1.33658628e+02, 4.88900000e+02],
                               [3.08355477e+03, 2.30000000e+01, 9.45779026e-02, 1.28427301e+01,
                                4.27991386e+02, 4.88900000e+02]])
        scatter_data = scatter_data.T
        raman_data = raman_data.T

        # test all columns of the respective dataframe to get a better sense of what is broken
        self.assertTrue(np.allclose(va_corr.scatter['freq'].values,           scatter_data[0]))
        self.assertTrue(np.allclose(va_corr.scatter['freqdx'].values,         scatter_data[1]))
        self.assertTrue(np.allclose(va_corr.scatter['beta_g*1e6'].values,     scatter_data[2]))
        self.assertTrue(np.allclose(va_corr.scatter['beta_A*1e6'].values,     scatter_data[3]))
        self.assertTrue(np.allclose(va_corr.scatter['alpha_g*1e6'].values,    scatter_data[4]))
        self.assertTrue(np.allclose(va_corr.scatter['backscatter'].values,    scatter_data[5]))
        self.assertTrue(np.allclose(va_corr.scatter['forwardscatter'].values, scatter_data[6]))
        self.assertTrue(np.allclose(va_corr.scatter['exc_freq'].values,       scatter_data[7]))

        self.assertTrue(np.allclose(va_corr.raman['freq'].values,          raman_data[0]))
        self.assertTrue(np.allclose(va_corr.raman['freqdx'].values,        raman_data[1]))
        self.assertTrue(np.allclose(va_corr.raman['alpha_squared'].values, raman_data[2]))
        self.assertTrue(np.allclose(va_corr.raman['beta_alpha'].values,    raman_data[3]))
        self.assertTrue(np.allclose(va_corr.raman['raman_int'].values,     raman_data[4]))
        self.assertTrue(np.allclose(va_corr.raman['exc_freq'].values,      raman_data[5]))

    def test_select_modes(self):
        self.naproxen_freq.parse_frequency()
        self.naproxen_freq.parse_frequency_ext()
        va_corr = VA()
        name='va-roa-naproxen-def2tzvp-355.9-'
        path='/'.join(resource(name+'000.out').split('/')[:-1])+'/*'
        va_corr.roa = get_data(path=path, attr='roa', soft=Output, f_start=name, f_end='.out')
        va_corr.roa['exc_freq'] = np.tile(355.9, len(va_corr.roa))
        va_corr.gradient = get_data(path=path, attr='gradient', soft=Output, f_start=name, f_end='.out')
        va_corr.gradient['exc_freq'] = np.tile(355.9, len(va_corr.gradient))
        delta = gen_delta(delta_type=2, freq=self.naproxen_freq.frequency.copy())
        va_corr.vroa(uni=self.naproxen_freq, delta=delta['delta'].values)
        scatter_data = np.array([[ 1.16119994e+03,  4.90000000e+01,  3.51909145e+05,
                                  -3.94184317e+04,  3.02252860e+04,  3.25218881e+01,
                                   2.80234471e+01,  3.55900000e+02],
                                 [ 1.17230225e+03,  5.00000000e+01,  2.49856690e+05,
                                  -4.87581709e+03,  2.79074907e+04,  2.38302161e+01,
                                   2.41691134e+01,  3.55900000e+02],
                                 [ 1.22141899e+03,  5.10000000e+01,  5.38221435e+05,
                                   2.80423047e+03,  5.19618525e+04,  5.17589931e+01,
                                   4.59792091e+01,  3.55900000e+02]])
        raman_data = np.array([[1.16119994e+03, 4.90000000e+01, 1.58325495e+02, 2.28935485e+03,
                                1.01757944e+05, 3.55900000e+02],
                               [1.17230225e+03, 5.00000000e+01, 1.33889082e+02, 1.04769569e+03,
                                5.76262968e+04, 3.55900000e+02],
                               [1.22141899e+03, 5.10000000e+01, 2.30415117e+02, 2.53491851e+03,
                                1.22592114e+05, 3.55900000e+02]])
        scatter_data = scatter_data.T
        raman_data = raman_data.T

        # test all columns of the respective dataframe to get a better sense of what is broken
        self.assertTrue(np.allclose(va_corr.scatter['freq'].values,           scatter_data[0]))
        self.assertTrue(np.allclose(va_corr.scatter['freqdx'].values,         scatter_data[1]))
        self.assertTrue(np.allclose(va_corr.scatter['beta_g*1e6'].values,     scatter_data[2]))
        self.assertTrue(np.allclose(va_corr.scatter['beta_A*1e6'].values,     scatter_data[3]))
        self.assertTrue(np.allclose(va_corr.scatter['alpha_g*1e6'].values,    scatter_data[4]))
        self.assertTrue(np.allclose(va_corr.scatter['backscatter'].values,    scatter_data[5]))
        self.assertTrue(np.allclose(va_corr.scatter['forwardscatter'].values, scatter_data[6]))
        self.assertTrue(np.allclose(va_corr.scatter['exc_freq'].values,       scatter_data[7]))

        self.assertTrue(np.allclose(va_corr.raman['freq'].values,          raman_data[0]))
        self.assertTrue(np.allclose(va_corr.raman['freqdx'].values,        raman_data[1]))
        self.assertTrue(np.allclose(va_corr.raman['alpha_squared'].values, raman_data[2]))
        self.assertTrue(np.allclose(va_corr.raman['beta_alpha'].values,    raman_data[3]))
        self.assertTrue(np.allclose(va_corr.raman['raman_int'].values,     raman_data[4]))
        self.assertTrue(np.allclose(va_corr.raman['exc_freq'].values,      raman_data[5]))

