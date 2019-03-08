#=======================================================================================================================

# a multiprocessing wrapper for nh3_testcubes.py

#=======================================================================================================================
import numpy as np
import pyspeckit.spectrum.models.ammonia_constants as nh3con
from pyspeckit.spectrum.units import SpectroscopicAxis as spaxis
from multiprocessing import Pool, cpu_count
import tqdm

import nh3_testcubes as testcubes

class TestSet:
    # store parameters generated for the test set
    def __init__(self, nCubes, nComps, nBorder, xarr, paras, grads, noise_rms, linename, output_dir):
        self.nCubes = nCubes
        self.nComps = nComps
        self.nBorder = nBorder
        self.xarr = xarr
        self.paras = paras
        self.grads = grads
        self.noise_rms = noise_rms
        self.linename = linename
        self.output_dir = output_dir


def make_n_write(i):
    # the wrapper function that will be multi-processed
    testcubes.make_and_write(tset.nCubes, tset.nComps[i], i, tset.nBorder, tset.xarr,
                             tset.paras['Temp'][i], tset.paras['Width'][i],
                             tset.paras['Voff'][i], tset.paras['logN'][i],
                             tset.grads['gradX'][i], tset.grads['gradY'][i],
                             tset.noise_rms, tset.linename, tset.output_dir)


def generate_cubes(nCubes=100, nBorder=1, noise_rms=0.1, output_dir='random_cubes', random_seed=None,
                   linenames=['oneone', 'twotwo'], n_cpu=None):

    global tset

    xarrList = []
    for linename in linenames:
        # generate spectral axis for each ammonia lines
        xarr = spaxis((np.linspace(-500, 499, 1000) * 5.72e-6
                       + nh3con.freq_dict[linename] / 1e9),
                      unit='GHz',
                      refX=nh3con.freq_dict[linename] / 1e9,
                      velocity_convention='radio', refX_unit='GHz')
        xarrList.append(xarr)

    # generate random parameters for nCubes
    nComps, Temp, Width, Voff, logN = testcubes.generate_parameters(nCubes, random_seed)
    gradX, gradY = testcubes.generate_gradients(nCubes, random_seed)
    paras = {'Temp':Temp, 'Width':Width, 'Voff':Voff, 'logN':logN}
    grads = {'gradX':gradX, 'gradY':gradY}

    if n_cpu is None:
        n_cpu = cpu_count() - 1
        print "number of cpu used: {}".format(n_cpu)

    for xarr, linename in zip(xarrList, linenames):
        tset = TestSet(nCubes, nComps, nBorder, xarr, paras, grads, noise_rms, linename, output_dir)
        print('----------- generating {0} lines ------------'.format(linename))
        pool = Pool(n_cpu)
        for i in tqdm.tqdm(pool.imap(make_n_write, range(nCubes)), total=nCubes, mininterval=0.01):
            pass
