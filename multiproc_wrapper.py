#=======================================================================================================================

# a multiprocessing wrapper for nh3_testcubes.py

#=======================================================================================================================
import os
import numpy as np
import pyspeckit.spectrum.models.ammonia_constants as nh3con
from pyspeckit.spectrum.units import SpectroscopicAxis as spaxis
from multiprocessing import Pool, cpu_count
import tqdm
import itertools
from itertools import repeat as rp

import nh3_testcubes as testcubes

def generate_cubes(nCubes=100, nBorder=1, noise_rms=0.1, output_dir='random_cubes', random_seed=None,
                   linenames=['oneone', 'twotwo'], n_cpu=None):

    if not os.path.isdir(output_dir):
        #  Make the directory first to avoid multiple processors attempting to make the directory at the same time
        os.mkdir(output_dir)

    if n_cpu is None:
        n_cpu = cpu_count() - 1
        print "number of cpu used: {}".format(n_cpu)

    pool = Pool(n_cpu)


    i = np.arange(nCubes)
    nComp, T, W, V, N = testcubes.generate_parameters(nCubes, random_seed)
    grdX, grdY = testcubes.generate_gradients(nCubes, random_seed)


    for linename in linenames:
        print('----------- generating {0} lines ------------'.format(linename))
        for j in tqdm.tqdm(pool.imap(f_star, itertools.izip(rp(nCubes), nComp, i, rp(nBorder), T, W, V, N, grdX, grdY,
                                        rp(noise_rms), rp(linename), rp(output_dir))), total=nCubes, mininterval=0.01):
            pass

def f(nCubes, nComp, i, nBorder, T, W, V, N, grdX, grdY, noise_rms, linename, output_dir):
    # wrapper function to generate and write the test cubes

    # generating SpectroscopicAxis objects here may be a little inefficient, but it's a work around to the pickling
    # error (i.e., PicklingError: Can't pickle <type 'function'>: attribute lookup __builtin__.function failed issue)
    xarr = generate_xarr(linename)
    testcubes.make_and_write(nCubes, nComp, i, nBorder, xarr, T, W, V, N, grdX, grdY, noise_rms, linename, output_dir)


def f_star(paras):
    """Convert `f([a,b,...])` to `f(a,b,...)` call."""
    return f(*paras)


def generate_xarr(linename):
    # generate SpectroscopicAxis objects
    xarr = spaxis((np.linspace(-500, 499, 1000) * 5.72e-6
                   + nh3con.freq_dict[linename] / 1e9),
                  unit='GHz',
                  refX=nh3con.freq_dict[linename] / 1e9,
                  velocity_convention='radio', refX_unit='GHz')
    return xarr



