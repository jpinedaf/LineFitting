#=======================================================================================================================

# an example file on how to build special test/training cubes using nh3_testcube.py

#=======================================================================================================================
import numpy as np
import pyspeckit.spectrum.models.ammonia_constants as nh3con
from pyspeckit.spectrum.units import SpectroscopicAxis as spaxis
from astropy.utils.console import ProgressBar
import sys

import nh3_testcubes as testcubes


def generate_cubes(nCubes=100, nBorder=1, noise_rms=0.1, output_dir='random_cubes', random_seed=None,
                   linenames=['oneone', 'twotwo'], remove_low_sep=True, noise_class=True):

    xarrList = []
    lineIDList = []

    for linename in linenames:
        # generate spectral axis for each ammonia lines
        xarr = spaxis((np.linspace(-500, 499, 1000) * 5.72e-6
                       + nh3con.freq_dict[linename] / 1e9),
                      unit='GHz',
                      refX=nh3con.freq_dict[linename] / 1e9,
                      velocity_convention='radio', refX_unit='GHz')
        xarrList.append(xarr)

        # specify the ID fore each line to appear in  saved fits files
        if linename is 'oneone':
            lineIDList.append('11')
        elif linename is 'twotwo':
            lineIDList.append('22')
        else:
            # use line names at it is for lines above (3,3)
            lineIDList.append(linename)

    # generate random parameters for nCubes
    nComps, Temp, Width, Voff, logN = testcubes.generate_parameters(nCubes, random_seed)
    gradX, gradY = testcubes.generate_gradients(nCubes, random_seed)

    if noise_class:
        # Creates a balanced training set with 1comp, noise, and 2comp classes
        nComps = np.concatenate((np.zeros(nCubes / 3).astype(int),
                                 np.ones(nCubes / 3).astype(int),
                                 np.ones(nCubes / 3 + nCubes%3).astype(int) + 1))

    if remove_low_sep:
        Voff = remove_low_vsep(Voff, Width)

    cubes = []

    for xarr, lineID in zip(xarrList, lineIDList):
        # generate cubes for each line specified
        cubeList = []
        print('----------- generating {0} lines ------------'.format(lineID))
        for i in ProgressBar(range(nCubes)):
            cube_i = testcubes.make_and_write(nCubes, nComps[i], i, nBorder, xarr, Temp[i], Width[i], Voff[i], logN[i], gradX[i], gradY[i]
                           , noise_rms, lineID, output_dir)

            cubeList.append(cube_i)
        cubes.append(cubeList)

    return cubes



def remove_low_vsep(Voff, Width):

    Voff = Voff.swapaxes(0, 1)
    Voff1, Voff2 = Voff[0], Voff[1]

    Width = Width.swapaxes(0, 1)
    Width1, Width2 = Width[0], Width[1]

    # Find where centroids are too close
    too_close = np.where(np.abs(Voff1 - Voff2) < np.max(np.column_stack((Width1, Width2)), axis=1))
    # Move the centroids farther apart by the length of largest line width
    min_Voff = np.min(np.column_stack((Voff2[too_close], Voff1[too_close])), axis=1)
    max_Voff = np.max(np.column_stack((Voff2[too_close], Voff1[too_close])), axis=1)
    Voff1[too_close] = min_Voff - np.max(np.column_stack((Width1[too_close], Width2[too_close])), axis=1) / 2.
    Voff2[too_close] = max_Voff + np.max(np.column_stack((Width1[too_close], Width2[too_close])), axis=1) / 2.
    Voff = np.array([Voff1, Voff2]).swapaxes(0, 1)

    return Voff


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) > 1:
        generate_cubes(nCubes=int(sys.argv[1]))
    else:
        generate_cubes()