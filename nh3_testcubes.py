import pyspeckit
import pyspeckit.spectrum.models.ammonia as ammonia
import pyspeckit.spectrum.models.ammonia_constants as nh3con
from pyspeckit.spectrum.units import SpectroscopicAxis as spaxis
from string import ascii_lowercase
import os
import sys
import numpy as np
import astropy.units as u
from astropy.io import fits
from spectral_cube import SpectralCube
from astropy.utils.console import ProgressBar
from astropy import log
log.setLevel('ERROR')



def generate_cubes(nCubes=100, nBorder=1, noise_rms=0.1, output_dir='random_cubes', random_seed=None,
                   linenames=['oneone', 'twotwo']):

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

    nComps, Temp, Width, Voff, logN = generate_parameters(nCubes, random_seed)

    gradX, gradY = generate_vGrad(nCubes, random_seed)

    cubes = []

    for xarr, lineID in zip(xarrList, lineIDList):
        # generate cubes for each line specified
        cubeList = []
        print('----------- generating {0} lines ------------'.format(lineID))
        for i in ProgressBar(range(nCubes)):
            results = make_cube(nComps[i], nBorder, i, xarr,
                                Temp, Width, Voff, logN, gradX, gradY, noise_rms)
            write_fits_cube(results['cube'], nCubes, nComps, i,
                            logN, Voff, Width, Temp, noise_rms,
                            results['Tmax'], results['Tmax_a'],
                            results['Tmax_b'], lineID,
                            output_dir=output_dir)
            cubeList.append(results['cube'])
        cubes.append(cubeList)

    return cubes



def generate_vGrad(nCubes, random_seed=None):
    # generate random velocity gradient in the X & Y directions to be apply to the cube model
    if random_seed:
        np.random.seed(random_seed)
    scale = np.array([[0.2, 0.1, 0.5, 0.01]])
    gradX1 = np.random.randn(nCubes, 4) * scale
    gradY1 = np.random.randn(nCubes, 4) * scale
    gradX2 = np.random.randn(nCubes, 4) * scale
    gradY2 = np.random.randn(nCubes, 4) * scale

    gradX = np.array([gradX1, gradX2])
    gradY = np.array([gradY1, gradY2])
    return gradX, gradY



def generate_parameters(nCubes, random_seed=None, fix_vlsr=True):
    # generate random parameters within a pre-defined distributions, for a two velocity slab model
    if random_seed:
        np.random.seed(random_seed)

    nComps = np.random.choice([1, 2], nCubes)

    Temp1 = 8 + np.random.rand(nCubes) * 17
    Temp2 = 8 + np.random.rand(nCubes) * 17

    if fix_vlsr:
        Voff1 = np.zeros(nCubes)
    else:
        Voff1 = np.random.rand(nCubes) * 5 - 2.5

    Voff2 = Voff1 + np.random.rand(nCubes) * 5 - 2.5

    logN1 = 13 + 1.5 * np.random.rand(nCubes)
    logN2 = 13 + 1.5 * np.random.rand(nCubes)

    Width1NT = 0.1 * np.exp(1.5 * np.random.randn(nCubes))
    Width2NT = 0.1 * np.exp(1.5 * np.random.randn(nCubes))

    Width1 = np.sqrt(Width1NT + 0.08 ** 2)
    Width2 = np.sqrt(Width2NT + 0.08 ** 2)

    Temp = np.array([Temp1, Temp2])
    Width = np.array([Width1, Width2])
    Voff = np.array([Voff1, Voff2])
    logN = np.array([logN1, logN2])

    return nComps, Temp, Width, Voff, logN



def make_cube(nComps, nBorder, i, xarr, Temp, Width, Voff, logN, gradX, gradY, noise_rms):
    # the length of Temp, Width, Voff, logN, gradX, and gradY should match the number of components
    xmat, ymat = np.indices((2 * nBorder + 1, 2 * nBorder + 1))
    cube = np.zeros((xarr.shape[0], 2 * nBorder + 1, 2 * nBorder + 1))

    results = {}
    results['Tmax_a'], results['Tmax_b'] = (0,) * 2

    for xx, yy in zip(xmat.flatten(), ymat.flatten()):

        spec = np.zeros(cube.shape[0])

        for j in range(nComps):
            # define parameters
            T = Temp[j][i] * (1 + gradX[j][i, 0] * (xx - 1) + gradY[j][i, 0] * (yy - 1)) + 5
            if T < 2.74:
                T = 2.74
            W = np.abs(Width[j][i] * (1 + gradX[j][i, 1] * (xx - 1) + gradY[j][i, 1] * (yy - 1)))
            V = Voff[j][i] + (gradX[j][i, 2] * (xx - 1) + gradY[j][i, 2] * (yy - 1))
            N = logN[j][i] * (1 + gradX[j][i, 3] * (xx - 1) + gradY[j][i, 3] * (yy - 1))

            # generate spectrum
            spec_j = ammonia.cold_ammonia(xarr, T, ntot=N, width=W, xoff_v=V)

            if (xx == nBorder) and (yy == nBorder):
                Tmaxj = np.max(spec_j)
                results['Tmax_{}'.format(ascii_lowercase[j])] = Tmaxj

            # add each component to the total spectrum
            spec = spec + spec_j

        cube[:, yy, xx] = spec
        if (xx == nBorder) and (yy == nBorder):
            Tmax = np.max(spec)
            results['Tmax'] = Tmax

    cube += np.random.randn(*cube.shape) * noise_rms
    results['cube'] = cube
    return results



def write_fits_cube(cube, nCubes, nComps, i, logN, Voff, Width, Temp, noise_rms,
                    Tmax, Tmax_a, Tmax_b, lineID='11', output_dir='random_cubes'):
    """
    This places nCubes random cubes into the specified output directory
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    logN1, logN2 = logN[0], logN[1]
    Voff1, Voff2 = Voff[0], Voff[1]
    Width1, Width2 = Width[0], Width[1]
    Temp1, Temp2 = Temp[0], Temp[1]

    nDigits = int(np.ceil(np.log10(nCubes)))

    hdrkwds = {'BUNIT': 'K',
               'INSTRUME': 'KFPA    ',
               'BMAJ': 0.008554169991270138,
               'BMIN': 0.008554169991270138,
               'TELESCOP': 'GBT',
               'WCSAXES': 3,
               'CRPIX1': 2,
               'CRPIX2': 2,
               'CRPIX3': 501,
               'CDELT1': -0.008554169991270138,
               'CDELT2': 0.008554169991270138,
               'CDELT3': 5720.0,
               'CUNIT1': 'deg',
               'CUNIT2': 'deg',
               'CUNIT3': 'Hz',
               'CTYPE1': 'RA---TAN',
               'CTYPE2': 'DEC--TAN',
               'CTYPE3': 'FREQ',
               'CRVAL1': 0.0,
               'CRVAL2': 0.0,
               'LONPOLE': 180.0,
               'LATPOLE': 0.0,
               'EQUINOX': 2000.0,
               'SPECSYS': 'LSRK',
               'RADESYS': 'FK5',
               'SSYSOBS': 'TOPOCENT'}
    truekwds = ['NCOMP', 'LOGN1', 'LOGN2', 'VLSR1', 'VLSR2',
                'SIG1', 'SIG2', 'TKIN1', 'TKIN2']


    hdu = fits.PrimaryHDU(cube)
    for kk in hdrkwds:
        hdu.header[kk] = hdrkwds[kk]
        for kk, vv in zip(truekwds, [nComps[i], logN1[i], logN2[i],
                                     Voff1[i], Voff2[i], Width1[i], Width2[i],
                                     Temp1[i], Temp2[i]]):
            hdu.header[kk] = vv
    hdu.header['TMAX'] = Tmax
    hdu.header['TMAX-1'] = Tmax_a
    hdu.header['TMAX-2'] = Tmax_b
    hdu.header['RMS'] = noise_rms
    hdu.header['CRVAL3'] = 23694495500.0
    hdu.header['RESTFRQ'] = 23694495500.0
    hdu.writeto(output_dir + '/random_cube_NH3_{0}_'.format(lineID)
                  + '{0}'.format(i).zfill(nDigits)
                  + '.fits',
                  overwrite=True)



if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) > 1:
        generate_cubes(nCubes=int(sys.argv[1]))
    else:
        generate_cubes()
