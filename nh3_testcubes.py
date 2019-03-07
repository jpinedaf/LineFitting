import pyspeckit.spectrum.models.ammonia as ammonia
import pyspeckit.spectrum.models.ammonia_constants as nh3con
from pyspeckit.spectrum.units import SpectroscopicAxis as spaxis
from string import ascii_lowercase
import os
import sys
import numpy as np
from astropy.io import fits
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

    # generate random parameters for nCubes
    nComps, Temp, Width, Voff, logN = generate_parameters(nCubes, random_seed)
    gradX, gradY = generate_gradients(nCubes, random_seed)

    cubes = []

    for xarr, linename in zip(xarrList, linenames):
        # generate cubes for each line specified
        cubeList = []
        print('----------- generating {0} lines ------------'.format(linename))
        for i in ProgressBar(range(nCubes)):
            cube_i = make_and_write(nCubes, nComps[i], i, nBorder, xarr, Temp[i], Width[i], Voff[i], logN[i], gradX[i], gradY[i]
                           , noise_rms, linename, output_dir)

            cubeList.append(cube_i)
        cubes.append(cubeList)

    return cubes



def make_and_write(nCubes, nComp, i, nBorder, xarr, T, W, V, N, grdX, grdY, noise_rms, linename, output_dir):
    # wrapper for make_cube() and write_fits_cube()

    results = make_cube(nComp, nBorder, xarr, T, W, V, N, grdX, grdY, noise_rms)

    write_fits_cube(results['cube'], nCubes, nComp, i, N, V, W, T, noise_rms,
                    results['Tmax'], results['Tmax_a'], results['Tmax_b'], linename,
                    output_dir)

    return results['cube']



def generate_gradients(nCubes, random_seed=None):
    # generate random gradient for temp, sigma, voff, and logN in the X & Y directions
    if random_seed:
        np.random.seed(random_seed)
    # scaling for the temp, sigma, voff, and logN parameters
    scale = np.array([[0.2, 0.1, 0.5, 0.01]])
    # 0.5 km/s/pix is about 39.7 km/s/pc at 260pc away at 10"/pix
    gradX1 = np.random.randn(nCubes, 4) * scale
    gradY1 = np.random.randn(nCubes, 4) * scale
    gradX2 = np.random.randn(nCubes, 4) * scale
    gradY2 = np.random.randn(nCubes, 4) * scale

    gradX = np.array([gradX1, gradX2])
    gradY = np.array([gradY1, gradY2])

    return gradX.swapaxes(0, 1), gradY.swapaxes(0, 1)



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

    Temp = np.array([Temp1, Temp2]).swapaxes(0, 1)
    Width = np.array([Width1, Width2]).swapaxes(0, 1)
    Voff = np.array([Voff1, Voff2]).swapaxes(0, 1)
    logN = np.array([logN1, logN2]).swapaxes(0, 1)

    return nComps, Temp, Width, Voff, logN



def make_cube(nComps, nBorder, xarr, Temp, Width, Voff, logN, gradX, gradY, noise_rms):
    # the length of Temp, Width, Voff, logN, gradX, and gradY should match the number of components
    xmat, ymat = np.indices((2 * nBorder + 1, 2 * nBorder + 1))
    cube = np.zeros((xarr.shape[0], 2 * nBorder + 1, 2 * nBorder + 1))

    results = {}
    results['Tmax_a'], results['Tmax_b'] = (0,) * 2

    for xx, yy in zip(xmat.flatten(), ymat.flatten()):

        spec = np.zeros(cube.shape[0])

        for j in range(nComps):
            # define parameters
            T = Temp[j] * (1 + gradX[j][0] * (xx - 1) + gradY[j][0] * (yy - 1)) + 5
            if T < 2.74:
                T = 2.74
            W = np.abs(Width[j] * (1 + gradX[j][1] * (xx - 1) + gradY[j][1] * (yy - 1)))
            V = Voff[j] + (gradX[j][2] * (xx - 1) + gradY[j][2] * (yy - 1))
            N = logN[j] * (1 + gradX[j][3] * (xx - 1) + gradY[j][3] * (yy - 1))

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

    # should we add noise independently on individual spectrum to avoid correlated noise across pixels?
    cube += np.random.randn(*cube.shape) * noise_rms
    results['cube'] = cube
    return results



def write_fits_cube(cube, nCubes, nComps, i, logN, Voff, Width, Temp, noise_rms,
                    Tmax, Tmax_a, Tmax_b, linename, output_dir='random_cubes'):
    """
    Function to write a test cube as a fits file
    Note: only currently compatible with nComp <= 2
    """
    if not os.path.isdir(output_dir):
        #  This places nCubes random cubes into the specified output directory
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
        for kk, vv in zip(truekwds, [nComps, logN1, logN2,
                                     Voff1, Voff2, Width1, Width2,
                                     Temp1, Temp2]):
            hdu.header[kk] = vv
    hdu.header['TMAX'] = Tmax
    hdu.header['TMAX-1'] = Tmax_a
    hdu.header['TMAX-2'] = Tmax_b
    hdu.header['RMS'] = noise_rms
    hdu.header['CRVAL3'] = nh3con.freq_dict[linename]
    hdu.header['RESTFRQ'] = nh3con.freq_dict[linename]

    # specify the ID fore each line to appear in  saved fits files
    if linename is 'oneone':
        lineID = '11'
    elif linename is 'twotwo':
        lineID = '22'
    else:
        # use line names at it is for lines above (3,3)
        lineID = linename

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
