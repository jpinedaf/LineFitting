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
#import h5py
log.setLevel('ERROR')

def generate_cubes(nCubes=100, nBorder=1, noise_rms=0.1, output_dir='random_cubes', fix_vlsr=True, random_seed=None,
                   remove_low_sep=False, noise_class=False):#, ml_output=False):
    """
    This places nCubes random cubes into the specified output directory
    """

    xarr11 = spaxis((np.linspace(-500, 499, 1000) * 5.72e-6
                     + nh3con.freq_dict['oneone'] / 1e9),
                    unit='GHz',
                    refX=nh3con.freq_dict['oneone'] / 1e9,
                    velocity_convention='radio', refX_unit='GHz')

    xarr22 = spaxis((np.linspace(-500, 499, 1000) * 5.72e-6
                     + nh3con.freq_dict['twotwo'] / 1e9), unit='GHz',
                    refX=nh3con.freq_dict['twotwo'] / 1e9,
                    velocity_convention='radio', refX_unit='GHz')

    # Create holders for ml_output
    #out_arr = []
    #out_y = []

    if random_seed:
        np.random.seed(random_seed)
    if noise_class:
        # Creates a balanced training set with 1comp, noise, and 2comp classes
        nComps = np.concatenate((np.ones(nCubes/3).astype(int), np.zeros(nCubes/3).astype(int), np.zeros(nCubes/3).astype(int)+2))
    else:
        nComps = np.random.choice([1, 2], nCubes)

    Temp1 = 8 + np.random.rand(nCubes) * 17
    Temp2 = 8 + np.random.rand(nCubes) * 17

    if fix_vlsr:
        Voff1 = np.zeros(nCubes)
    else:
        Voff1 = np.random.rand(nCubes) * 5 - 2.5

    Voff2 = Voff1 + np.random.rand(nCubes)* 5 - 2.5

    logN1 = 13 + 1.5 * np.random.rand(nCubes)
    logN2 = 13 + 1.5 * np.random.rand(nCubes)

    Width1NT = 0.1 * np.exp(1.5 * np.random.randn(nCubes))
    Width2NT = 0.1 * np.exp(1.5 * np.random.randn(nCubes))

    Width1 = np.sqrt(Width1NT + 0.08**2)
    Width2 = np.sqrt(Width2NT + 0.08**2)
    
    if remove_low_sep:
        # Find where centroids are too close
        too_close = np.where(np.abs(Voff1-Voff2)<np.max(np.column_stack((Width1, Width2)), axis=1))
        # Move the centroids farther apart by the length of largest line width 
        min_Voff = np.min(np.column_stack((Voff2[too_close],Voff1[too_close])), axis=1)
        max_Voff = np.max(np.column_stack((Voff2[too_close],Voff1[too_close])), axis=1)
        Voff1[too_close]=min_Voff-np.max(np.column_stack((Width1[too_close], Width2[too_close])), axis=1)/2.
        Voff2[too_close]=max_Voff+np.max(np.column_stack((Width1[too_close], Width2[too_close])), axis=1)/2.

    scale = np.array([[0.2, 0.1, 0.5, 0.01]])
    gradX1 = np.random.randn(nCubes, 4) * scale
    gradY1 = np.random.randn(nCubes, 4) * scale
    gradX2 = np.random.randn(nCubes, 4) * scale
    gradY2 = np.random.randn(nCubes, 4) * scale

    cubeList11 = []
    cubeList22 = []

    for i in ProgressBar(range(nCubes)):
        Temp = np.array([Temp1, Temp2])
        Width = np.array([Width1, Width2])
        Voff = np.array([Voff1, Voff2])
        logN = np.array([logN1, logN2])
        gradX = np.array([gradX1, gradX2])
        gradY = np.array([gradY1, gradY2])

        results11 = make_cube(nComps[i], nBorder, i, xarr11, Temp, Width, Voff, logN, gradX, gradY, noise_rms)
        results22 = make_cube(nComps[i], nBorder, i, xarr22, Temp, Width, Voff, logN, gradX, gradY, noise_rms)

        Tmax11, Tmax11a, Tmax11b  = results11['Tmax'], results11['Tmax_a'], results11['Tmax_b']
        cube11 = results11['cube']

        Tmax22, Tmax22a, Tmax22b  = results22['Tmax'], results22['Tmax_a'], results22['Tmax_b']
        cube22 = results22['cube']

        write_fits_cube(cube11, nCubes, nComps, i, logN1, logN2, Voff1, Voff2, Width1, Width2, Temp1, Temp2, noise_rms,
                    Tmax11, Tmax11a, Tmax11b, lineID='11', output_dir=output_dir)

        write_fits_cube(cube22, nCubes, nComps, i, logN1, logN2, Voff1, Voff2, Width1, Width2, Temp1, Temp2, noise_rms,
                    Tmax22, Tmax22a, Tmax22b, lineID='22', output_dir=output_dir)

        cubeList11.append(cube11)
        cubeList22.append(cube22)

    return cubeList11, cubeList22

    '''
        if ml_output:	
            # Grab central pixel and normalize
            loc11 = cube11[:,1,1].reshape(1000,1)
            loc11 = loc11/np.max(loc11)
            # Grab 3x3 average and normalize
            glob11 = np.mean(cube11.reshape(1000,9),axis=1)
            glob11 = glob11/np.max(glob11)
            z = np.column_stack((loc11,glob11))
            # Append to arrays
            out_arr.append(z)
            out_y.append(nComps[i])

    if ml_output:
        out_y1 = np.where(np.array(out_y)==1, 1, 0)
        out_y2 = np.where(np.array(out_y)==0, 1, 0)
        out_y3 = np.where(np.array(out_y)==2, 1, 0)
        with h5py.File('nh3_three_class.h5', 'w') as hf:
	          hf.create_dataset('data', data=np.array(out_arr))
	          hf.close()
        with h5py.File('labels_nh3_three_class.h5', 'w') as hf:
	          hf.create_dataset('data', data=np.column_stack((out_y1, out_y2, out_y3)))
	          hf.close()
    '''


def make_cube(nComps, nBorder, i, xarr, Temp, Width, Voff, logN, gradX, gradY, noise_rms):
    # the length of Temp, Width, Voff, logN, gradX, and gradY should match the number of components

    results = {}

    xmat, ymat = np.indices((2 * nBorder + 1, 2 * nBorder + 1))
    cube = np.zeros((xarr.shape[0], 2 * nBorder + 1, 2 * nBorder + 1))

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



def write_fits_cube(cube, nCubes, nComps, i, logN1, logN2, Voff1, Voff2, Width1, Width2, Temp1, Temp2, noise_rms,
                    Tmax, Tmax_a, Tmax_b, lineID='11', output_dir='random_cubes'):

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

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
