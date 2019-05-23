import glob
from astropy.io import fits
import numpy as np

file_11 = glob.glob('random_cubes/*11*fits')
n_spec = len(file_11)

hd11 = fits.getheader( file_11[0])
hd22 = fits.getheader( file_11[0].replace('NH3_11','NH3_22'))
cube11 = np.zeros( (hd11['NAXIS3'], 1, n_spec))
cube22 = np.zeros( (hd22['NAXIS3'], 1, n_spec))
rms11 = np.zeros( (1, n_spec))
rms22 = np.zeros( (1, n_spec))
params = np.zeros( ( 12, 1, n_spec))
n_comp = np.zeros( ( 1, n_spec))

hd11['NAXIS1'] = n_spec
hd22['NAXIS1'] = n_spec
hd11['NAXIS2'] = 1
hd22['NAXIS2'] = 1

for i in range(n_spec):
    file_22_i=file_11[i].replace('NH3_11','NH3_22')
    data11_i, hd11_i = fits.getdata( file_11[i], header=True)
    data22_i, hd22_i = fits.getdata( file_22_i, header=True)
    cube11[:,0,i] = data11_i[:,1,1]
    cube22[:,0,i] = data22_i[:,1,1]
    rms11[0,i] = hd11_i['RMS']
    rms22[0,i] = hd22_i['RMS']
    n_comp[0,i] = hd11_i['NCOMP']
    #  Tk, Tex, log(N), sigma_v, v_lsr, f_ortho
    # Tk
    params[0+0,0,i] = hd11_i['TKIN1']
    params[0+6,0,i] = hd11_i['TKIN2']
    # Tex (LTE)
    params[1+0,0,i] = hd11_i['TKIN1']
    params[1+6,0,i] = hd11_i['TKIN2']
    # log(N)  
    params[2+0,0,i] = hd11_i['LOGN1']
    params[2+6,0,i] = hd11_i['LOGN2']
    # sigma_v
    params[3+0,0,i] = hd11_i['SIG1']
    params[3+6,0,i] = hd11_i['SIG2']
    # V_LSR
    params[4+0,0,i] = hd11_i['VLSR1']
    params[4+6,0,i] = hd11_i['VLSR2']

fits.writeto('random_cubes/combined_NH3_11.fits', cube11, hd11, overwrite=True)
fits.writeto('random_cubes/combined_NH3_22.fits', cube22, hd22, overwrite=True)

hd11_2d = hd11.copy()
hd22_2d = hd22.copy()
key_list=['NAXIS3', 'CRPIX3', 'CDELT3', 'CUNIT3', 'CTYPE3', 'CRVAL3', 
          'NCOMP', 'LOGN1', 'LOGN2', 'VLSR1', 'VLSR2', 'SIG1', 'SIG2', 
          'TKIN1', 'TKIN2', 'RMS']
for key_i in key_list:
    hd11_2d.remove(key_i)
    hd22_2d.remove(key_i)
hd11_2d['NAXIS'] = 2
hd22_2d['NAXIS'] = 2

fits.writeto('random_cubes/combined_NH3_11_rms.fits', rms11, hd11_2d, overwrite=True)
fits.writeto('random_cubes/combined_NH3_22_rms.fits', rms22, hd22_2d, overwrite=True)

fits.writeto('random_cubes/combined_NH3_N_par.fits', n_comp, hd11_2d, overwrite=True)
hd11_2d['NAXIS'] = 3
hd11_2d['NAXIS3'] = 12
fits.writeto('random_cubes/combined_NH3_params.fits', params, hd11_2d, overwrite=True)
