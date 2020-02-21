import glob
from astropy.io import fits
import numpy as np

#file_11 = glob.glob('random_cubes/*11*fits')
n_spec = 20000 #len(file_11)

f_test='random_cubes/random_cube_NH3_11_00000.fits'
hd11 = fits.getheader( f_test)
hd22 = fits.getheader( f_test.replace('NH3_11','NH3_22'))
cube11 = np.zeros( (hd11['NAXIS3'], 1, n_spec))
cube22 = np.zeros( (hd22['NAXIS3'], 1, n_spec))
rms11 = np.zeros( (1, n_spec))
rms22 = np.zeros( (1, n_spec))
params = np.zeros( ( 12, 1, n_spec))
n_comp = np.zeros( ( 1, n_spec))
T_max = np.zeros( ( 2, 1, n_spec))

hd11['NAXIS1'] = n_spec
hd22['NAXIS1'] = n_spec
hd11['NAXIS2'] = 1
hd22['NAXIS2'] = 1

for i in range(n_spec):
    file_11_i='random_cubes/random_cube_NH3_11_{0:05d}.fits'.format(i)
    file_22_i='random_cubes/random_cube_NH3_22_{0:05d}.fits'.format(i)
    #
    data11_i, hd11_i = fits.getdata( file_11_i, header=True)
    data22_i, hd22_i = fits.getdata( file_22_i, header=True)
    cube11[:,0,i] = data11_i[:,1,1]
    cube22[:,0,i] = data22_i[:,1,1]
    rms11[0,i] = hd11_i['RMS']
    rms22[0,i] = hd22_i['RMS']
    n_comp[0,i] = hd11_i['NCOMP']
    #
    T_max[0,0,i] = hd11_i['TMAX-1']
    T_max[1,0,i] = hd11_i['TMAX-2']
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

fits.writeto('random_cubes/combined_NH3_11_n{0}.fits'.format(n_spec), cube11, hd11, overwrite=True)
fits.writeto('random_cubes/combined_NH3_22_n{0}.fits'.format(n_spec), cube22, hd22, overwrite=True)

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

fits.writeto('random_cubes/combined_NH3_11_rms_n{0}.fits'.format(n_spec), rms11, hd11_2d, overwrite=True)
fits.writeto('random_cubes/combined_NH3_22_rms_n{0}.fits'.format(n_spec), rms22, hd22_2d, overwrite=True)

fits.writeto('random_cubes/combined_NH3_N_par_n{0}.fits'.format(n_spec), n_comp, hd11_2d, overwrite=True)
hd11_2d['NAXIS'] = 3
hd11_2d['NAXIS3'] = 12
fits.writeto('random_cubes/combined_NH3_params_n{0}.fits'.format(n_spec), params, hd11_2d, overwrite=True)

hd11_2d['NAXIS3'] = 2
fits.writeto('random_cubes/combined_NH3_Tmaxs_n{0}.fits'.format(n_spec), T_max, hd11_2d, overwrite=True)

