import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

n_comp_in = fits.getdata('combined_NH3_N_par.fits')
Ks = fits.getdata('combined_NH3-Ks.fits')

Kcut = 5 
npeaks_map = np.full_like(Ks[0], np.nan)
npeaks_map[Ks[0] <= Kcut] = 0
npeaks_map[Ks[0] > Kcut] = 1
npeaks_map[(Ks[0] > Kcut) & (Ks[1] > Kcut)] = 2

fits.writeto( 'combined_N_par_out.fits', npeaks_map, fits.getheader('combined_NH3_N_par.fits'), overwrite=True)

plt.ion()
plt.scatter(n_comp_in, Ks[0,:,:], color='blue', alpha=0.2)
plt.scatter(n_comp_in, Ks[1,:,:], color='red', alpha=0.2)

plt.scatter(n_comp_in, npeaks_map, color='red', alpha=0.2)
