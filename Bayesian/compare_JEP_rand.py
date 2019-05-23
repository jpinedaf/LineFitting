import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import analysis_tools as at


n_comp_in = fits.getdata('Bayesian/combined_NH3_N_par.fits')
Ks = fits.getdata('Bayesian/combined_NH3-Ks.fits')

Kcut = 5 
npeaks_map = np.full_like(Ks[0], np.nan)
npeaks_map[Ks[0] <= Kcut] = 0
npeaks_map[Ks[0] > Kcut] = 1
npeaks_map[(Ks[0] > Kcut) & (Ks[1] > Kcut)] = 2

fits.writeto( 'Bayesian/combined_N_par_out.fits', npeaks_map, 
              fits.getheader('Bayesian/combined_NH3_N_par.fits'), overwrite=True)

plt.ion()
plt.scatter(n_comp_in, Ks[0,:,:], color='blue', alpha=0.2)
plt.scatter(n_comp_in, Ks[1,:,:], color='red', alpha=0.2)

plt.scatter(n_comp_in, npeaks_map, color='red', alpha=0.2)

peak_map = np.zeros( (2,2))
peak_map[ 0, 0] = (npeaks_map[ n_comp_in == 1] == 1).sum()*(1.0 / (n_comp_in == 1).sum())
peak_map[ 0, 1] = (npeaks_map[ n_comp_in == 1] == 2).sum()*(1.0 / (n_comp_in == 1).sum())

peak_map[ 1, 0] = (npeaks_map[ n_comp_in == 2] == 1).sum()*(1.0 / (n_comp_in == 2).sum())
peak_map[ 1, 1] = (npeaks_map[ n_comp_in == 2] == 2).sum()*(1.0 / (n_comp_in == 2).sum())

at.plot_confusion_matrix(peak_mapi, ['1 comp', '2 comp'])
