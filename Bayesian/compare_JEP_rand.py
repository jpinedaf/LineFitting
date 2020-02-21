import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import analysis_tools as at

workDir='Bayesian'
cubeDir='random_cubes'
#dict_truepara = read_cubes(cubeDir=cubeDir, nCubes=nCubes)
#results = read_cubes(cubeDir=cubeDir, nCubes=1000)
tableName = "{}/Bayes_cube_test_results.txt".format(workDir)

n_comp_in = fits.getdata('{}/combined_NH3_N_par.fits'.format(workDir))
param_in = fits.getdata('{}/combined_NH3_params.fits'.format(workDir))
#Ks = fits.getdata('Bayesian/combined_NH3-Ks.fits')
Ks = fits.getdata('{}/combined_NH3_n1000-Ks.fits'.format(workDir))
Tmax = fits.getdata('{}/combined_NH3_Tmaxs.fits'.format(cubeDir))
rms = fits.getdata('{}/combined_NH3_11_rms.fits'.format(cubeDir))

Kcut = 5 
npeaks_map = np.full_like(Ks[0], np.nan)
npeaks_map[Ks[0] <= Kcut] = 0
npeaks_map[Ks[0] > Kcut] = 1
npeaks_map[(Ks[0] > Kcut) & (Ks[1] > Kcut)] = 2

par_1comp=fits.getdata('{}/combined_NH3_n1000-mle-x1.fits'.format(workDir))
par_2comp=fits.getdata('{}/combined_NH3_n1000-mle-x2.fits'.format(workDir))


#table = Table(data, names=names)
#table.write(tableName, format='ascii', overwrite=True)

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

#at.plot_confusion_matrix(peak_map, ['1 comp', '2 comp'])


#fit1 = fits.getdata('Bayesian/combined_NH3-mle-x1.fits')
#fit2 = fits.getdata('Bayesian/combined_NH3-mle-x2.fits')
combine_par = par_2comp
ss = (npeaks_map[0,:] == 1)
combine_par[ :, 0, ss] = 0.0
combine_par[ 0:12, 0, ss] = par_1comp[ :, 0, ss]


from astropy.table import Table

####eTEX1_FIT TEX2_FIT TAU2_FIT LOGN1 RMS LOGN2 NCOMP SIG1 SIG2 eSIG2_FIT VLSR1_FIT eSIG1_FIT TMAX-2 TMAX-1 eTAU1_FIT eTAU2_FIT eTEX2_FIT SIG2_FIT LN_K_21 VLSR2 VLSR1 eVLSR2_FIT VLSR2_FIT TKIN2 TKIN1 TEX1_FIT TAU1_FIT SIG1_FIT NCOMP_FIT eVLSR1_FIT RMS_FIT TMAX
#
#eTEX1_FIT TEX2_FIT TAU2_FIT LOGN1 LOGN2 SIG1 SIG2 eSIG2_FIT VLSR1_FIT eSIG1_FIT eTAU1_FIT eTAU2_FIT eTEX2_FIT SIG2_FIT LN_K_21 VLSR2 VLSR1 eVLSR2_FIT VLSR2_FIT TEX1_FIT TAU1_FIT SIG1_FIT eVLSR1_FIT RMS_FIT

t = Table([Tmax[0,0], Tmax[1,0], rms[0], n_comp_in[0], npeaks_map[0], Tmax[0,0],
          param_in[0,0], param_in[6,0], # TKin input parameters
          combine_par[0,0], combine_par[6,0], # First component with uncertainty
          combine_par[12,0], combine_par[18,0], # second component with uncertainty
          param_in[1,0], param_in[7,0], # Tex Input parameters
          combine_par[1,0], combine_par[7,0],
          combine_par[13,0], combine_par[19,0],
          param_in[4,0], param_in[10,0], # Vlsr Input parameters
          combine_par[4,0], combine_par[10,0],
          combine_par[16,0], combine_par[22,0],
          param_in[3,0], param_in[9,0], # Sigma_v Input parameters
          combine_par[3,0], combine_par[9,0],
          combine_par[15,0], combine_par[21,0]],
          names=('TMAX-1', 'TMAX-2', 'RMS', 'NCOMP', 'NCOMP_FIT', 'TMAX',
                 'TKIN1', 'TKIN2', 
                 'TKIN1_FIT', 'eTKIN1_FIT',
                 'TKIN2_FIT', 'eTKIN2_FIT', 
                 'TEX1', 'TEX2', 
                 'TEX1_FIT', 'eTEX1_FIT', 
                 'TEX2_FIT', 'eTEX2_FIT',
                 'VLSR1', 'VLSR2', 
                 'VLSR1_FIT', 'eVLSR1_FIT', 
                 'VLSR2_FIT', 'eVLSR2_FIT', 
                 'SIG1', 'SIG2', 
                 'SIG1_FIT', 'eSIG1_FIT',
                 'SIG2_FIT', 'eSIG2_FIT'))

t.write('{}/table_Bayesian.dat'.format(workDir), format='ascii', overwrite=True) 
#
    ## Tk
    #params[0+0,0,i] = hd11_i['TKIN1']
    #params[0+6,0,i] = hd11_i['TKIN2']
    ## Tex (LTE)
    ##params[1+0,0,i] = hd11_i['TKIN1']
    #params[1+6,0,i] = hd11_i['TKIN2']
    ## log(N)  
    #params[2+0,0,i] = hd11_i['LOGN1']
    #params[2+6,0,i] = hd11_i['LOGN2']
    ## sigma_v
    #params[3+0,0,i] = hd11_i['SIG1']
    #params[3+6,0,i] = hd11_i['SIG2']
    ## V_LSR
    #params[4+0,0,i] = hd11_i['VLSR1']
    #params[4+6,0,i] = hd11_i['VLSR2']
#
