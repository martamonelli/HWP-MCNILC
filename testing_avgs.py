'''
Here I compare the maps obtained by beam-smoothing first and then applying the HWP,
with the maps obtained by applying the HWP first and then beam-smoothing.
'''

import numpy as np
import healpy as hp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import camb
import pysm3
import pysm3.units as u

import os
import time

plt.rcParams.update({
    "font.size":10.95,
    "text.usetex":True,
    "font.family":"serif", 
    "font.serif":"cm"
    })

start = time.time()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# main input parameters
HWP = 'ideal' 		#HWP type, can be 'ideal' or 'realistic'
dust_model = 'd0'	#PySM model for dust
sync_model = 's0'	#PySM model for synchrotron
NSIDE = 64		#nside
nreal = 100		#number of realizations
r = 0.00461 		#tensor-to-scalar ratio

skymodel = dust_model + sync_model

np.random.seed(0) #FIXME: eventually remove

if not os.path.exists(skymodel+'_'+str(NSIDE)+'_'+HWP):
    os.mkdir(skymodel+'_'+str(NSIDE)+'_'+HWP)
    os.mkdir(skymodel+'_'+str(NSIDE)+'_'+HWP+'/noise-only')
    os.mkdir(skymodel+'_'+str(NSIDE)+'_'+HWP+'/FG-only')
    os.mkdir(skymodel+'_'+str(NSIDE)+'_'+HWP+'/coadded')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# defining the ideal Mueller matrix elements values:
default = np.zeros((4,4))
default[0,0] += 1
default[1,1] += 1
default[2,2] += -1
default[3,3] += -1

# loading Mueller matrix elements for the three telescopes:
for telescope in ['LFT','MFT','HFT']:
    f = np.load('input/'+telescope+'_mueller.npz')
    freqs = f['freq']*1e9               # frequency [Hz]
    M = f['m'].real                     # mueller matrix

    M_telescope = np.empty((len(M[0,0,:]),4,4))

    for i in np.arange(4):
        for j in np.arange(4):
            M_telescope[:,i,j] = M[i,j,:]
            
    exec(f'freqs_{telescope} = freqs')  
    exec(f'M_{telescope} = M_telescope')

if HWP == 'ideal':
    for i in np.arange(len(M_LFT[:])):
        M_LFT[i] = np.diag([1,1,-1,-1])
    for i in np.arange(len(M_MFT[:])):
        M_MFT[i] = np.diag([1,1,-1,-1])
    for i in np.arange(len(M_HFT[:])):
        M_HFT[i] = np.diag([1,1,-1,-1])
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# healpy-related stuff
NPIX = 12*NSIDE**2
lmax = 2*NSIDE
ell = np.arange(2,lmax+1)
C2D = ell*(ell+1)/(2*np.pi)
D2C = np.append([1,1],1/C2D)

# prepare CMB input angular power spectra
# parameter file from https://github.com/cmbant/CAMB/blob/master/inifiles/planck_2018.ini
# (only change: I've set get_tensor_cls = T)
params_path = 'input/planck_2018.ini' 
pars = camb.read_ini(params_path)

results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

CAMB = powers['lensed_scalar']
Dls = np.array([CAMB[:,0], CAMB[:,1], CAMB[:,2], CAMB[:,3]])
Cls_scalar = Dls[:,:lmax+1]*D2C

CAMB = powers['tensor']
Dls = np.array([CAMB[:,0], CAMB[:,1], CAMB[:,2], CAMB[:,3]])
Cls_tensor = Dls[:,:lmax+1]*D2C
  
Cls_CMB = Cls_scalar + r*Cls_tensor              

# prepare PySM skies for dust and synchrotron
dust_sky = pysm3.Sky(nside=NSIDE, preset_strings=[dust_model])
sync_sky = pysm3.Sky(nside=NSIDE, preset_strings=[sync_model])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# introduce instrument specifics (PTEP paper):
chan_dicts = np.array([{'telescope':'LFT', 'nu':40. , 'delta':12. , 'fwhm':70.5 , 'sensitivity':37.42, 'sigma_alpha':49.8, 'name':'L040' },
                       {'telescope':'LFT', 'nu':50. , 'delta':15. , 'fwhm':58.5 , 'sensitivity':33.46, 'sigma_alpha':39.8, 'name':'L050' },
                       {'telescope':'LFT', 'nu':60. , 'delta':14. , 'fwhm':51.1 , 'sensitivity':21.31, 'sigma_alpha':16. , 'name':'L060' },
                       {'telescope':'LFT', 'nu':68. , 'delta':16. , 'fwhm':41.6 , 'sensitivity':19.91, 'sigma_alpha':1.09, 'name':'L068a'},
                       {'telescope':'LFT', 'nu':68. , 'delta':16. , 'fwhm':47.1 , 'sensitivity':31.77, 'sigma_alpha':35.9, 'name':'L068b'},
                       {'telescope':'LFT', 'nu':78. , 'delta':18. , 'fwhm':36.9 , 'sensitivity':15.55, 'sigma_alpha':8.6 , 'name':'L078a'},
                       {'telescope':'LFT', 'nu':78. , 'delta':18. , 'fwhm':43.8 , 'sensitivity':19.13, 'sigma_alpha':13.0, 'name':'L078b'},
                       {'telescope':'LFT', 'nu':89. , 'delta':20. , 'fwhm':33.0 , 'sensitivity':12.28, 'sigma_alpha':5.4 , 'name':'L089a'},
                       {'telescope':'LFT', 'nu':89. , 'delta':20. , 'fwhm':41.5 , 'sensitivity':28.77, 'sigma_alpha':29.4, 'name':'L089b'},
                       {'telescope':'LFT', 'nu':100., 'delta':23. , 'fwhm':30.2 , 'sensitivity':10.34, 'sigma_alpha':3.8 , 'name':'L100' },
                       {'telescope':'MFT', 'nu':100., 'delta':23. , 'fwhm':37.8 , 'sensitivity':8.48 , 'sigma_alpha':2.6 , 'name':'M100' },
                       {'telescope':'LFT', 'nu':119., 'delta':36. , 'fwhm':26.3 , 'sensitivity':7.69 , 'sigma_alpha':2.1 , 'name':'L119' },
                       {'telescope':'MFT', 'nu':119., 'delta':36. , 'fwhm':33.6 , 'sensitivity':5.70 , 'sigma_alpha':1.2 , 'name':'M119' },
                       {'telescope':'LFT', 'nu':140., 'delta':42. , 'fwhm':23.7 , 'sensitivity':7.25 , 'sigma_alpha':1.8 , 'name':'L140' },
                       {'telescope':'MFT', 'nu':140., 'delta':42. , 'fwhm':30.8 , 'sensitivity':6.38 , 'sigma_alpha':1.5 , 'name':'M140' },
                       {'telescope':'MFT', 'nu':166., 'delta':50. , 'fwhm':28.9 , 'sensitivity':5.57 , 'sigma_alpha':1.1 , 'name':'M166' },
                       {'telescope':'MFT', 'nu':195., 'delta':59. , 'fwhm':28.0 , 'sensitivity':7.05 , 'sigma_alpha':1.8 , 'name':'M195' },
                       {'telescope':'HFT', 'nu':195., 'delta':59. , 'fwhm':28.6 , 'sensitivity':10.50, 'sigma_alpha':3.9 , 'name':'H195' },
                       {'telescope':'HFT', 'nu':235., 'delta':71. , 'fwhm':24.7 , 'sensitivity':10.79, 'sigma_alpha':4.1 , 'name':'H235' },
                       {'telescope':'HFT', 'nu':280., 'delta':84. , 'fwhm':22.5 , 'sensitivity':13.80, 'sigma_alpha':6.8 , 'name':'H280' },
                       {'telescope':'HFT', 'nu':337., 'delta':101., 'fwhm':20.9 , 'sensitivity':21.95, 'sigma_alpha':17.1, 'name':'H337' },
                       {'telescope':'HFT', 'nu':402., 'delta':92. , 'fwhm':17.9 , 'sensitivity':47.45, 'sigma_alpha':80.0, 'name':'H402' },
                       ])
                       
# counting how many objects are in data_files
nkey = len(chan_dicts)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# producing maps affected by the HWP, channel by channel.
for i in np.arange(nkey):
    name = chan_dicts[i]['name']
    telescope = chan_dicts[i]['telescope']
    fwhm_radians = np.radians(chan_dicts[i]['fwhm']/60)
    
    if telescope == 'LFT':
        freqs = freqs_LFT 
        muellers = M_LFT
        c_string = 'crimson'
    
    if telescope == 'MFT':
        freqs = freqs_MFT 
        muellers = M_MFT
        c_string = 'purple'
        
    if telescope == 'HFT':
        freqs = freqs_HFT 
        muellers = M_HFT
        c_string = 'mediumslateblue'

    bandcenter = chan_dicts[i]['nu']*1e9   #[Hz]
    bandwidth = chan_dicts[i]['delta']*1e9 #[Hz]
    
    fmin = bandcenter - bandwidth/2
    fmax = bandcenter + bandwidth/2

    bandidx = np.where((freqs >= fmin) & (freqs <= fmax))
    
    freqs_channel = freqs[bandidx]
    fmin_ch = freqs_channel[:-1]
    fmax_ch = freqs_channel[1:]
    
    dust_maps = np.empty((len(freqs_channel),3,NPIX))
    sync_maps = np.empty((len(freqs_channel),3,NPIX))
    
    for j in np.arange(len(freqs_channel)):
        freq = freqs_channel[j]
        dust_maps[j] = dust_sky.get_emission(freq*u.Hz).to(getattr(u,'uK_CMB'),equivalencies=u.cmb_equivalencies(freq*u.Hz))
        sync_maps[j] = sync_sky.get_emission(freq*u.Hz).to(getattr(u,'uK_CMB'),equivalencies=u.cmb_equivalencies(freq*u.Hz))
        
    mIIs = muellers[bandidx,0,0][0]
    mQQs = muellers[bandidx,1,1][0]
    mQUs = muellers[bandidx,1,2][0]
    mUQs = muellers[bandidx,2,1][0]
    mUUs = muellers[bandidx,2,2][0]  
    
    rho = (mQQs-mUUs)/2
    eta = (mQUs+mUQs)/2
    
    # apply HWP to dust maps
    I_dust_ch = np.einsum('ij,i->j',dust_maps[:-1,0,:],mIIs[:-1]) + np.einsum('ij,i->j',dust_maps[1:,0,:],mIIs[1:])
    Q_dust_ch = np.einsum('ij,i->j',dust_maps[:-1,1,:],rho[:-1]) + np.einsum('ij,i->j',dust_maps[1:,1,:],rho[1:]) \
                + np.einsum('ij,i->j',dust_maps[:-1,2,:],eta[:-1]) + np.einsum('ij,i->j',dust_maps[1:,2,:],eta[1:])
    U_dust_ch = np.einsum('ij,i->j',dust_maps[:-1,2,:],rho[:-1]) + np.einsum('ij,i->j',dust_maps[1:,2,:],rho[1:]) \
                - np.einsum('ij,i->j',dust_maps[:-1,1,:],eta[:-1]) - np.einsum('ij,i->j',dust_maps[1:,1,:],eta[1:])
    maps_dust_ch = np.array([I_dust_ch,Q_dust_ch,U_dust_ch])/(2*(len(freqs_channel)-1))
    del(I_dust_ch)
    del(Q_dust_ch)
    del(U_dust_ch)
    
    # apply HWP to synchrotron maps    
    I_sync_ch = np.einsum('ij,i->j',sync_maps[:-1,0,:],mIIs[:-1]) + np.einsum('ij,i->j',sync_maps[1:,0,:],mIIs[1:])
    Q_sync_ch = np.einsum('ij,i->j',sync_maps[:-1,1,:],rho[:-1]) + np.einsum('ij,i->j',sync_maps[1:,1,:],rho[1:]) \
                + np.einsum('ij,i->j',sync_maps[:-1,2,:],eta[:-1]) + np.einsum('ij,i->j',sync_maps[1:,2,:],eta[1:])
    U_sync_ch = np.einsum('ij,i->j',sync_maps[:-1,2,:],rho[:-1]) + np.einsum('ij,i->j',sync_maps[1:,2,:],rho[1:]) \
                - np.einsum('ij,i->j',sync_maps[:-1,1,:],eta[:-1]) - np.einsum('ij,i->j',sync_maps[1:,1,:],eta[1:])
    maps_sync_ch = np.array([I_sync_ch,Q_sync_ch,U_sync_ch])/(2*(len(freqs_channel)-1))
    del(I_sync_ch)
    del(Q_sync_ch)
    del(U_sync_ch)
    
    # apply beam smoothing to FG-only maps
    maps_FG_ch_12 = hp.smoothing(maps_dust_ch + maps_sync_ch, fwhm=fwhm_radians, lmax=lmax)  
    
    ###################
    
    dust_maps = np.empty((len(freqs_channel),3,NPIX))
    sync_maps = np.empty((len(freqs_channel),3,NPIX))
    
    for j in np.arange(len(freqs_channel)):
        freq = freqs_channel[j]
        dust_maps[j] = hp.smoothing(dust_sky.get_emission(freq*u.Hz).to(getattr(u,'uK_CMB'),equivalencies=u.cmb_equivalencies(freq*u.Hz)), fwhm=fwhm_radians, lmax=lmax)  
        sync_maps[j] = hp.smoothing(sync_sky.get_emission(freq*u.Hz).to(getattr(u,'uK_CMB'),equivalencies=u.cmb_equivalencies(freq*u.Hz)), fwhm=fwhm_radians, lmax=lmax)  
    
    # apply HWP to dust maps
    I_dust_ch = np.einsum('ij,i->j',dust_maps[:-1,0,:],mIIs[:-1]) + np.einsum('ij,i->j',dust_maps[1:,0,:],mIIs[1:])
    Q_dust_ch = np.einsum('ij,i->j',dust_maps[:-1,1,:],rho[:-1]) + np.einsum('ij,i->j',dust_maps[1:,1,:],rho[1:]) \
                + np.einsum('ij,i->j',dust_maps[:-1,2,:],eta[:-1]) + np.einsum('ij,i->j',dust_maps[1:,2,:],eta[1:])
    U_dust_ch = np.einsum('ij,i->j',dust_maps[:-1,2,:],rho[:-1]) + np.einsum('ij,i->j',dust_maps[1:,2,:],rho[1:]) \
                - np.einsum('ij,i->j',dust_maps[:-1,1,:],eta[:-1]) - np.einsum('ij,i->j',dust_maps[1:,1,:],eta[1:])
    maps_dust_ch = np.array([I_dust_ch,Q_dust_ch,U_dust_ch])/(2*(len(freqs_channel)-1))
    del(I_dust_ch)
    del(Q_dust_ch)
    del(U_dust_ch)
    
    # apply HWP to synchrotron maps    
    I_sync_ch = np.einsum('ij,i->j',sync_maps[:-1,0,:],mIIs[:-1]) + np.einsum('ij,i->j',sync_maps[1:,0,:],mIIs[1:])
    Q_sync_ch = np.einsum('ij,i->j',sync_maps[:-1,1,:],rho[:-1]) + np.einsum('ij,i->j',sync_maps[1:,1,:],rho[1:]) \
                + np.einsum('ij,i->j',sync_maps[:-1,2,:],eta[:-1]) + np.einsum('ij,i->j',sync_maps[1:,2,:],eta[1:])
    U_sync_ch = np.einsum('ij,i->j',sync_maps[:-1,2,:],rho[:-1]) + np.einsum('ij,i->j',sync_maps[1:,2,:],rho[1:]) \
                - np.einsum('ij,i->j',sync_maps[:-1,1,:],eta[:-1]) - np.einsum('ij,i->j',sync_maps[1:,1,:],eta[1:])
    maps_sync_ch = np.array([I_sync_ch,Q_sync_ch,U_sync_ch])/(2*(len(freqs_channel)-1))
    del(I_sync_ch)
    del(Q_sync_ch)
    del(U_sync_ch)
    
    maps_FG_ch_21 = maps_dust_ch + maps_sync_ch
    
    diff = maps_FG_ch_21 - maps_FG_ch_12
    
    print('T diff between '+str(min(diff[0]))+' and '+str(max(diff[0])))
    print('Q diff between '+str(min(diff[1]))+' and '+str(max(diff[1])))
    print('U diff between '+str(min(diff[2]))+' and '+str(max(diff[2])))
       
    print('channel '+name+' done.')

hp.mollview(maps_FG_ch_21[0])
hp.mollview(maps_FG_ch_21[1])
hp.mollview(maps_FG_ch_21[2])
plt.show()
