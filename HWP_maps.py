import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import pickle
import json

import pysm3
import pysm3.units as u

import healpy as hp

import time
start = time.time()

#Defining the ideal Mueller matrix elements values:

default = np.zeros((4,4))
default[0,0] += 1
default[1,1] += 1
default[2,2] += -1
default[3,3] += -1

#Loading Mueller matrix elements for the three telescopes:

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

# uncomment to assume IDEAL Mueller matrix elements (testing purposes)
#for i in np.arange(len(M_LFT[:])):
#    M_LFT[i] = np.diag([1,1,-1,-1])
#for i in np.arange(len(M_MFT[:])):
#    M_MFT[i] = np.diag([1,1,-1,-1])
#for i in np.arange(len(M_HFT[:])):
#    M_HFT[i] = np.diag([1,1,-1,-1])

#Introducing the maps/sky models:

NSIDE = 64
NPIX = 12*NSIDE**2

CMB_sky = pysm3.Sky(nside=NSIDE, preset_strings=['c1'])
dust_sky = pysm3.Sky(nside=NSIDE, preset_strings=['d1'])
sync_sky = pysm3.Sky(nside=NSIDE, preset_strings=['s1'])

# Specifying instrument specifics (PTEP paper):

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
                       
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Counting how many objects are in data_files

nkey = len(chan_dicts)

#Plotting the Mueller matrix elements channel by channel:

fig, axs = plt.subplots(3, 3)

for i in np.arange(nkey):
    telescope = chan_dicts[i]['telescope']
    
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
    
    def label_cond(string):
        label = ''
        if i in [0,13,18]:
            label = string
        return label
    
    for s1 in np.arange(3):
        for s2 in np.arange(3):
            axs[s1,s2].axhline(default[s1,s2],color='gray')
            axs[s1,s2].plot(freqs_channel*1e-9,muellers[bandidx,s1,s2][0],color=c_string,label=label_cond(telescope))
            axs[s1,s2].tick_params(direction='in')
            axs[s1,s2].set_xticks([100,200,300,400])
            if s1 != 3:
                axs[s1,s2].set_xticklabels(['', '', '', ''])
            ymin = np.array(axs[s1,s2].get_ylim())[0]
            ymax = np.array(axs[s1,s2].get_ylim())[1]
            ymean = np.mean([ymin,ymax])
            axs[s1,s2].set_yticks([ymean+(ymin-ymean)*2/3,ymean+(ymax-ymean)*2/3])
            axs[s1,s2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs[s1,s2].tick_params(color='gray', labelcolor='gray')
            for spine in axs[s1,s2].spines.values():
                spine.set_edgecolor('gray')

axs[0,0].legend()
fig.tight_layout(pad=0)
fig.set_size_inches(10.5, 7)
plt.savefig('mueller.pdf')
plt.clf()

#UNTIL HERE!

#Producing maps affected by the HWP, channel by channel.

for i in np.arange(nkey):
    name = chan_dicts[i]['name']
    telescope = chan_dicts[i]['telescope']
    
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

    bandidx = np.where((freqs >= fmin) & (freqs <= fmax))
    freqs_channel = freqs[bandidx]
    fmin_ch = freqs_channel[:-1]
    fmax_ch = freqs_channel[1:]
    
    CMB_maps = np.empty((len(freqs_channel),3,NPIX))
    dust_maps = np.empty((len(freqs_channel),3,NPIX))
    sync_maps = np.empty((len(freqs_channel),3,NPIX))
    
    for j in np.arange(len(freqs_channel)):
        freq = freqs_channel[j]
        CMB_maps[j] = CMB_sky.get_emission(freq*u.Hz).to(getattr(u,'uK_CMB'),equivalencies=u.cmb_equivalencies(freq*u.Hz))
        dust_maps[j] = dust_sky.get_emission(freq*u.Hz).to(getattr(u,'uK_CMB'),equivalencies=u.cmb_equivalencies(freq*u.Hz))
        sync_maps[j] = sync_sky.get_emission(freq*u.Hz).to(getattr(u,'uK_CMB'),equivalencies=u.cmb_equivalencies(freq*u.Hz))
        
    mIIs = muellers[bandidx,0,0][0]
    mQQs = muellers[bandidx,1,1][0]
    mQUs = muellers[bandidx,1,2][0]
    mUQs = muellers[bandidx,2,1][0]
    mUUs = muellers[bandidx,2,2][0]  
    
    rho = (mQQs-mUUs)/2
    eta = (mQUs+mUQs)/2
    
    I_CMB_ch = np.einsum('ij,i->j',CMB_maps[:-1,0,:],mIIs[:-1]) + np.einsum('ij,i->j',CMB_maps[1:,0,:],mIIs[1:])
    Q_CMB_ch = np.einsum('ij,i->j',CMB_maps[:-1,1,:],rho[:-1]) + np.einsum('ij,i->j',CMB_maps[1:,1,:],rho[1:]) \
                + np.einsum('ij,i->j',CMB_maps[:-1,2,:],eta[:-1]) + np.einsum('ij,i->j',CMB_maps[1:,2,:],eta[1:])
    U_CMB_ch = np.einsum('ij,i->j',CMB_maps[:-1,2,:],rho[:-1]) + np.einsum('ij,i->j',CMB_maps[1:,2,:],rho[1:]) \
                - np.einsum('ij,i->j',CMB_maps[:-1,1,:],eta[:-1]) - np.einsum('ij,i->j',CMB_maps[1:,1,:],eta[1:])
    maps_CMB_ch = np.array([I_CMB_ch,Q_CMB_ch,U_CMB_ch])/(2*(len(freqs_channel)-1))
    del(I_CMB_ch)
    del(Q_CMB_ch)
    del(U_CMB_ch)
    
    I_dust_ch = np.einsum('ij,i->j',dust_maps[:-1,0,:],mIIs[:-1]) + np.einsum('ij,i->j',dust_maps[1:,0,:],mIIs[1:])
    Q_dust_ch = np.einsum('ij,i->j',dust_maps[:-1,1,:],rho[:-1]) + np.einsum('ij,i->j',dust_maps[1:,1,:],rho[1:]) \
                + np.einsum('ij,i->j',dust_maps[:-1,2,:],eta[:-1]) + np.einsum('ij,i->j',dust_maps[1:,2,:],eta[1:])
    U_dust_ch = np.einsum('ij,i->j',dust_maps[:-1,2,:],rho[:-1]) + np.einsum('ij,i->j',dust_maps[1:,2,:],rho[1:]) \
                - np.einsum('ij,i->j',dust_maps[:-1,1,:],eta[:-1]) - np.einsum('ij,i->j',dust_maps[1:,1,:],eta[1:])
    maps_dust_ch = np.array([I_dust_ch,Q_dust_ch,U_dust_ch])/(2*(len(freqs_channel)-1))
    del(I_dust_ch)
    del(Q_dust_ch)
    del(U_dust_ch)
        
    I_sync_ch = np.einsum('ij,i->j',sync_maps[:-1,0,:],mIIs[:-1]) + np.einsum('ij,i->j',sync_maps[1:,0,:],mIIs[1:])
    Q_sync_ch = np.einsum('ij,i->j',sync_maps[:-1,1,:],rho[:-1]) + np.einsum('ij,i->j',sync_maps[1:,1,:],rho[1:]) \
                + np.einsum('ij,i->j',sync_maps[:-1,2,:],eta[:-1]) + np.einsum('ij,i->j',sync_maps[1:,2,:],eta[1:])
    U_sync_ch = np.einsum('ij,i->j',sync_maps[:-1,2,:],rho[:-1]) + np.einsum('ij,i->j',sync_maps[1:,2,:],rho[1:]) \
                - np.einsum('ij,i->j',sync_maps[:-1,1,:],eta[:-1]) - np.einsum('ij,i->j',sync_maps[1:,1,:],eta[1:])
    maps_sync_ch = np.array([I_sync_ch,Q_sync_ch,U_sync_ch])/(2*(len(freqs_channel)-1))
    del(I_sync_ch)
    del(Q_sync_ch)
    del(U_sync_ch)
    
    maps_sky_ch = maps_CMB_ch + maps_dust_ch + maps_sync_ch
    
    hp.write_map('output_'+str(NSIDE)+'/maps_'+name, maps_sky_ch, overwrite=True, dtype=['float64','float64','float64'])
    print('channel '+name+' done.')

print('it took ' + str((time.time()-start)) + ' seconds at NSIDE ' + str(NSIDE))
