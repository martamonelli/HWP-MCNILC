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

NSIDE = 512
NPIX = 12*NSIDE**2

CMB_sky = pysm3.Sky(nside=NSIDE, preset_strings=['c1'])
dust_sky = pysm3.Sky(nside=NSIDE, preset_strings=['d1'])
sync_sky = pysm3.Sky(nside=NSIDE, preset_strings=['s1'])

#Loading data from IMO:

# list of LiteBIRD channels
chan_dicts = np.array([{'IMO':'L1-040'},
                       {'IMO':'L2-050'},
                       {'IMO':'L1-060'},
                       {'IMO':'L3-068'},
                       {'IMO':'L2-068'},
                       {'IMO':'L4-078'},
                       {'IMO':'L1-078'},
                       {'IMO':'L3-089'},
                       {'IMO':'L2-089'},
                       {'IMO':'L4-100'},
                       {'IMO':'L3-119'},
                       {'IMO':'L4-140'},
                       {'IMO':'M1-100'},
                       {'IMO':'M2-119'},
                       {'IMO':'M1-140'},
                       {'IMO':'M2-166'},
                       {'IMO':'M1-195'},
                       {'IMO':'H1-195'},
                       {'IMO':'H2-235'},
                       {'IMO':'H1-280'},
                       {'IMO':'H2-337'},
                       {'IMO':'H3-402'},
                       ])
                       
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# opening IMO schema.json file and interpreting it as a dictionary
f = open('input/schema.json',)
data = json.load(f)  

# looking into the IMO, data['data_files'] is where the relevant info is stored
data_files = data['data_files']

# counting how many objects are in data_files
nkey=0
for key in data_files:
    nkey = nkey+1

#Plotting the Mueller matrix elements channel by channel:

fig, axs = plt.subplots(4, 4)

for i in np.arange(len(chan_dicts)):
    channel_IMO = chan_dicts[i]['IMO']
    
    if channel_IMO[0] == 'L':
        freqs = freqs_LFT 
        muellers = M_LFT
        c_string = 'crimson'
    
    if channel_IMO[0] == 'M':
        freqs = freqs_MFT 
        muellers = M_MFT
        c_string = 'purple'
        
    if channel_IMO[0] == 'H':
        freqs = freqs_HFT 
        muellers = M_HFT
        c_string = 'mediumslateblue'

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # looking for the detectors belonging to the selected channel
    for j in range(nkey):
        test = data_files[j]
        if(test['name'] == 'channel_info'):
            metadata = test['metadata']
            if(metadata['channel'] == channel_IMO):
                detector_names = metadata['detector_names']
                break

    nIMO = len(detector_names)
    det_indices = range(nIMO)

    list_of_dictionaries = []

    # looking for the metadata of the detectors in detector_names
    for d in detector_names:
        for j in range(nkey):
            test = data_files[j]
            if(test['name'] == 'detector_info'):
                metadata = test['metadata']
                if (metadata['name'] == d):
                    list_of_dictionaries.append(metadata)
                    break
                    
    bandcenter = list_of_dictionaries[0]['bandcenter_ghz']*1e9 #[Hz]
    bandwidth = list_of_dictionaries[0]['bandwidth_ghz']*1e9   #[Hz]
    
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
    
    for s1 in np.arange(4):
        for s2 in np.arange(4):
            axs[s1,s2].axhline(default[s1,s2],color='gray')
            axs[s1,s2].plot(freqs_channel*1e-9,muellers[bandidx,s1,s2][0],color=c_string,label=label_cond(channel_IMO[0]+'FT'))
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

#Producing maps affected by the HWP, channel by channel.

for i in np.arange(len(chan_dicts)):
    channel_IMO = chan_dicts[i]['IMO']
    
    if channel_IMO[0] == 'L':
        freqs = freqs_LFT 
        muellers = M_LFT
    
    if channel_IMO[0] == 'M':
        freqs = freqs_MFT 
        muellers = M_MFT
        
    if channel_IMO[0] == 'H':
        freqs = freqs_HFT 
        muellers = M_HFT
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # looking for the detectors belonging to the selected channel
    for j in range(nkey):
        test = data_files[j]
        if(test['name'] == 'channel_info'):
            metadata = test['metadata']
            if(metadata['channel'] == channel_IMO):
                detector_names = metadata['detector_names']
                break

    nIMO = len(detector_names)
    det_indices = range(nIMO)

    list_of_dictionaries = []

    # looking for the metadata of the detectors in detector_names
    for d in detector_names:
        for j in range(nkey):
            test = data_files[j]
            if(test['name'] == 'detector_info'):
                metadata = test['metadata']
                if (metadata['name'] == d):
                    list_of_dictionaries.append(metadata)
                    break
                    
    bandcenter = list_of_dictionaries[0]['bandcenter_ghz']*1e9 #[Hz]
    bandwidth = list_of_dictionaries[0]['bandwidth_ghz']*1e9   #[Hz]
    
    fmin = bandcenter - bandwidth/2
    fmax = bandcenter + bandwidth/2

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
    
    hp.write_map('output_'+str(NSIDE)+'/maps_'+channel_IMO, maps_sky_ch, overwrite=True, dtype=['float64','float64','float64'])
    print('channel '+str(channel_IMO)+' done.')

print('it took ' + str((time.time()-start)/60) + ' minutes at NSIDE ' + str(NSIDE))
