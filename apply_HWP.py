import numpy as np
import healpy as hp

def apply_HWP(CMB_path, NSIDE=512, FGmodel='d1s1'):
    
    sky_HWP = np.empty((22,3,12*NSIDE**2))
    
    CMB_map = hp.read_map(CMB_path, field=None)

    input_coeff = 'intermediate/CMB_coeffs.npz'
    f = np.load(input_coeff)
    ch_idxs = f['ch_idx']
    gs = f['g']
    rhos = f['rho']
    etas = f['eta']

    # list of LiteBIRD channels
    chan_dicts = np.array([{'name':'L1-040'},
                           {'name':'L2-050'},
                           {'name':'L1-060'},
                           {'name':'L3-068'},
                           {'name':'L2-068'},
                           {'name':'L4-078'},
                           {'name':'L1-078'},
                           {'name':'L3-089'},
                           {'name':'L2-089'},
                           {'name':'L4-100'},
                           {'name':'L3-119'},
                           {'name':'L4-140'},
                           {'name':'M1-100'},
                           {'name':'M2-119'},
                           {'name':'M1-140'},
                           {'name':'M2-166'},
                           {'name':'M1-195'},
                           {'name':'H1-195'},
                           {'name':'H2-235'},
                           {'name':'H1-280'},
                           {'name':'H2-337'},
                           {'name':'H3-402'},
                           ])
                       
    for i in np.arange(len(chan_dicts)):
        ch_name = chan_dicts[i]['name']
        FGs_HWP = hp.read_map('intermediate/'+FGmodel+'_'+str(NSIDE)+'_'+ch_name, field=None)
        
        mixing = np.array([[gs[i],0,0],[0,rhos[i],etas[i]],[0,-etas[i],rhos[i]]])
        CMB_HWP = np.dot(mixing,CMB_map)
        
        sky_HWP[i] = FGs_HWP + CMB_HWP
    
    return sky_HWP
 
CMB_path = 'random_CMB'
    
maps = apply_HWP(CMB_path, FGmodel='d1s1')
