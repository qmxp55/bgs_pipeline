import fitsio
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, '/global/homes/q/qmxp55/DESI/BGS_paper/')
from desitarget.cuts import select_targets
from desitarget import io
#from desitarget.sv1.sv1_targetmask import desi_mask, bgs_mask, mws_mask

#nside = io.desitarget_nside()
sweep_dir_dr8south = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/south/sweep/8.0'
sweep_dir_dr8north = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/north/sweep/8.0'
dest = '/global/cscratch1/sd/qmxp55/desitarget_output/sweeps-dr8_south_r20.fits'

infilessouth = io.list_sweepfiles(sweep_dir_dr8south)
infilesnorth = io.list_sweepfiles(sweep_dir_dr8north)
#infiles = np.concatenate((infilessouth, infilesnorth)).tolist()
infiles = infilessouth

#
import time
start = time.time()

def flux_to_mag(flux):
    mag = 22.5 - 2.5*np.log10(flux)
    return mag

targets = select_targets(infiles, numproc=4, mask=True, tcnames=["BGS"], survey='main', backup=False)
bits = {'bright':1, 'faint':0, 'wise':2}
#['BGS_FAINT', 'BGS_BRIGHT', 'BGS_FAINT_EXT', 'BGS_LOWQ', 'BGS_FIBMAG', ]

end = time.time()
print('Total run time: %f sec' %(end - start))

rmag = flux_to_mag(targets['FLUX_R']/targets['MW_TRANSMISSION_R'])

for name, bit in zip(bits.keys(), bits.values()):
    bgsbit = ((targets['BGS_TARGET'] & 2**(bit)) != 0)
    print(name, np.sum(bgsbit))
    print('rmag min: %f, rmag max: %f' %(rmag[bgsbit].min(),rmag[bgsbit].max()))
    
# ADM write the targets to file.
io.write_targets(dest, targets, survey="main")