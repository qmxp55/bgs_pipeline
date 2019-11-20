import os
import sys
sys.path.append('/global/homes/q/qmxp55/DESI/omarlibs')
import raichoorlib
import numpy as np
from raichoorlib import search_around
from astropy.table import Table
from geometric_def import query_catalog_mask, LSLGA_fit, LSLGA_veto, circular_mask_radii_func
import astropy.io.fits as fits
import fitsio
import glob
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as units
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from main_def import get_random

from argparse import ArgumentParser
ap = ArgumentParser(description='Get a list of targets masked by the defined masking radius')
ap.add_argument('--n', type=float, default=1.2,
                help="n times the masking radius used in tractor to mask around the bright stars.")

ns = ap.parse_args()

n = ns.n


inptfiles = {}
inptfiles['dr8pix']     = '/project/projectdirs/desi/target/catalogs/dr8/0.31.1/pixweight/pixweight-dr8-0.31.1.fits'
inptfiles['bgsdr8relax'] = '/global/cscratch1/sd/qmxp55/bgs_dr8_0.31.1_relaxed.npy'
inptfiles['dr8maskbitsource'] = '/global/cscratch1/sd/qmxp55/sweep_files/dr8_sweep_whole_maskbitsource.npy'
inptfiles['desitile']   = '/global/cscratch1/sd/raichoor/desi-tiles-viewer.fits' # from Eddie, see [desi-survey 647]


# for healpy
hdr          = fits.getheader(inptfiles['dr8pix'],1)
nside,nest   = hdr['hpxnside'],hdr['hpxnest']
npix         = hp.nside2npix(nside)
pixarea      = hp.nside2pixarea(nside,degrees=True)

# is in desi nominal footprint? (using tile radius of 1.6 degree)
# small test shows that it broadly works to cut on desi footprint 
def get_isdesi(ra,dec):
    radius   = 1.6 # degree
    tmpnside = 16
    tmpnpix  = hp.nside2npix(tmpnside)
    # first reading desi tiles, inside desi footprint (~14k deg2)
    hdu  = fits.open(inptfiles['desitile'])
    data = hdu[1].data
    keep = (data['in_desi']==1)
    data = data[keep]
    tra,tdec = data['ra'],data['dec']
    # get hppix inside desi tiles
    theta,phi  = hp.pix2ang(tmpnside,np.arange(tmpnpix),nest=nest)
    hpra,hpdec = 180./np.pi*phi,90.-180./np.pi*theta
    hpindesi   = np.zeros(tmpnpix,dtype=bool)
    _,ind,_,_,_= raichoorlib.search_around(tra,tdec,hpra,hpdec,search_radius=1.6*3600)
    hpindesi[np.unique(ind)] = True
    ## small hack to recover few rejected pixels inside desi. Avoid holes if any
    tmp  = np.array([i for i in range(tmpnpix) 
                     if hpindesi[hp.get_all_neighbours(tmpnside,i,nest=nest)].sum()==8])
    hpindesi[tmp] = True
    ##
    pixkeep    = np.where(hpindesi)[0]
    # now compute the hppix for the tested positions
    pix  = hp.ang2pix(tmpnside,(90.-dec)*np.pi/180.,ra*np.pi/180.,nest=nest)
    keep = np.in1d(pix,pixkeep)
    return keep

cat = np.load(inptfiles['bgsdr8relax']) # catalogue
catindesi = get_isdesi(cat['RA'],cat['DEC']) # True if is in desi footprint
print('------------- Relaxed BGS DR8 catalogue DONE... ------------------')

Nranfiles = 3
randoms = get_random(N=Nranfiles, sweepsize=None, dr='dr8') # randoms
indesiranfile = '/global/cscratch1/sd/qmxp55/random_dr8_indesi_N%s.npy' %(Nranfiles)
is_indesiranfile = os.path.isfile(indesiranfile)
if is_indesiranfile: 
    ranindesi = np.load(indesiranfile)
else: 
    ranindesi = get_isdesi(randoms['RA'],randoms['DEC']) # True is is in desi footprint
    np.save(indesiranfile, ranindesi)
    
print('------------- Random catalogue DONE... ------------------')
    
masksources = np.load(inptfiles['dr8maskbitsource']) #load dr8 masksource catalogue containing GAIA, TYCHO and LSLGA objects

#
def get_bsmask():
    
    mag = np.zeros_like(masksources['RA'])
    ingaia = (masksources['REF_CAT'] == b'G2') & (masksources['G'] <= 13)
    intycho = (masksources['REF_CAT'] == b'T2')
    
    # get MAG_VT mag from Tycho
    path = '/global/homes/q/qmxp55/DESI/matches/'
    tycho = fitsio.read(path+'tycho2.fits')
    idx2, idx1, d2d, d_ra, d_dec = search_around(masksources['RA'][intycho], masksources['DEC'][intycho], tycho['RA'], tycho['DEC'], search_radius=0.2)
    mag[intycho] = tycho['MAG_VT'][idx1]
    
    mag[np.where(ingaia)] = masksources['G'][ingaia]
    keep = (ingaia) | (intycho)
    
    tab = Table()
    for col in ['RA', 'DEC', 'MAG', 'REF_CAT']:
        if col == 'MAG': tab[col] = mag[keep]
        else: tab[col] = masksources[col][keep]
    print('%i Bright Stars' %(np.sum(keep)))
    
    return tab

bscat = get_bsmask()
#bscat = bscat[:1000]

#
for n in [n]:
    
    start = raichoorlib.get_date()
    bs_dir = '/global/cscratch1/sd/qmxp55/BS_mask/'
    
    #Dustin_radii's
    mag = np.linspace(0, 20, 50)
    Dustin_MS_radii = []
    for i,j in enumerate(mag):
        Dustin_MS_radii.append([j, n*np.minimum(1800., 150. * 2.5**((11. - j)/3.)) * 0.262/1])
    
    stars_cat = query_catalog_mask(cat['RA'][catindesi], cat['DEC'][catindesi], bscat, Dustin_MS_radii, nameMag='MAG', diff_spikes=False, 
                             length_radii=None, widht_radii=None, return_diagnostics=False, bestfit=False)
    print(n, 'cat stars DONE...')

    np.save(bs_dir+'bs_main_%sxdustinradii_cat' %(str(n)), stars_cat)
    
    stars_ran = query_catalog_mask(randoms['RA'][ranindesi], randoms['DEC'][ranindesi], bscat, Dustin_MS_radii, nameMag='MAG', diff_spikes=False, 
                             length_radii=None, widht_radii=None, return_diagnostics=False, bestfit=False)
    print(n, 'ran stars DONE...')
    
    np.save(bs_dir+'bs_main_%sxdustinradii_ran' %(str(n)), stars_ran)
    
    end = raichoorlib.get_date()
    print(n, start)
    print(n, end)