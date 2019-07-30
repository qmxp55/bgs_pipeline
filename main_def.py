import numpy as np
import sys, os, time, argparse, glob
import fitsio
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from matplotlib.ticker import NullFormatter
from astropy import units as u
import pandas as pd
from astropy.io import ascii

#from desiutil.log import get_logger, DEBUG
#log = get_logger()
class Point:

    def __init__(self, xcoord=0, ycoord=0):
        self.x = xcoord
        self.y = ycoord

class Rectangle:
    def __init__(self, bottom_left, top_right, colour):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.colour = colour

    def intersects(self, other):
        return not (self.top_right.x <= other.bottom_left.x or self.bottom_left.x >= other.top_right.x or self.top_right.y <= other.bottom_left.y or self.bottom_left.y >= other.top_right.y)
    
    def plot(self, other):
        fig, ax = plt.subplots(figsize=(15,8))
        rect = patches.Rectangle((self.bottom_left.x,self.bottom_left.y), abs(self.top_right.x - self.bottom_left.x), abs(self.top_right.y - self.bottom_left.y),linewidth=1.5, alpha=0.5, color='r')
        rect2 = patches.Rectangle((other.bottom_left.x,other.bottom_left.y), abs(other.top_right.x - other.bottom_left.x), abs(other.top_right.y - other.bottom_left.y),linewidth=1.5, alpha=0.5, color='blue')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        xlims = np.array([self.bottom_left.x, self.top_right.x, other.bottom_left.x, other.top_right.x])
        ylims = np.array([self.bottom_left.y, self.top_right.y, other.bottom_left.y, other.top_right.y])
        ax.set_xlim(xlims.min()-1, xlims.max()+1)
        ax.set_ylim(ylims.min()-1, ylims.max()+1)
        
def cut(ramin, ramax, decmin, decmax, catalog):
    
    mask = np.logical_and(catalog['RA'] >= ramin, catalog['RA'] <= ramax)
    mask &= np.logical_and(catalog['DEC'] >= decmin, catalog['DEC'] <= decmax)
    cat = catalog[mask]
    
    return cat

def get_sweep_patch(patch, rlimit=None, dr='dr7'):
    """
    Extract data from DECaLS DR7 SWEEPS files only.
    
    Parameters
    ----------
    patch: :class:`array-like`
        Sky coordinates in RA and DEC of the rectangle/square patch in format [RAmin, RAmax, DECmin, DECmax]
    rlimit: :class:`float`
        magnitude limit of data in the r-band with extinction correction
    
    Returns
    -------
    Subsample catalogue of SWEEP data.
    The subsample catalogue will be also stored with name 'sweep_RAmin_RAmax_DECmin_DECmax_rmag_rlimit' and numpy format '.npy'
    
    """
    import time
    start = time.time()
    
    if len(patch) != 4:
        log.warning('This require an input array of four arguments of the form [RAmin, RAmax, DECmin, DECmax]')
        raise ValueError
        
    if rlimit is None:
        log.warning('rlimit input is required')
        raise ValueError

    #patch
    ramin, ramax, decmin, decmax = patch[0], patch[1], patch[2], patch[3]
    sweep_file_name = '%s_sweep_%s_%s_%s_%s_rmag_%s' %(dr, str(ramin), str(ramax), str(decmin), str(decmax), str(rlimit))
    sweep_file = os.path.isfile(sweep_file_name+'.npy')
    if not sweep_file:
        if dr is 'dr7':
            sweep_dir = os.path.join('/global/project/projectdirs/cosmo/data/legacysurvey/','dr7', 'sweep', '7.1')
        elif dr == 'dr8-south':
            print('HERE!!!!!!!!')
            #sweep_dir = '/global/cscratch1/sd/adamyers/dr8/decam/sweep'
            sweep_dir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/south/sweep/8.0'
            #sweep_dir = '/global/cscratch1/sd/ioannis/dr8/decam/sweep-patched'
            
        elif dr is 'dr8-north':
            sweep_dir = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/north/sweep/8.0'
            
        df = cut_sweeps(ramin, ramax, decmin, decmax, sweep_dir, rlimit=rlimit)
        np.save(sweep_file_name, df)
    else:
        print('sweep file already exist at:%s' %(os.path.abspath(sweep_file_name+'.npy')))

    end = time.time()
    print('Total run time: %f sec' %(end - start))
    get_area(patch, get_val = False)
    print('Weight of %s catalogue: %s' %(sweep_file_name+'.npy', convert_size(os.path.getsize(sweep_file_name+'.npy'))))
    
    if not sweep_file:
        
        return df
    else:
        return np.load(os.path.abspath(sweep_file_name+'.npy'))
    
def convert_size(size_bytes): 
    import math
    if size_bytes == 0: 
            return "0B" 
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB") 
    i = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, i) 
    size = round(size_bytes / power, 2) 
    return "%s %s" % (size, size_name[i])
    
def get_area(patch, get_val = False):
    
    alpha1 = np.radians(patch[0])
    alpha2 = np.radians(patch[1])
    delta1 = np.radians(patch[2])
    delta2 = np.radians(patch[3])
    
    A = (alpha2 - alpha1)*(np.sin(delta2) - np.sin(delta1))*(180/np.pi)**(2)
    print('Area of %g < RA < %g & %g < DEC < %g: %2.4g deg^2' %(patch[0],
                                    patch[1], patch[2], patch[3], A))
    if get_val:
        return A

    
def cut_sweeps(ramin, ramax, decmin, decmax, sweep_dir, rlimit=None):
    '''Main function to extract the data from the SWEEPS'''
    
    cat1_paths = sorted(glob.glob(os.path.join(sweep_dir, '*.fits')))
    j = 0
    
    for fileindex in range(len(cat1_paths)):

        cat1_path = cat1_paths[fileindex]
        filename = cat1_path[-26:-5]
        brick = cat1_path[-20:-5]
        ra1min = float(brick[0:3])
        ra1max = float(brick[8:11])
        dec1min = float(brick[4:7])
        if brick[3]=='m':
            dec1min = -dec1min
        dec1max = float(brick[-3:])
        if brick[-4]=='m':
            dec1max = -dec1max
        
        r1=Rectangle(Point(ramin,decmin), Point(ramax, decmax), 'red')
        r2=Rectangle(Point(ra1min, dec1min), Point(ra1max, dec1max), 'blue')
        
        if not r1.intersects(r2):
            continue
        
        if j == 0:
            cat = fitsio.read(cat1_path)
            cat = cut(ramin, ramax, decmin, decmax, cat)
            if rlimit != None:
                rflux = cat['FLUX_R'] / cat['MW_TRANSMISSION_R']
                print('%s with %i objects out of %i at rmag=%2.2g' %(filename, len(cat[rflux > 10**((22.5-rlimit)/2.5)]), len(cat), rlimit))
                cat = cat[rflux > 10**((22.5-rlimit)/2.5)]
            else:
                print('%s with %i objects' %(filename, len(name)))
            j += 1
            continue
        
        name = fitsio.read(cat1_path, ext=1)
        name = cut(ramin, ramax, decmin, decmax, name)
        if rlimit != None:
                rflux2 = name['FLUX_R'] / name['MW_TRANSMISSION_R']
                print('%s with %i objects out of %i at rmag=%2.2g' %(filename, len(name[rflux2 > 10**((22.5-rlimit)/2.5)]), len(name), rlimit))
                name = name[rflux2 > 10**((22.5-rlimit)/2.5)]
        else:
            print('%s with %i objects' %(filename, len(cat)))
        
        cat = np.concatenate((cat, name))
        j += 1
        
    print('Bricks that matched: %i' %(j))
    print('Sample region # objects: %i' %(len(cat)))
    
    return cat

def get_random_patch(patch, N=3, sweepsize=None, dr='dr7'):
    
    import time
    start = time.time()
    
    if len(patch) != 4:
        log.warning('This require an input array of four arguments of the form [RAmin, RAmax, DECmin, DECmax]')
        raise ValueError
        
    if (dr is 'dr7') & (N < 2):
        log.warning('Number of RANDOMS files must be greater than one')
        raise ValueError
    
    import glob
    #ranpath = '/global/project/projectdirs/desi/target/catalogs/dr7.1/0.29.0/' #issues with MASKBITS...
    
    ramin, ramax, decmin, decmax = patch[0], patch[1], patch[2], patch[3]
    random_file_name = '%s_random_%s_%s_%s_%s_N_%s' %(dr, str(ramin), str(ramax), str(decmin), str(decmax), str(N))
        
    random_file = os.path.isfile(random_file_name+'.npy')
    if not random_file:
        if dr is 'dr7':
            ranpath = '/global/project/projectdirs/desi/target/catalogs/dr7.1/0.22.0/'
            randoms = glob.glob(ranpath + 'randoms*')
        elif (dr == 'dr8-south') or (dr == 'dr8-north'):
            ranpath = '/project/projectdirs/desi/target/catalogs/dr8/0.31.0/randoms/'
            randoms = glob.glob(ranpath + 'randoms-inside*')
            
        randoms.sort()
        randoms = randoms[0:N]

        for i in range(len(randoms)):
            df_ran = fitsio.read(randoms[i], columns=['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'MASKBITS'],upper=True, ext=1)
            df_ranS = cut(ramin, ramax, decmin, decmax, df_ran)

            print('total Size of randoms in %s: %i (within patch: %2.3g %%)' 
                  %(randoms[i][-9:-5], len(df_ran), 100*len(df_ranS)/len(df_ran)))
        
            if i == 0:
                df_ranS1 = df_ranS
                continue
        
            df_ranS1 = np.concatenate((df_ranS1, df_ranS))
            
        #elif dr is 'dr8c':
        #    ranpath = '/project/projectdirs/desi/target/catalogs/dr8c/PR490/'
        #    df_ran = fitsio.read(ranpath+'randoms-dr8c-PR490.fits', columns=['RA', 'DEC', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'MASKBITS'],upper=True, ext=1)
        #    df_ranS1 = cut(ramin, ramax, decmin, decmax, df_ran)
            
        np.save(random_file_name, df_ranS1)
            
        print('# objects in RANDOM patch: %i' %(len(df_ranS1)))
        if sweepsize is not None:
            print('fraction of RANDOM catalogue in patch compared to SWEEP catalogue in patch: %2.3g' 
                      %(len(df_ranS1)/sweepsize))
    else:
        print('RANDOM file already exist at:%s' %(os.path.abspath(random_file_name+'.npy')))

    end = time.time()
    print('Total run time: %f sec' %(end - start))
    get_area(patch, get_val = False)
    print('Weight of %s catalogue: %s' %(random_file_name+'.npy', convert_size(os.path.getsize(random_file_name+'.npy'))))
    
    if not random_file:
        return df_ranS1
    else:
        return np.load(os.path.abspath(random_file_name+'.npy'))
    
def flux_to_mag(flux):
    mag = 22.5 - 2.5*np.log10(flux)
    return mag

def search_around(ra1, dec1, ra2, dec2, search_radius=1., verbose=True):
    '''
    Using the astropy.coordinates.search_around_sky module to find all pairs within
    some search radius.
    Inputs:
    RA and Dec of two catalogs;
    search_radius (arcsec);
    Outputs:
        idx1, idx2: indices of matched objects in the two catalogs;
        d2d: angular distances (arcsec);
        d_ra, d_dec: the differences in RA and Dec (arcsec);
    '''

    # protect the global variables from being changed by np.sort
    ra1, dec1, ra2, dec2 = map(np.copy, [ra1, dec1, ra2, dec2])

    # Matching catalogs
    sky1 = SkyCoord(ra1*u.degree,dec1*u.degree, frame='icrs')
    sky2 = SkyCoord(ra2*u.degree,dec2*u.degree, frame='icrs')
    idx1, idx2, d2d, d3d = sky2.search_around_sky(sky1, seplimit=search_radius*u.arcsec)
    if verbose:
        print('%d nearby objects ~ %g %%'%(len(idx1), 100*len(idx1)/len(ra2)))

    # convert distances to numpy array in arcsec
    d2d = np.array(d2d.to(u.arcsec))


    d_ra = (ra2[idx2]-ra1[idx1])*3600.    # in arcsec
    d_dec = (dec2[idx2]-dec1[idx1])*3600. # in arcsec
    ##### Convert d_ra to actual arcsecs #####
    mask = d_ra > 180*3600
    d_ra[mask] = d_ra[mask] - 360.*3600
    mask = d_ra < -180*3600
    d_ra[mask] = d_ra[mask] + 360.*3600
    d_ra = d_ra * np.cos(dec1[idx1]/180*np.pi)
    ##########################################

    return idx1, idx2, d2d, d_ra, d_dec