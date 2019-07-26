import numpy

def radec2pos(ra, dec):
    """ converting ra dec to position on a unit sphere.
        ra, dec are in degrees.
    """
    pos = numpy.empty(len(ra), dtype=('f8', 3))
    ra = ra * (numpy.pi / 180)
    dec = dec * (numpy.pi / 180)
    pos[:, 2] = numpy.sin(dec)
    pos[:, 0] = numpy.cos(dec) * numpy.sin(ra)
    pos[:, 1] = numpy.cos(dec) * numpy.cos(ra)
    return pos

def veto(coord, center, R):
    """
        Returns a veto mask for coord. any coordinate within R of center
        is vet.

        Parameters
        ----------
        coord : (RA, DEC)
        center : (RA, DEC)
        R     : degrees

        Returns
        -------
        Vetomask : True for veto, False for keep.

    """
    from kdcount import KDTree

    pos = radec2pos(center[0], center[1])
    tree = KDTree(pos)
    
    if numpy.isscalar(R):
        #print('This is the value of R =%g'%(R))
        R = center[0]*0 + R
        
    R = 2 * numpy.sin(numpy.radians(R) * 0.5) 

    pos = radec2pos(coord[0], coord[1])
    other = KDTree(pos)
    vetoflag = numpy.zeros(len(pos), dtype='?')
    
    Rmax = R.max()

    def process(r, i, j):
        # i is tycho, j is objects
        rcut = R[i]
        jcut = j[(r < rcut) & (r != 0)]
        vetoflag[jcut] |= True


    tree.root.enum(other.root, Rmax, process)
    return vetoflag


#OMAR.TWOMASS
def veto_ellip(coord, center, a, b, l):
    """
        Returns a veto mask for coord. any coordinate within R of center
        is vet.

        Parameters
        ----------
        coord : (RA, DEC)
        center : (RA, DEC)
        R     : degrees

        Returns
        -------
        Vetomask : True for veto, False for keep.

    """
    from kdcount import KDTree

    pos = radec2pos(center[0], center[1])
    tree = KDTree(pos)

    #R = 2 * np.sin(np.radians(a) * 0.5)
    R = a
    pos = radec2pos(coord[0], coord[1])
    other = KDTree(pos)
    
    vetoflag = numpy.zeros(len(pos), dtype='?')
    #j_near = numpy.zeros(len(pos), dtype=numpy.float64)
    
    Rmax = R.max()
    #print (Rmax)
    
    x = numpy.array(coord[0])
    y = numpy.array(coord[1])
    h = numpy.array(center[0])
    k = numpy.array(center[1])
    a1 = numpy.array(a)
    b1 = numpy.array(b)
    l1 = numpy.array(l)
    #j2 = numpy.array()
    
    def process(r, i, j):
        # i is 2mass, j is objects
        #rcut = R[i]
        X = x[j]
        Y = y[j]
        H = h[i]
        K = k[i]
        A = a1[i]
        B = b1[i]
        L = numpy.radians(l1[i])
        #JEXT = j2[i] 
        c1 = (1/A**2)*(numpy.cos(L))**2 + (1/B**2)*(numpy.sin(L))**2
        c2 = 2*numpy.cos(L)*numpy.sin(L)*(1/A**2 - 1/B**2)
        c3 = (1/A**2)*(numpy.sin(L))**2 + (1/B**2)*(numpy.cos(L))**2
        rell = c1*(X - H)**2 + c2*(X - H)*(Y - K) + c3*(Y - K)**2
        jcut = j[rell < 1]
        vetoflag[jcut] |= True
        #j_near[jcut] = JEXT

    tree.root.enum(other.root, Rmax, process)
    return vetoflag
        
def match(coord, center, R):
    """
        Returns a veto mask for coord. any coordinate within R of center
        is vet.

        Parameters
        ----------
        coord : (RA, DEC)
        center : (RA, DEC)
        R     : degrees

        Returns
        -------
        Vetomask : True for veto, False for keep.

    """
    from kdcount import KDTree

    pos = radec2pos(center[0], center[1])
    tree = KDTree(pos)
    
    if numpy.isscalar(R):
        #print('This is the value of R =%g'%(R))
        R = center[0]*0 + R

    R = 2 * numpy.sin(numpy.radians(R) * 0.5) 

    pos = radec2pos(coord[0], coord[1])
    other = KDTree(pos)
    #vetoflag = numpy.zeros(len(pos), dtype='?')
    test_j = []
    test_i = []
    
    Rmax = R.max()

    def process(r, i, j):
        # i is tycho, j is objects
        rcut = R[i]
        jcut = (j[r < rcut], i)
        j1 = jcut[0]
        j2 =jcut[1]
        test_j.append(j1)
        test_i.append(j2)

    tree.root.enum(other.root, Rmax, process)
    return numpy.array(test_j[0]), numpy.array(test_i[0])