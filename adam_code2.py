# Adam Jussila
# 11/1/2017
# 1D_raypath.py - this code is intended to run a set of "photons" moving radially 
# outward from a point source representing a pulsar. These photons will travel 
# randomly with gassian deflections through several screens and be forced to hit
# a specified point at the end, representing us as the observer recieving them.
# It also will calculate the delay of a given photon at each step relative to a 
# straight line path.

# v2b - 3/23/18 - Adam & Dan - takes Adam's 3/23 code and adds Dan's
#                 more general screen strength specification.

import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from astropy.io import fits
from scipy.stats import kde


######################################################################
#################### RAYTRACE FUNCTION HERE ##########################
######################################################################

def raytrace_2d(nrays, sigma0, vel, nscr):

# seed random number generator here (a second one below!!!@#@)
    np.random.seed(38430340)
    rand.seed(3242202105)
    
#    The screen strength weighting scheme is as follows (screen j):

#        sigma0 = passed in: sets the scale of all the deflections
#        Sr[j] = strength of the randdom component of the ray
#        Sd[j] = strength of the directed component of the ray
#          A[j] = axial ratio   sigmax/sigmay ; y gets divided by this value
#          psi[j] = position angle (rel. to x-axis) of the directed component



    # creating variables. Path is x or y location at each screen.
    # thetax and thetay are the angular positions at each screen.
    # omega and tau are the delay and fringe frequency corroloaries
    nscreen = nscr
    pathx = np.zeros((nrays, nscreen+1)) # x position. nscreen indices. 
    pathy = np.zeros((nrays, nscreen+1)) # y position with nscreen indices.
    thetax = np.zeros((nrays, nscreen+1)) # one deflection angle at each screen plus the initial one.
    thetay = np.zeros((nrays, nscreen+1))
    tau = np.zeros(nrays) # array that holds delays
    omega = np.zeros(nrays) # array of omega values for each ray.

    Sd = np.zeros(nrays)  # strength (in units of sigma0) of directed component
    psi = np.zeros(nrays)  # position angle (reltative to x-axis) of directed component
    AR = np.ones(nrays)    # axial ratio of ellipse  sigmay = sigmax / AR
    Sr = np.ones(nrays)

    AR[15] = 15.
    Sd[15] = 30.
    psi[15] = np.pi/6.

# We specify the simulation parameters here
    # Run 1 - 3/23/23  15:30
    # directed screen at s=0.2

    # amplitudes
    amp = np.zeros(nrays)
    dz = 3.1e19/nscreen # distance between screens (1kpc is 3.1e19 meters)

    # GENERATE EACH RAY HERE:
    for i in range(nrays):
        pathx[i,0] = 0
        pathy[i,0] = 0
        for j in range(nscreen+1):
            
            # this is the general ray creation mechanism.
            gx = rand.gauss(0,1) # x amplitude of directed component
            gy = rand.gauss(0,1) # y amplitude of directed component
            gr = rand.gauss(0,1) # amplitude of the random component
            psi_rand = 2. * np.pi * rand.random()  # uniform [0, 2 pi)
            
            theta1 = sigma0*(Sd[j]*(gx*np.cos(psi[j]) - gy*np.sin(psi[j])/AR[j])
                              + Sr[j]*gr*np.cos(psi_rand))
            theta2 = sigma0*(Sd[j]*(gx*np.sin(psi[j]) + gy*np.cos(psi[j])/AR[j])
                              + Sr[j]*gr*np.sin(psi_rand))
            
            # Calculate the amplitude of this ray
               # may need tweaking - Dan (3/23/18)

            amp[i] += (gx**2 + gy**2)
            
            # adjust theta based on what our deflection did to photons.
            thetax[i,j] = thetax[i,j-1] + theta1
            thetay[i,j] = thetay[i,j-1] + theta2

            # tracks the path the ray takes.
            pathx[i,j] = ((thetax[i,j])*dz) + pathx[i,j-1]
            pathy[i,j] = ((thetay[i,j])*dz) + pathy[i,j-1]

        amp[i] = amp[i]/(2*nscreen)  # 2 because of adding two squared N(0,1) variables

# finish the amplitude determination
 #  make largest amplitude (which is an exponent) be 0
 #  then exponentiate it
    
    maxamp = amp.max()
    print('maximum amplitude before rescaling: ',maxamp)
    for y in range(nrays):
        amp[i] = np.exp(maxamp - amp[i])


    # converge on the observer more neatly by subtracting a small amount from
    # each step.
    for ray in range(nrays):
        dispx = (pathx[ray,nscreen])/(nscreen)
        dispy = (pathy[ray,nscreen])/(nscreen)
        for scr in range(1,nscreen+1):
            pathx[ray,scr] = pathx[ray,scr] - (dispx*(scr))
            pathy[ray,scr] = pathy[ray,scr] - (dispy*(scr))

    # calculate the omega values assuming the pulsar is moving in only x or 
    # y direction and not a combination of the two.
    
    # the total weighting sum, divides at end.
    sum_weights = 0
    sj = 0.0
    wj = 0.0
    for i in range(1,nscreen):
        # the screen fractional distance
        sj = float(i)/float(nscreen)
        # the screen weighting
        wj = sj/(1-sj)
        # adding each weighting to the total
        sum_weights += wj


    # getting omega values
    for ray in range(nrays):
        for i in range(nscreen):
            # the screen fractional distance
            sj = float(i)/float(nscreen)
            # the screen weighting
            wj = sj/(1-sj)

            # update theta values for the bent ray
            thetax[ray,i] = (pathx[ray,i]-pathx[ray,i-1])/dz
            thetay[ray,i] = (pathy[ray,i]-pathy[ray,i-1])/dz

            # add the screen plus the weighting
            omega[ray] += (thetax[ray,i]*wj)

            # gets the tau delays relative to to straight-line path. (small
            # angle approximation used here.
            xdelay = ((thetax[ray,i]**2)*dz)/(2*(3e8)) #seconds
            ydelay = ((thetay[ray,i]**2)*dz)/(2*(3e8)) #seconds
            tau[ray] += np.sqrt(xdelay**2+ydelay**2)

#        print thetax[ray,nscreen]

        # calculate the final omega by getting right units/undo weighting
        omega[ray] = 2*np.pi*omega[ray]*vel/(sum_weights*0.37) #divide by wavelength of .37m

    # random phase approximation for a given ray.
#    phi = 2.0 * np.pi * np.random.rand(nrays) # random phase

    # set the omega and tau values to zero and make ray 0 the source point.
    amp[0] = 1.e5
    omega[0] = 0
    tau[0] = 0

    # FIND ALL INTERFERENCE TERMS HERE
    sec = [ (0,0) for i in range((nrays*nrays))]
    sec_amp = np.zeros((nrays*nrays))
    idx = 0
    for ray1 in range(nrays):
        for ray2 in range(nrays):
            if (ray1 != ray2):
                # difference term
                sec[idx] = ((omega[ray2]-omega[ray1]),(tau[ray2]-tau[ray1]))
            elif (ray1 == ray2):
                # self term
                sec[idx] = (0,0)

            #saving the amplitude of the combined rays for every interference.
            sec_amp[idx] = amp[ray1]*amp[ray2]
            idx += 1
# if you wanted to, this is where you would create a dynamic.
#########    dyn = makeDyn(nx,ny,nscreen,phi,omega,tau, nrays)

    return pathx, pathy, thetax, thetay, sec, tau, omega, sec_amp



######################################################################
##################### MAKE DYNAMIC HERE ##############################
######################################################################

# make the dynamic when given all of the component (delay, omega, phi)
def makeDyn(nx,ny, nscreen, phi, omega, tau, nrays):

    # make 1-D arrays that store the index values of each point.
    t = np.linspace(0,3600,nx)
    v = np.linspace(800000000,870000000,ny)

    # making empty arrays to store the 2-D indices
    tt = np.zeros((nx,ny))
    vv = np.zeros((nx,ny))

    for row in range(nx):
        for col in range(ny):
            tt[row][col] = t[col]
            vv[row][col] = v[row]

    efield = np.zeros((nx,ny))

    for i in range(nrays):
        c = (np.cos(np.pi*((omega[i]*tt) + (tau[i]*vv)) + phi[i]))
        efield = efield + c
    efield = efield**2
    
    efield = scipy.ndimage.filters.gaussian_filter(efield,1)

    return efield



#####################################################################
###################### MAKE SECONDARY HERE ##########################
#####################################################################

def makeSec(dyn, nrays):
    sec = np.fft.fftn(dyn-np.mean(dyn))
    sec = np.absolute(np.fft.fftshift(sec))**2
    sec = 10*np.log10(sec/np.max(sec))
    return sec



######################################################################
################# MAIN FUNCTION HERE #################################
######################################################################

def main():
    vel = 200000 #meters per second
    nscreen = 51
    trials = 300; #number of raypaths
    sigma0 = 0.06; #overall angular scale (milliarcseconds = mas)
    sigma0 = sigma0 / 2.06e+08      # convert to radians
    pathx, pathy, thetax, thetay, secondary, tau, omega, sec_amp  = \
            raytrace_2d(trials,sigma0, vel, nscreen)
    sec = secondary

    #separating out the tuples into an x and a y and add each one amp times.
    sec_amp = sec_amp.astype(float)
    x = np.zeros(trials*trials); y = np.zeros(trials*trials)
    idx = 0
    for point in range(len(sec)-1):
        x[idx] = sec[point][0]
        y[idx] = sec[point][1]
        idx += 1

    max = 0
    for ray in range(trials):
        if (np.abs(omega[ray]) > np.abs(omega[max])):
            max = ray
            
    sec_amp[0] = 0.

    #create a power spectrum with aplitudes and coordinates.
    full_sec, xlim, ylim = np.histogram2d(y,x,bins=350,weights=sec_amp)

# logarithm to see the power in fainter features
    full_sec = np.log(full_sec)

# ray tracing plot
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(221)
    ax1.set_title(str(trials) +"rays Cross Section")
    ax1.set_ylabel("Distance (AU)")
    ax1.set_xlabel("Distance (AU)")

    for i in range(trials):
        plt.plot(pathx[i,18]/(10**11),pathy[i,18]/(10**11), c='black', marker='o')

# dynamic spectrum
    ax1 = fig.add_subplot(222)
    ax1.set_title(str(trials) +" Raypaths Dynamic Spectrum")
    ax1.set_ylabel("Frequency (MHz.)")
    ax1.set_xlabel("Time (min)")

    # making the dynamic spectrum stuff
    # random phase approximation for a given ray.
    phi = 2.0 * np.pi * np.random.rand(trials) # random phase
    nx = 1024
    ny = 1024
    dyn = makeDyn(nx,ny, nscreen, phi, omega, tau, trials)

#    sec = makeSec(dyn, trials)
    plt.imshow(dyn, extent=[0,60,800,870], aspect='auto', cmap='bone_r')
    
# wavefield picture
    ax1 = fig.add_subplot(223)
    ax1.set_title(str(trials) +" rays Wavefield Representation")
    ax1.set_xlabel("Omega (Hz)")
    ax1.set_ylabel("Tau (s)")
    plt.plot(omega,tau,linestyle='none',marker='o')

# secondary spectrum
    ax1 = fig.add_subplot(224)
    ax1.set_title( str(trials) +" rays Secondary Spectrum")
    ax1.set_xlabel("Omega (Hz)")
    ax1.set_ylabel("Tau (s)")
    plt.imshow(full_sec, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], aspect='auto',cmap='bone_r')
#hb = ax1.hexbin(x, y, gridsize=50, cmap='inferno')
#td = ax1.hist2d(x, y, bins=50, cmap=plt.cm.BuGn_r)
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins=50
    k = kde.gaussian_kde([x,y])
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
 
# Make the plot
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
 
# Change color palette
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Greens_r)

# Add shading    

    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)

# Contour

    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    plt.contour(xi, yi, zi.reshape(xi.shape) )

    
# plot all of the images
    plt.tight_layout()
    plt.show()

main()