
"""
Continuous wavelet transform module for Python. Includes a collection
of routines for wavelet transform computation of local dominant wavenumber
and smoothing through convolution with Hann window.
To be used together with LWA code

Authors: Georgios Fragkoulidis, Paolo Ghinassi

"""

######################
# modules to import  #
######################


from numpy import append, asarray, arange, array, argsort, arctanh, ceil, concatenate, conjugate, cos, diff, exp, intersect1d, isnan, isreal, log, log2, mod, ones, pi, prod, real, round, sort, sqrt, unique, zeros, polyval, nan, ma, floor, interp, loadtxt, savetxt, angle, argmax
#from numpy.fft import fft, ifft, fftfreq
from scipy import fft, ifft, arange
from scipy.signal import hann
from scipy.fftpack import fftfreq, fftshift, rfft
from numpy.random import randn
from numpy.lib.polynomial import polyval
from scipy.stats import chi2
from scipy.special import gamma
from scipy.signal import convolve2d, lfilter
from scipy.special.orthogonal import hermitenorm
from os import makedirs
from os.path import expanduser
from sys import stdout
from time import time
from scipy import ndimage
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hann
from scipy.ndimage import convolve
from scipy import fft, ifft
from scipy.fftpack import fftfreq, fftshift, rfft

class Morlet:
    """Implements the Morlet wavelet class.

    Note that the input parameters f and f0 are angular frequencies.
    f0 should be more than 0.8 for this function to be correct, its
    default value is f0=6.

    """

    name = 'Morlet'

    def __init__(self, f0=6.0):
        self._set_f0(f0)

    def psi_ft(self, f, sigma):
        """Fourier transform of the approximate Morlet wavelet."""
        #return (pi ** -.25) * exp(-0.5 * (f - self.f0) ** 2.)  # for shape=1
        #return (2 ** .5) * sqrt(2) * (pi ** .25) * exp(-2. * (f - self.f0) ** 2.) # for shape=2
        #return 2 * (2 ** -.5) *(pi ** -(1./4.)) * exp(-2. * (f - self.f0) ** 2.)  # for shape=2 and modified Morlet
        #return 2 * exp(-1. * (f - self.f0) ** 2.)  # fot shape=1 and modified Morlet
        return (4*pi*(sigma**2.))**(1./4.) * exp(-1.* (sigma**2. *(f - self.f0)**2. )/2.   )  # Yi full Morlet version (Eq. 9.5)

    def psi(self, t, sigma):
        """Morlet wavelet as described in Torrence and Compo (1998)."""
        #return (pi ** -.25) * exp(1j * self.f0 * t - t ** 2. / 2.)  # for shape=1
        #return (2 ** -.5) * (pi ** -.25) * exp(1j * self.f0 * t - t ** 2. / 8.)  # for shape=2 
        #return (2 ** -1.5) * (2 ** .5) *(pi ** -(3./4.)) * exp(1j * self.f0 * t - t ** 2. / 8.)  # for shape=2 and modified Morlet
        #return (2 ** .5) *(pi ** (1./4.)) * exp(1j * self.f0 * t - t ** 2. / 2.)  # for shape=1 and modified Morlet
        return sqrt(2./(pi*(sigma**2.))) * exp(1j * self.f0 * t - t**2./(2.*(sigma**2.)))  # Yi full Morlet version (Eq.5)
        
    def flambda(self):
        """Fourier wavelength as of Torrence and Compo (1998)."""
        return (4 * pi) / (self.f0 + sqrt(2 + self.f0 ** 2))

    def coi(self):
        """e-Folding Time as of Torrence and Compo (1998)."""
        return 1. / sqrt(2.)

    def sup(self):
        """Wavelet support defined by the e-Folding time."""
        return 1. / coi

    def _set_f0(self, f0):
        # Sets the Morlet wave number, the degrees of freedom and the
        # empirically derived factors for the wavelet bases C_{\delta}, \gamma,
        # \delta j_0 (Torrence and Compo, 1998, Table 2)
        self.f0 = f0             # Wave number
        self.dofmin = 2          # Minimum degrees of freedom
        if self.f0 == 6.:
            self.cdelta = 0.776  # Reconstruction factor
            self.gamma = 2.32    # Decorrelation factor for time averaging
            self.deltaj0 = 0.60  # Factor for scale averaging
        else:
            self.cdelta = 1
            self.gamma = 1
            self.deltaj0 = 1
    
    
        



def fftconv(x, y):
    """ Convolution of x and y using the FFT convolution theorem. """
    N = len(x)
    n = int(2 ** ceil(log2(N))) + 1
    X, Y, x_y = fft(x, n), fft(y, n), []
    for i in range(n):
        x_y.append(X[i] * Y[i])

    # Returns the inverse Fourier transform with padding correction
    return ifft(x_y)[4:N+4]



def cwt(signal, ap, sigma, dx=1., dj=1./12, s0=-1, J=-1, wavelet=Morlet(), result=None):
    """Continuous wavelet transform of the signal at specified scales.

    PARAMETERS
        signal (array like) :
            Input signal array
        ap (integer):
            Number of times the signal is appended. 
        dx (float) :
            Sample spacing.
        dj (float, optional) :
            Spacing between discrete scales. Default value is 0.25.
            Smaller values will result in better scale resolution, but
            slower calculation and plot.
        s0 (float, optional) :
            Smallest scale of the wavelet. Default value is 2*dt.
        J (float, optional) :
            Number of scales less one. Scales range from s0 up to
            s0 * 2**(J * dj), which gives a total of (J + 1) scales.
            Default is J = (log2(N*dt/so))/dj.
        wavelet (class, optional) :
            Mother wavelet class. Default is Morlet()
        result (string, optional) :
            If set to 'dictionary' returns the result arrays as itens
            of a dictionary.

    RETURNS
        W (array like) :
            Wavelet transform according to the selected mother wavelet.
            Has (J+1) x N dimensions.
        sj (array like) :
            Vector of scale indices given by sj = s0 * 2**(j * dj),
            j={0, 1, ..., J}.
        freqs (array like) :
            Vector of Fourier frequencies (in 1 / time units) that
            corresponds to the wavelet scales.
        coi (array like) :
            Returns the cone of influence, which is a vector of N
            points containing the maximum Fourier period of useful
            information at that particular time. Periods greater than
            those are subject to edge effects.
        fft (array like) :
            Normalized fast Fourier transform of the input signal.
        fft_freqs (array like):
            Fourier frequencies (in 1/time units) for the calculated
            FFT spectrum.

    EXAMPLE
        mother = wavelet.Morlet(6.)
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var,
            0.25, 0.25, 0.5, 28, mother)

    """
    n0 = len(signal)                              # Original signal length.
    if s0 == -1: s0 = 2 * dx / wavelet.flambda()  # Smallest resolvable scale
    if J == -1: J = int(log2(n0 * dx / s0) / dj)  # Largest resolvable scale. Determines the number of scales. 
    
    N = 2 ** (int(log2(n0)) + 1)                  # Next higher power of 2. Look at http://www.bitweenie.com/listings/fft-zero-padding/
    signal_ft = fft(signal,n0)                   # Signal Fourier transform + normalization
    #signal_ft = 2*abs(signal_ft)
    ftfreqs = fftfreq(n0)*(n0/ap)                # Fourier angular frequencies. Multiply by n0 so that fftfreq=0,1,2,3,... This gives the same as ftfreqs = 2*pi*fftfreq(n0,dt).  2p/(n0*dt)=1 when n0=2p/dt which is always true. The signal has to be of length 2p (complete cycle).
    

    sj = s0 * 2. ** (arange(0, J+1) * dj)         # The scales
    freqs = 1. / (wavelet.flambda() * sj)         # As of Mallat 1999

    # Creates an empty wavelet transform matrix and fills it for every discrete scale using the convolution theorem.
    W = zeros((len(sj), n0), 'complex')
    for n, s in enumerate(sj):
        psi_ft_bar = ((s * ftfreqs[1] * n0) ** 0.5 * conjugate(wavelet.psi_ft(s * ftfreqs, sigma)))
        W[n, :] = ifft(signal_ft * psi_ft_bar, n0)

    # Checks for NaN in transform results and removes them from the scales, frequencies and wavelet transform.
    sel = ~isnan(W).all(axis=1)
    sj = sj[sel]
    freqs = freqs[sel]
    W = W[sel, :]

    # Determines the cone-of-influence. Note that it is returned as a function
    # of time in Fourier periods. Uses triangular Bartlett window with non-zero
    # end-points.
    coi = (n0 / 2. - abs(arange(0, n0) - (n0 - 1) / 2))
    coi = wavelet.flambda() * wavelet.coi() * dx * coi
    
    return (W[:, :n0], sj, freqs, coi, signal_ft/(n0**0.5), ftfreqs)
    #return (W[:, :n0], sj, freqs, coi, signal_ft[1:N/2] / N ** 0.5, ftfreqs[1:N/2] / (2. * pi))







def icwt(signal, ap, sigma, emp_cdelta, W, sj, dx, dj=0.25, wavelet=Morlet()):
    """Inverse continuous wavelet transform.

    PARAMETERS
        W (array like):
            Wavelet transform, the result of the cwt function.
        sj (array like):
            Vector of scale indices as returned by the cwt function.
        dx (float) :
            Sample spacing.
        dj (float, optional) :
            Spacing between discrete scales as used in the cwt
            function. Default value is 0.25.
        w (class, optional) :
            Mother wavelet class. Default is Morlet()

    RETURNS
        iW (array like) :
            Inverse wavelet transform.

    EXAMPLE
        mother = wavelet.Morlet(6.)
        wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var,
            0.25, 0.25, 0.5, 28, mother)
        iwave = wavelet.icwt(wave, scales, 0.25, 0.25, mother)

    """
    n0 = len(signal) 
    N = 2 ** (int(log2(n0)) + 1) 
    #signal_ft = fft(signal,n0)                   # Signal Fourier transform + normalization
    ftfreqs = fftfreq(n0)*(n0/ap)        
    a, b = W.shape
    c = sj.size
    if a == c:
        sj = (ones([b, 1]) * sj).transpose()
    elif b == c:
        sj = ones([a, 1]) * sj
    else:
        raise Warning('Input array dimensions do not match.')
    
    # Calculate Cdelta from TC98 Eq. 13
    #W_del = zeros((len(sj), n0), 'complex')
    #print(W_del.shape)
    #for n, s in enumerate(sj):
    #    psi_ft_bar = ((s * ftfreqs[1] * n0 ) ** 0.5 * conjugate(w.psi_ft(s * ftfreqs))) 
    #    W_del[n, :] = ifft(psi_ft_bar, n0) 
    #my_cdelta = dj * sqrt(dx) / w.psi(0) * (real(W_del[:,0]) / sqrt(sj[:,0])).sum(axis=0)
    #ggg = (real(W_del) / sqrt(sj)).sum(axis=0)
    #print(W_del.shape,ggg.shape)
    #print(my_cdelta)
    #iW = dj * sqrt(dx) / (my_cdelta * w.psi(0)) * (real(W) / sqrt(sj)).sum(axis=0)
    
    # Empirical Cdelta for the Yi (2012) full Morlet(6) equation
    emp_cdelta = 3.1
    iW = dj * sqrt(dx) / (emp_cdelta * wavelet.psi(0, sigma)) * (real(W) / sqrt(sj)).sum(axis=0)
    # As of Torrence and Compo (1998), eq. (11)
    #iW = dj * sqrt(dx) / (wavelet.cdelta * wavelet.psi(0)) * (real(W) / sqrt(sj)).sum(axis=0)
    return iW



def wnedit(y, kmin, kmax):
        """
        Take a function and return it with only a range of wavenumbers using fft and ifft
        """
        ffty = fft(y)
        mask = zeros(len(ffty))
        mask[kmin:kmax+1] = 1 # Keep positive freqs only. Values outside the selected range remain zero. 
        mafft = ffty*mask
        fedit = ifft(mafft)
        fedit = 2*fedit.real # Since the ignored negative frequencies would contribute the same as the positive ones
        if kmin == 0:
            fedit = fedit - ffty.real[0]/len(ffty) # Subtract the pre-edited function's mean. We don't want a double contribution from it in fedit (as was demanded by fedit=2*fedit.real).
        elif kmin > 0:
            fedit = fedit + ffty.real[0]/len(ffty) # Add the pre-edited function's mean. The zero frequency in FFT should never be left out when editing a function. 
        return fedit
        


def dom_wn(signal, input_smooth, output_smooth, hann_width_z):
    """Calculation of dominant wavelength at every longitude

    INPUT:
        signal (array like):
            Input signal array
        input_smooth (boolean):
            If True, the input signal is smoothed to wavenumbers 0-20 before we do the wavelet analysis
        output_smooth (boolean):
            If True, the dominant wavenumber series is smoothed with a Hann window of hann_width before we output it
        hann_width (integer):
            Number of grid points in longitude for the smoothing (Hann window width). 

    OUTPUT:
        dom_wavenumber (array like) :
            Dominant wavenumber in every longitude of the input signal.
    """
    rectify = True  # If we don't use the rectification technique there is a bias toward low wavenumbers.
    kmin = 0  
    kmax = 20
    mother = Morlet(6)          # Morlet mother wavelet with wavenumber=6
    sigma = 0.7
    ap = 3                      # append the signal 2 times
    std = signal.std()                      # Standard deviation
    std2 = std ** 2                      # Variance
    if input_smooth:
        signal = wnedit(signal, kmin, kmax) # Edit function to only contain the frequencies we are interested in 
    var_temp = append(signal,signal,0) 
    var = append(var_temp,signal,0) 
    N = var.size     
    
    # Which scales (wavenumbers) to resolve with CWT
    dx = 0.1
    s0 = 2 * dx                          # Smallest resolvable scale (largest wavenumber)
    dj = 0.02                            # Determines scale resolution
    J = 6 / dj                           # Largest resolvable scale. Determines total number of scales

    wave, scales, freqs, coi, fft, fftfreqs = cwt(var, ap, sigma, dx, dj, s0, J, mother)
    wnumber = 2*pi*freqs
    power = (abs(wave)) ** 2 /std2           # Normalized wavelet power spectrum
    if rectify:
        power = power / (scales[:,None])

    max_wn = zeros(N)
    max_wn_smooth = zeros(N)



    for l in range(0,N):
        wnu = argmax(power[:,l])        # array index of dominant wavenumber in this latitude
        max_wn[l] = wnumber[wnu]        # the dominant wavenumber in this latitude

	
	# remove padding added for wavelets #
    limit1 = int(N/3)
    limit2 = int(2*N/3)
    max_wn = max_wn[limit1:limit2]

	# smooth along the zonal with circular comvolution (periodic b'ries) #
    if output_smooth:
        
        hann_zon = hann(hann_width_z)
        max_wn_smooth = ndimage.convolve(max_wn, hann_zon, mode='wrap')/sum(hann_zon)

    if output_smooth:
        return max_wn_smooth
    else:
        return max_wn




def dom_wavenumber_2D (v, lons):

	"""
	Computation of local [time, lev, lat, lon] dominant wavenumber through wavelet analysis

    INPUT:
        v (array like):
            Input meridional wind. Shape must be [time, lev, lat, lon]

    OUTPUT:
        dom_wavenumber_2D_smooth (array like) :
            Local dominant zonal wavenumber. Shape is [time, lev, lat, lon].
			Some smoothing is applied in the zonal and meridional direction to avoid jumps between integer wavenumers. 
    """

	notime=v.shape[0]
	nolev=v.shape[1]
	noLats=v.shape[2]
	noLons=v.shape[3]
	
	dom_wavenumber_2D = np.zeros([notime,nolev,noLats,noLons])
	dom_wavenumber_2D_smooth = np.zeros_like(dom_wavenumber_2D)

	# determine dominant zonal wavenumber from meridional wind using wavelet
	# hann_width_z,m is the width (in terms of grid points) of the Hann window in terms of zonal and meridional direction

	# grid resolution
	
	res=abs(int(lons[1]-lons[0]))

	# width in the meridional set to 10 degrees of latitude

	hann_merid_width =10
	hann_merid = signal.hann(hann_merid_width//res)

	# width along the zonal set to 10 degrees of latitude

	hann_zonal_width =40

	for t in range(0,notime): 
		for i in range(0,nolev): 
			for lat_index in range(0,noLats): 
				dom_wavenumber_2D[t,i,lat_index,:] = dom_wn(v[t,i,lat_index,:],input_smooth=True,output_smooth=True,hann_width_z=(hann_zonal_width//res))
			for long_index in range(0,len(lons)):
				dom_wavenumber_2D_smooth[t,i,:,long_index] = signal.convolve(dom_wavenumber_2D[t,i,:,long_index], hann_merid, mode='same') / sum(hann_merid)

	return dom_wavenumber_2D_smooth




def convolve_Hann_lon(arr1, lons, k):

	"""
	Smoothing of a signal through convolution with hann window whose width depends on longitude too.

    INPUT:
        arr1 (array like):
            1-D Input signal array

		lons:
			array with longitues values

		k (array like) :
            Local dominant wavenumber at every longitude.
			Used to set the width of the Hann window.

    OUTPUT:
        conv (array like) :
            1-D Convolved signal with Hann window
    """

	narr1 = len(arr1)

	# repeat the first array for periodic boundaries #
	arr1 = np.tile(arr1, 3)

	#length of the convolution array 

	nconv = len(arr1) + len(hann(int(len(lons)/k[-1]+1)))		  #length of the convolution array 
	conv = np.zeros(nconv)

	# extend the zonal wavenumber array
	endk=np.ones(nconv-len(k))*k[-1]
	k = np.append(k, [endk]) 
	if len(arr1) == 0:
		raise ValueError('arr1 cannot be empty')

	for i in range (0, nconv):
		i1 = i
		tmp = 0.0
		arr2 = hann(int(len(lons)/k[i]+1))
		for j in range (0, len(arr2)):
		
			if i1 >= 0 and i1 < len(arr1):
				tmp = tmp + (arr1[i1]*arr2[j])
 
			i1 = i1-1
			conv[i] = tmp

	nb = (len(arr2) - 1) % 2
	
	if (nb != 0):
		nb = int((len(arr2) + 1) / 2)
		conv = conv[(narr1+nb):-(nb+narr1)]
	else:
		nb = int((len(arr2) - 1 )  / 2)
		conv = conv[(narr1+nb):-(nb+narr1+1)]

	#remove padding/extended boundaries

	arr1=arr1[narr1:-narr1]

	# do some calibration to preserve area #


	coeff = abs(np.trapz(arr1)/np.trapz(conv))
	conv = conv*coeff

	return conv

def HannSmoothing_time_2D (signal, lons, k):

	"""
	Smoothing/filtering of a signal through convolution with hann window whose width depends on longitude too.

    INPUT:
        signal (array like):
            Input signal array. Shape must be [time, lev, lat, lon].

		k: Local dominant wavenumber at every longitude. Shape is [time, lev, lat, lon].
		

    OUTPUT:
        smooth_signal (array like) :
           Convolved signal with Hann window
    """

	notime=signal.shape[0]
	nolev=signal.shape[1]
	noLats=signal.shape[2]
	noLons=signal.shape[3]
	
	smooth_signal = np.zeros([notime,nolev,noLats,noLons])

	for t in range(0,notime): 
		for i in range(0,nolev): 
			for lat_index in range(0,noLats): 
				smooth_signal[t,i,lat_index,:] = convolve_Hann_lon(signal[t,i,lat_index,:],lons,k[t,i,lat_index,:])

	return smooth_signal

######################################################################################
# Function to compute dominant zonal wavenumber for each longitude at each time step #
######################################################################################

def zonalWN_fourier(signal, lats, lons):

	"""
	Function to compute dominant zonal wavenumber at a latitude circle at each time step 

    INPUT:
        signal (array like):
            Input signal array. Shape must be [time, lev, lat, lon].

	lats: array with latitude values.
	lons: array with longitude values.

    OUTPUT:
        maxWN_mrunmean (array like) :
            Convolved signal with Hann window of constant width. Shape is [time, lev, lat, lon].
    """


	notime = signal.shape[0]
	nolev=signal.shape[1]

	# loop over all latitudes between lat1 and lat2 (descending in lat)

	n = (len(lons)+1)//2
	A = np.zeros([notime, nolev, len(lats), len(lons)], dtype='complex') # complex array to hold the fourier transf of the input signal
	power = np.zeros([notime, nolev, len(lats),n])
	#k=np.arange(1,n,1) # zonal wavenumber
	maxWN = np.zeros([notime,nolev, len(lats)])

	#loop over time, isentropes and latitudes
	for t in range (0,notime):
		for l in range (0,nolev):
			for ilat in range(0,len(lats)):
				A[t,l,ilat,:] = pl.fft(signal[t,l,ilat,:]) # fourier transform of the signal (v) at each latitude
				for kk in range(0,n):			       # exclude wavenumber 0
					power[t,l,ilat,kk] = 4*(A[t,l,ilat,kk].real**2 + A[t,l,ilat,kk].imag**2)
		
				maxWN[t,l,ilat] = power[t,l,ilat].argmax(axis=0) # the index at which the maximum in the power spectrum occurs, which is the maximum wavenumber

	### Smoothing of the signal ###	
	
	# fill the remaining latutues with the boundary values 
	
	ext_lat=10       		# extension of the additional boundaries in degrees of latitude
	res=int(abs(lats[1]-lats[0]))		# resolution in lats
	N = ext_lat//res 		# window for the moving average in terms of index in the lats array!!! so N=5 is an extended 10 deg boundary for a 2*2 resolution

	# now apply some smoothing to avoid jumps between different integers wavenumers

	maxWN_mrunmean=np.zeros([notime, nolev, len(lats)])

	for t in range (0,notime):
		for l in range (0,nolev):

			maxWN_mrunmean[t,l,:] = np.convolve(maxWN[t,l,:], np.ones((N,))/N, mode='same')

	maxWN_mrunmean=np.where(maxWN_mrunmean<1,1,maxWN_mrunmean)

	return maxWN_mrunmean

#####################################################################################
# Convolution of the signal with Hann window to preform a phase averaging/smoothing #
#####################################################################################

def Hann_convolution(signal, lats, lons, maxWNRunMean, calibration):

	"""
	Smoothing/filtering of a signal through convolution with Hann window of constant width.

    INPUT:
        signal (array like):
            Input signal array. Shape must be [time, lev, lat, lon].

		lats: array with latitude values.
		lons: array with longitude values.

		maxWNRunMean: dominant wavenumber at a latitude circle. Shape is [time, lev, lat].
		calibration (bool):
			set to 1 if you want to do some calibration to preserve the area

    OUTPUT:
        smoothed_signal (array like) :
           Convolved signal with Hann window
    """

	signal = np.nan_to_num(signal)
	
	# number of points of the original signal over longitude

	N = len(lons)                       
	smoothed_signal=np.zeros_like(signal)

	notime = signal.shape[0]
	nolev=signal.shape[1]


	for t in range(0,notime):
		for l in range (0,nolev):
			for i in range(0,len(lats)):
			

				# hann function with length (N*pi/max_zonal_wavenumber)
				win = hann(int(N/(maxWNRunMean[t,l,i])))
				# do convolution
				smoothed_signal[t,l,i,:] = convolve(signal[t,l,i,:], win, mode='wrap') / np.trapz(win)
				# print "the length of the hann window is ", dlambda*len(win), "deg in longitude."

				# do some calibration - USE ONLY WITH POSITIVE QUANTITIES
				if calibration :
					const = np.trapz(signal[t,l,i])/np.trapz(smoothed_signal[t,l,i])
					smoothed_signal[t,l,i] = smoothed_signal[t,l,i]*const


	return smoothed_signal

def hilbert(y):
	"""
	############################################################################################################################
	- Envelope calculation using the Hilbert transform technique (Marple, 1999, Zimin et al 2003)
	############################################################################################################################
	- INPUT:
		* y: 1-D function for which we want the envelope
	############################################################################################################################
	"""
	N = len(y)
	# FFT of y
	z = fft(y)
	# Zero-out the negative frequencies
	z[(int(N/2)+1):N] = 0
	# Double the positive frequencies except from the 0th and (N/2)th ones
	z = 2*z
	z[0] = z[0]/2
	z[int(N/2)] = z[int(N/2)]/2
	# Inverse FFT
	z = ifft(z)
	# Envelope
	z = abs(z)
	return z

"Calculates the Fourier power spectrum of the signal"

def fourierPower(signal):

    
    # input checking
    if np.ndim(signal) != 1:
        raise ValueError("In fourierPower(signal), signal must be one dimensional array")

    A = fft(signal)
    n = (len(signal)+1)//2
    power = np.zeros(n)
    for i in range(n):
        power[i] = 4*(A[i].real**2 + A[i].imag**2)
    return power

def add_cyclic_point(data, coord=None, axis=-1):
    """
    Add a cyclic point to an array and optionally a corresponding
    coordinate.

    Args:

    * data:
        An n-dimensional array of data to add a cyclic point to.

    Kwargs:

    * coord:
        A 1-dimensional array which specifies the coordinate values for
        the dimension the cyclic point is to be added to. The coordinate
        values must be regularly spaced.

    * axis:
        Specifies the axis of the data array to add the cyclic point to.
        Defaults to the right-most axis.

    Returns:

    * cyclic_data:
        The data array with a cyclic point added.

    * cyclic_coord:
        The coordinate with a cyclic point, only returned if the coord
        keyword was supplied.

  
    """
    if coord is not None:
        if coord.ndim != 1:
            raise ValueError('The coordinate must be 1-dimensional.')
        if len(coord) != data.shape[axis]:
            raise ValueError('The length of the coordinate does not match '
                             'the size of the corresponding dimension of '
                             'the data array: len(coord) = {}, '
                             'data.shape[{}] = {}.'.format(
                                 len(coord), axis, data.shape[axis]))
        delta_coord = np.diff(coord)
        if not np.allclose(delta_coord, delta_coord[0]):
            raise ValueError('The coordinate must be equally spaced.')
        new_coord = ma.concatenate((coord, coord[-1:] + delta_coord[0]))
    slicer = [slice(None)] * data.ndim
    try:
        slicer[axis] = slice(0, 1)
    except IndexError:
        raise ValueError('The specified axis does not correspond to an '
                         'array dimension.')
    new_data = ma.concatenate((data, data[slicer]), axis=axis)
    if coord is None:
        return_value = new_data
    else:
        return_value = new_data, new_coord
    return return_value




