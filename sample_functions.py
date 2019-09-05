# -*- coding: utf-8 -*-

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Compute eq latitude and barotropic LWA 
	from a given absolute vort field on a lat lon grid
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# modules to import #
from __future__ import division
import numpy as np


# some constants

OMEGA = 7.29*10**-5 # earth angular velocity [rad/s]
a = 6371200 #earth radius [m]


def PV_contours_sample(PV, lons, lats):

	""" 

	computes numerically the relation between equivalent latitudes 
	and associated PV contours such that they enclose the same area on the sphere.

	input variables:

	PV = potential vorticity. Shape must be [lat,lon]
	lons = longitudes from grid
	lats = latitudes from grid 

	returns:

	Q_interp = PV contours associated to GRID lats. Shape is [PV_bins]

	NOTE: in this version of the code equivalent latitude are computed just for a matter of interpolation to find Q_interp.
		  grid latitudes are used as eq. latitudes then Q_interp are the Q contours associated to grid lats.
	"""
	
	
	PV_bins=len(lats) #number of PV contour Q, here is set coincident to len(lats)
	noLats = len(lats)
	noLons = len(lons)
	notime = PV.shape[0]


	# initialise the integrals for INT 1
	integ = np.zeros([PV_bins,noLons]) # initialise the integrals to 0
	integral = np.zeros(PV_bins)
	q = np.zeros(PV_bins)
	Q_interp=np.zeros_like(q)
	integ1=integ2=integ3=0

	# initialise the integrals for INT 2
	int2 = np.zeros([PV_bins,noLats])
	phiM = np.zeros(PV_bins)
	integ_zonmean = 0
	

	### INT 1: calculate area north of PV contours ###

	# prescribed PV contours equispaced from min to max of PV range of values

	q[:] = np.linspace(np.nanmin(PV[:,:]), np.nanmax(PV[:,:]), num=PV_bins, endpoint=True)

	# loop in PV_bins: eq lat
	# loop in nolats: grid lats

	for j in xrange(0, PV_bins):
		for h in xrange(0,noLons-1):		
			for k in xrange(0,noLats-1):
	
				if (PV[k,h]-q[j]) > 0 and (PV[k+1,h]-q[j]) > 0:  # if PV-q is positive at both grid points use trapezoidal rule to calculate the integral in dphi
					#print ('a')
					integ1 = (abs((lats[k]-lats[k+1])*np.pi/180))*np.cos((lats[k]+lats[k+1])*np.pi/360)*(abs((lons[h+1]-lons[h])*np.pi/180))*a**2
		
				elif (PV[k,h]-q[j])*(PV[k+1,h]-q[j]) < 0 and (PV[k,h]-q[j])<(PV[k+1,h]-q[j]):
					#print ('b')
					phistar2 = np.interp(0, [(PV[k,h]-q[j]), (PV[k+1,h]-q[j])], [lats[k], lats[k+1]])
					integ2 = (abs((phistar2-lats[k+1])*np.pi/180))*np.cos((phistar2+lats[k+1])*np.pi/360)*(abs((lons[h+1]-lons[h])*np.pi/180))*a**2

				elif (PV[k,h]-q[j])*(PV[k+1,h]-q[j]) < 0 and (PV[k,h]-q[j])>(PV[k+1,h]-q[j]):
					#print ('c')
					phistar1 = np.interp(0, [(PV[k+1,h]-q[j]), (PV[k,h]-q[j])], [lats[k+1], lats[k]])	
					integ3 = (abs((lats[k]-phistar1)*np.pi/180))*np.cos((phistar1+lats[k+1])*np.pi/360)*(abs((lons[h+1]-lons[h])*np.pi/180))*a**2
		
				integ[j,h] += integ1+integ2+integ3
				integ1=integ2=integ3=0

		integral[j]=np.sum(integ[j,:]) #integrate in dlambda
	
	### INT 2: calculate area north to eq lat circle ###


	for j in xrange(0, PV_bins):
		for k in xrange(0,noLats-1):
			if np.sum(int2[j,:]) < integral[j]:	
				# compute the eulerian mean then integrate in dphi until int1=int2
				integ_zonmean = 2*np.pi*a**2*(abs((lats[k]-lats[k+1])*np.pi/180))*np.cos((lats[k]+lats[k+1])*np.pi/360)
				int2[j,k] += integ_zonmean
	
			else:
				break

		#interpolate to find the equivalent latitude given the first guess of Q contours
		phiM[j] = np.interp(integral[j], [np.sum(int2[j,:(k-1)]), np.sum(int2[j,:k])], [lats[k-1], lats[k]])

	# now interpolate back to find the q values that corresponds to the original grid lats
	Q_interp = np.interp(lats, phiM, q)

	

	return Q_interp


# compute local wave activity (LWA) from PV field


def local_wave_activity_sample(lons, lats, PV, Q):

	""" 
	Computes barotropic LWA from a given absolute vorticity field on a lat/lon regular grid.
 	First the relation bewteen latitudes and PV contours must be determined. 

	input variables:

	lons = longitudes from grid
	lats = latitudes from grid 
	Q = prescribed PV contours. Shape must be [PV_bins]
	PV = potential vorticity. Shape must be [lat,lon]
	PV_bins= scalar number equals to the number of prescribed PV contours

	returns:

	integ = Local Wave activity (LWA). Shape is [lat,lon]
	
	NOTE: in this version of the code LWA is computed setting eq lats = grid lats
	"""

	noLats = len(lats)
	noLons = len(lons)

	# set grid lats as eq lats

	phiM=lats  

	# set PV_bins as the number of grid lats

	PV_bins = len(lats)

	integ2=integ3=integ4=integ6=integ7=integ8=0

	integ = np.zeros([PV_bins,noLons])


	for j in xrange (0,PV_bins): #loop for equivalent latitudes, that now are coincident with grid lats
		for h in xrange(0,noLons):		
			for k in xrange(0,noLats-1): #grid lat

				# compute the integrals on the area south to Phi_M
			
				if  lats[k] <= phiM[j] and (PV[k,h]-Q[j]) > 0 and (PV[k+1,h]-Q[j]) > 0 :  
					#print 'a'
					integ2 = (1/np.cos(phiM[j]*np.pi/180))*0.5*((PV[k,h]-Q[j])+(PV[k+1,h]-Q[j]))*(abs((lats[k]-lats[k+1])*np.pi/180))*a*np.cos((lats[k]+lats[k+1])*np.pi/360)
				
				if lats[k] <= phiM[j] and (PV[k,h]-Q[j])*(PV[k+1,h]-Q[j]) < 0 and (PV[k,h]-Q[j])<(PV[k+1,h]-Q[j]):
					#print 'b'
					phistar = np.interp(0, [(PV[k,h]-Q[j]), (PV[k+1,h]-Q[j])], [lats[k], lats[k+1]])
					integ3 = (1/np.cos(phiM[j]*np.pi/180))*0.5*(PV[k+1,h]-Q[j])*(abs((phistar-lats[k+1])*np.pi/180))*a*np.cos((lats[k]+lats[k+1])*np.pi/360)

				if lats[k] <= phiM[j] and (PV[k,h]-Q[j])*(PV[k+1,h]-Q[j]) < 0 and (PV[k,h]-Q[j])>(PV[k+1,h]-Q[j]):
					#print 'c'
					phistar = np.interp(0, [(PV[k+1,h]-Q[j]), (PV[k,h]-Q[j])], [lats[k+1], lats[k]])
					integ4 = (1/np.cos(phiM[j]*np.pi/180))*0.5*(PV[k,h]-Q[j])*(abs((lats[k]-phistar)*np.pi/180))*a*np.cos((lats[k]+lats[k+1])*np.pi/360)
	
				# then compute the integrals on the area north to Phi_M

	
				if  lats[k] > phiM[j] and (PV[k,h]-Q[j]) <= 0 and (PV[k+1,h]-Q[j]) <= 0 :
					#print 'd'
					integ6 = (1/np.cos(phiM[j]*np.pi/180))*0.5*((Q[j]-PV[k,h])+(Q[j]-PV[k+1,h]))*(abs((lats[k]-lats[k+1])*np.pi/180))*a*np.cos((lats[k]+lats[k+1])*np.pi/360)
		
				if  lats[k] > phiM[j] and (Q[j]-PV[k,h])*(Q[j]-PV[k+1,h]) < 0 and (Q[j]-PV[k,h])<=(Q[j]-PV[k+1,h]):
					#print 'e'
					phistar = np.interp(0, [(Q[j]-PV[k,h]), (Q[j]-PV[k+1,h])], [lats[k], lats[k+1]])
					integ7 = (1/np.cos(phiM[j]*np.pi/180))*0.5*(Q[j]-PV[k+1,h])*(abs((phistar-lats[k+1])*np.pi/180))*a*np.cos((lats[k]+lats[k+1])*np.pi/360)
				
				if  lats[k] > phiM[j] and (Q[j]-PV[k,h])*(Q[j]-PV[k+1,h]) < 0 and (Q[j]-PV[k,h])>=(Q[j]-PV[k+1,h]):
					#print 'f'
					phistar = np.interp(0, [(Q[j]-PV[k+1,h]), (Q[j]-PV[k,h])], [lats[k+1], lats[k]])
					integ8 = (1/np.cos(phiM[j]*np.pi/180))*0.5*(Q[j]-PV[k,h])*(abs((lats[k]-phistar)*np.pi/180))*a*np.cos((lats[k]+lats[k+1])*np.pi/360)

				integ[j,h] += integ2+integ3+integ4+integ6+integ7+integ8
	
				integ2=integ3=integ4=integ6=integ7=integ8=0

	#return the integral which is the LWA [lat,lon]
	return integ 

