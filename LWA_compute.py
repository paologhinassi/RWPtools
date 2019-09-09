"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	Computes local wave activity (LWA) in isentropic coordinates
	according to Ghinassi et al 2018 Monthly Weather Review

	tested with python 2.7
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


import datetime as dt
from netCDF4 import Dataset, date2num
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from cartopy import config
import cartopy.crs as ccrs

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
	the execution of this file requires the files:
	LWA_additional.py
	colorbar.py

	to be in the same dyrectory

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# code to compute the dominant zonal wavenumber and then smooth LWA
exec(open("./LWA_additional.py").read())

# colorbar repository
exec(open("./colorbar.py").read())

# set the path for plotting
plotdir = "./plots/"
datadir = "/nas/reference/ERA5/daily/"


# some constants

R = 287     # dry air constant [J/kg K]
cp = 1004   # specific heat at constant pressure for dry air [J/kg K]
pref = 100000   # p ref [Pa]
OMEGA = 7.29*10**-5 # earth angular velocity [rad/s]
a = 6370000 #earth radius [m]
g = 9.81    # constant gravitational acceleration [m/s^2]



#read the netcdf file be careful that pressure has to INCREASE in the array and has to be in Pa, res specifies resolution in terms of grid points
def read_data(space_res, time_res, y1, m1, d1, t1, y2, m2, d2, t2):

	fh = Dataset(datadir + 'u/u_day_ERA5_201101_201112.nc','r')
	lons = fh.variables['longitude'][::space_res]
	#exclude +-90 degrees due to cos lat at the denominator later on
	lats = fh.variables['latitude'][1:-1:space_res]
	pressure = fh.variables['level'][:]

	# check if pressure is increasing: #

	if np.all(np.diff(pressure) > 0):
		print ("pressure check ok")
	else:
		print ("pressure is not monotonically increasing in the array, the program is aborted!")
		raise SystemExit

	# check that latitude is decreasing and that there is no 90 deg due to cos phi at denominator: #

	if np.all(np.diff(lats) < 0) and lats[0]<90:
		print ("latitude check ok")
	else:
		print ("Latitude must decrease and be <90 deg!")
		raise SystemExit


	### TIME ###
	time = fh.variables['time'][:]
	time_units = fh.variables['time'].units # Reading in the time units
	cal_temps = fh.variables['time'].calendar # Calendar in use (proleptic_gregorian)

	# Pick a date and find the corresponding time index of the file.
	day1 = dt.datetime(y1, m1, d1, t1) # yr-month-date-time
	day2 = dt.datetime(y2, m2, d2, t2) # yr-month-date-time
	time_value1 = date2num(day1,units=time_units,calendar=cal_temps) # what's the value of the chosen day in the time array
	time_value2 = date2num(day2,units=time_units,calendar=cal_temps)
	t_list=list(time) # convert time array to list
	ts1=t_list.index(time_value1) # time index / timestep
	ts2=t_list.index(time_value2)
	
	u = fh.variables['u'][ts1:ts2:time_res,:,1:-1:space_res,::space_res] #zonal wind

	fh.close()

	fh1 = Dataset(datadir + 'v/v_day_ERA5_201101_201112.nc','r')
	v = fh1.variables['v'][ts1:ts2:time_res,:,1:-1:space_res,::space_res] #meridional wind
	fh1.close()

	fh2 = Dataset(datadir + 'temperature/temp_day_ERA5_201101_201112.nc','r')

	temperature = fh2.variables['t'][ts1:ts2:time_res,:,1:-1:space_res,::space_res]
	fh2.close()

	return lats, lons, pressure*100, u, v, temperature, time[ts1:ts2:time_res], day1


"""""""""""""""""""""""""""
		Functions
"""""""""""""""""""""""""""

#compute potential temperature

def potential_temperature(temperature, pressure):
	""" 
 	temperature = temperature. Shape must be [time,pressure,lat,lon] 
	pressure = array with original pressure levels. Check whether original pressure leveles are in PA/hPa!

	returns

	potential_temp = potential temperature. Shape must be [time,pressure,lat,lon] 
	"""
	potential_temp=np.zeros_like(temperature)

	potential_temp[:,:,:,:]=temperature[:,:,:,:]*((pressure[None,:,None,None]/pref)**(-R/cp))
	
	return potential_temp

def remove_underground_isentropes(isentropes_old, pot_temp):
	""" 
	isentropes_old = array with some isentropic levels
	pot_temp = potential temperature. Shape must be [time,pressure,lat,lon] 

	return

	isentropes = new array in which the isentropes which are below the ground are removed
	"""

	ii=0
	for i in range(0,len(isentropes_old)):
		if isentropes_old[i]<np.max(pot_temp[:,-1,:,:]): #if it is true do
			ii+=1

	if ii>0:
		isentropes = isentropes_old[ii:]
	elif ii==0:
		isentropes = isentropes_old
	
	return isentropes

#compute tge horizontal velocities in isentropic coordinates

def velocities_isen(lats, lons, u, v, potential_temp, pressure, isentropes):
	""" 
	u,v = horizontal wind components in pressure levels. Shape must be [time,pressure,lat,lon]
	lons = longitudes from grid
	lats = latitudes from grid 
	potential_temp = potential temperature. Shape must be [time,pressure,lat,lon] 
	pressure = array with original pressure levels
	isentropes = set of isentropes as vertical level (theta).

	returns

	u_isen, v_isen = horizontal wind components on isentropic levels. Shape is [time,theta,lat,lon]
	"""

	noLats = u.shape[2]
	noLons = u.shape[3]
	notime = u.shape[0]

	pisen=np.zeros([notime,len(isentropes),noLats,noLons])
	u_isen=np.zeros([notime,len(isentropes),noLats,noLons])
	v_isen=np.zeros([notime,len(isentropes),noLats,noLons])

	for t in range (0,notime):
		for k in range (0,noLats):
			for h in range (0,noLons):
				pisen[t,:,k,h]=np.interp(isentropes, potential_temp[t,::-1,k,h], pressure[::-1])
				u_isen[t,:,k,h]=np.interp(pisen[t,:,k,h], pressure, u[t,:,k,h])
				v_isen[t,:,k,h]=np.interp(pisen[t,:,k,h], pressure, v[t,:,k,h])

	u_isen=np.where(pisen == pref, 0, u_isen)
	v_isen=np.where(pisen == pref, 0, v_isen)
	return u_isen, v_isen


#compute absolute vorticity on the isentropic surfaces in spherical coordinates using centered differences:

def absolutevorticity(u, v, lons, lats, isentropes):

	""" 
	u,v = horizontal wind components. Shape must be [time,theta,lat,lon]
	lons = longitudes from grid
	lats = latitudes from grid 
	isentropes = set of isentropes as vertical level (theta)
	returns

	abs_vort = absolute vorticity. Shape is the same as u,v [time,theta,lat,lon]
	"""

	noLats = u.shape[2]
	noLons = u.shape[3]
	notime = u.shape[0]

	rel_vort=np.zeros([notime,len(isentropes),noLats,noLons+1])

	u=np.append(u, u[:,:,:,0:2], axis=3) 
	v=np.append(v, v[:,:,:,0:2], axis=3) 
	dlambda=(lons[1]-lons[0])*np.pi/180
	for k in range (1,noLats-1):
		for h in range (1,noLons+1):
			rel_vort[:,:,k,h]= (1/(a*np.cos(lats[k]*np.pi/180)))*(((v[:,:,k,h+1]-v[:,:,k,h-1]))/(2*dlambda)-(u[:,:,k+1,h]*np.cos(lats[k+1]*np.pi/180)-u[:,:,k-1,h]*np.cos(lats[k-1]*np.pi/180))/((lats[k+1]-lats[k-1])*np.pi/180))
	
	#set the boundaries for latitude using FW-BW in space
	for h in range (1,noLons+1):
		rel_vort[:,:,0,h]= (1/(a*np.cos(lats[0]*np.pi/180)))*(((v[:,:,0,h+1]-v[:,:,0,h-1]))/(2*dlambda)-(u[:,:,1,h]*np.cos(lats[1]*np.pi/180)-u[:,:,0,h]*np.cos(lats[0]*np.pi/180))/((lats[1]-lats[0])*np.pi/180))
		rel_vort[:,:,-1,h]= (1/(a*np.cos(lats[-1]*np.pi/180)))*(((v[:,:,-1,h+1]-v[:,:,-1,h-1]))/(2*dlambda)-(u[:,:,-2,h]*np.cos(lats[-2]*np.pi/180)-u[:,:,-1,h]*np.cos(lats[-1]*np.pi/180))/(abs(lats[-2]-lats[-1])*np.pi/180))
	
	#periodic boundaries in longitude
	rel_vort[:,:,:,0]=rel_vort[:,:,:,-1]
	rel_vort=rel_vort[:,:,:,:-1]

	abs_vort=np.zeros_like(rel_vort)
	abs_vort=2*OMEGA*np.sin(lats[None,None,:,None]*np.pi/180)+rel_vort

	return abs_vort


"""""""""""""""""""""""""""""""""
		COMPUTE SIGMA AND PV
"""""""""""""""""""""""""""""""""

# compute isentropic layer density (named sigma) on isentropic surfaces
def compute_sigma (pressure, potential_temp, isentropes):
	""" 
	pressure = array with original pressure levels
	potential_temp = potential temperature; shape is [time,pressure,lat,lon] 
	isentropes = set of isentropes as vertical level (theta)

	returns

	sigmaisen = isentropic layer density in isentropic coordinates. Shape is [time,theta,lat,lon]
	"""

	noLev = potential_temp.shape[1]
	noLats = potential_temp.shape[2]
	noLons = potential_temp.shape[3]
	notime = potential_temp.shape[0]

	dpdtheta=np.zeros([notime,noLev,noLats,noLons])

	pisen=np.zeros([notime,len(isentropes),noLats,noLons])
	sigmaisen=np.zeros([notime,len(isentropes),noLats,noLons])
	dpdtheta_isen=np.zeros([notime,len(isentropes),noLats,noLons], dtype="float64")

	# compute the gradient dtheta/dp in pressure coordinates using centered differences 
	eps=10**-8
	dpdtheta[:,1:-1,:,:] = (pressure[None,2:,None,None]-pressure[None,:-2,None,None])/(potential_temp[:,2:,:,:]-potential_temp[:,:-2,:,:]+eps)
	dpdtheta[:,0,:,:] = (pressure[None,1,None,None]-pressure[None,0,None,None])/(potential_temp[:,1,:,:]-potential_temp[:,0,:,:]+eps)
	dpdtheta[:,-1,:,:] = (pressure[None,-1,None,None]-pressure[None,-2,None,None])/(potential_temp[:,-1,:,:]-potential_temp[:,-2,:,:]+eps)

	# calculate the gradient isentropic coordinates using interpolation

	for t in range (0,notime):
		for k in range (0,noLats):
			for h in range (0,noLons):
				pisen[t,:,k,h]=np.interp(isentropes, potential_temp[t,::-1,k,h], pressure[::-1])	# remember that values in x arrays have to increase
				dpdtheta_isen[t,:,k,h]=np.interp(pisen[t,:,k,h], pressure, dpdtheta[t,:,k,h])   																					

	sigmaisen=dpdtheta_isen*(-1/g)

	# set sigma==0 where the isentrope is under the ground
	sigmaisen=np.where(pisen == pref, 0, sigmaisen)
	# artificial setting for grid points at which sigma is negative (unstable atmosphere) or too large
	sigmaisen=np.where(sigmaisen < 0 , 1000, sigmaisen)
	sigmaisen=np.where(sigmaisen > 1000 , 1000, sigmaisen)

	return sigmaisen

# now compute PV on isentropic surfaces as omega/sigma when sigma is >0
def compute_PV (omega, sigma):
	""" 
	omega = absolute vorticity in isentropic coord. Shape is [time,theta,lat,lon].
	sigma = isentropic layer density in isentropic coord. Shape is [time,theta,lat,lon].

	returns

	PV = potential vorticity in isentropic coord. Shape is [time,theta,lat,lon]
	"""

	PV=np.zeros_like(omega)

	#set sigma to a very large value so PV goes to zero where sigma is 0 too
	#sigma=np.where(sigma == 0 , 10**12, sigma)
	PV=omega/sigma
	#PV=np.where(PV == np.nan , 0, PV)
	
	return PV

#compute the PV contours Q associated to the grid latutudes that are also the equivalent latitudes

def PV_contours_trap(PV, sigma, lons, lats, isentropes):
	""" 
	PV = potential vorticity in isentropic coord. Shape is [time,theta,lat,lon].
	sigma = isentropic layer density in isentropic coord. Shape is [time,theta,lat,lon].
	isentropes = set of isentropes as vertical level (theta)
	lons = longitudes from grid
	lats = latitudes from grid 

	returns

	Q_interp = Array with PV contours associated to GRID latitudes. Shape is [time,theta,lat]
	phi_M = equivalent latitudes. Shape is [time,theta,PV_bins]
	PV_bins = scalar number equals to the number of prescribed PV contours. Here is set to be equal to len(lats)

	NOTE: in this version of the code equivalent latitude are computed just for a matter of interpolation to find Q_interp.
		  grid latitudes are used as eq. latitudes then Q_interp are the Q contours associated to grid lats.
		  version in which the integrals are computed using the trapezoidal rule

	"""

	noLats = PV.shape[2]
	noLons = PV.shape[3]
	notime = PV.shape[0]

	dphi=abs(lats[0]-lats[1])*np.pi/180
	dlambda=(lons[1]-lons[0])*np.pi/180
	
	PV_bins=noLats
	# initialise the integrals for INT 1
	integ = np.zeros([notime,len(isentropes),PV_bins,noLons]) # initialise the integrals to 0
	integral = np.zeros([notime,len(isentropes),PV_bins])
	q = np.zeros([notime,len(isentropes),PV_bins])
	integ1=integ2=integ3=0

	# initialise the integrals for INT 2
	int2 = np.zeros([notime,len(isentropes),PV_bins,noLats])
	phiM = np.zeros([notime,len(isentropes),PV_bins])
	integ_zonmean = 0
	Q_interp=np.zeros([notime,len(isentropes), noLats])

	### INT 1 ###
	for t in range (0,notime):
		for i in range(0, len(isentropes)):
			#prescribed PV contours (in PVU) for each isentropic surface from min to max of PV range of values
			# NOTE: here q are set at each timestep, remove the t dependence if you want Qs to be constant for all times!
			q[t,i]=np.linspace(np.nanmin(PV[t,i,:,:]), np.nanmax(PV[t,i,:,:]), num=PV_bins, endpoint=True)

			for j in range(0, PV_bins):
				for h in range(0,noLons):		
					for k in range(0,noLats-1):
			
						if (PV[t,i,k,h]-q[t,i,j]) > 0 and (PV[t,i,k+1,h]-q[t,i,j]) > 0:  # if PV-q is positive at both grid points use trapezoidal rule to calculate the integral in dphi
							#print 'a'
							integ1 = 0.5*(sigma[t,i,k,h]+sigma[t,i,k+1,h])*(abs((lats[k]-lats[k+1])*np.pi/180))*np.cos((lats[k]+lats[k+1])*np.pi/360)*dlambda*a**2
				
						elif (PV[t,i,k,h]-q[t,i,j])*(PV[t,i,k+1,h]-q[t,i,j]) < 0 and (PV[t,i,k,h]-q[t,i,j])<(PV[t,i,k+1,h]-q[t,i,j]):
							#print 'b'
							phistar2 = np.interp(0, [(PV[t,i,k,h]-q[t,i,j]), (PV[t,i,k+1,h]-q[t,i,j])], [lats[k], lats[k+1]])
							sigmastar = np.interp(phistar2, [lats[k+1], lats[k]], [sigma[t,i,k+1,h], sigma[t,i,k,h]])
							integ2 = 0.5*(sigma[t,i,k+1,h]+sigmastar)*(abs((phistar2-lats[k+1])*np.pi/180))*np.cos((phistar2+lats[k+1])*np.pi/360)*dlambda*a**2

						elif (PV[t,i,k,h]-q[t,i,j])*(PV[t,i,k+1,h]-q[t,i,j]) < 0 and (PV[t,i,k,h]-q[t,i,j])>(PV[t,i,k+1,h]-q[t,i,j]):
							#print 'c'
							phistar1 = np.interp(0, [(PV[t,i,k+1,h]-q[t,i,j]), (PV[t,i,k,h]-q[t,i,j])], [lats[k+1], lats[k]])
							sigmastar = np.interp(phistar1, [lats[k+1], lats[k]], [sigma[t,i,k+1,h], sigma[t,i,k,h]])
							integ3 = 0.5*(sigma[t,i,k,h]+sigmastar)*(abs((lats[k]-phistar1)*np.pi/180))*np.cos((phistar1+lats[k+1])*np.pi/360)*dlambda*a**2
				
						integ[t,i,j,h] += integ1+integ2+integ3
						integ1=integ2=integ3=0
	
				integral[t,i,j]=np.sum(integ[t,i,j,:]) #integrate in dlambda
		
		### INT 2 ###
		# compute zonal mean at half grid points
		sigma_zonal=np.zeros([notime,len(isentropes),noLats])
		for k in range(0,noLats-1):
			sigma_zonal[t,:,k]=np.mean(sigma[t,:,k,:], axis=1)

		for i in range (0, len(isentropes)):
			for j in range(0, PV_bins):
				for k in range(0,noLats):
					if np.sum(int2[t,i,j,:]) < integral[t,i,j]:	
						# compute the eulerian mean then integrate in dphi until int1=int2
						integ_zonmean = 2*np.pi*a**2*sigma_zonal[t,i,k]*dphi*np.cos(lats[k]*np.pi/180)
						int2[t,i,j,k] += integ_zonmean
			
					else:
						break
	
				#interpolate to find the equivalent latitude
				phiM[t,i,j] = np.interp(integral[t,i,j], [np.sum(int2[t,i,j,:(k-1)]), np.sum(int2[t,i,j,:k])], [lats[k-1], lats[k]])
				
				# set the values at the b'ries equal to lats
				phiM[t,i,0]=lats[-1]
				phiM[t,i,-1]=lats[0]
		# now interpolate back to find the q values that corresponds to the original lats at given gridpoints
		
		for i in range (0, len(isentropes)):
			Q_interp[t,i,:]=np.interp(lats[:], phiM[t,i,:], q[t,i,:])

	return Q_interp, PV_bins, phiM

#compute the PV contours Q associated to the grid latutudes that are also the equivalent latitudes

def PV_contours_box(PV, sigma, lons, lats, isentropes):
	""" 
	PV = potential vorticity in isentropic coord. Shape is [time,theta,lat,lon].
	sigma = isentropic layer density in isentropic coord. Shape is [time,theta,lat,lon].
	isentropes = set of isentropes as vertical level (theta)
	lons = longitudes from grid
	lats = latitudes from grid 

	returns

	Q_interp = Array with PV contours associated to GRID latitudes. Shape is [time,theta,lat]
	phi_M = equivalent latitudes. Shape is [time,theta,PV_bins]
	PV_bins = scalar number equals to the number of prescribed PV contours. Here is set to be equal to len(lats)

	NOTE: in this version of the code equivalent latitude are computed just for a matter of interpolation to find Q_interp.
		  grid latitudes are used as eq. latitudes then Q_interp are the Q contours associated to grid lats.
		  version in which the integrals are computed using the rectangular rule (conditional box counting)

	"""

	noLats = PV.shape[2]
	noLons = PV.shape[3]
	notime = PV.shape[0]
	
	dphi=abs(lats[0]-lats[1])*np.pi/180
	dlambda=(lons[1]-lons[0])*np.pi/180
	PV_bins=noLats
	# initialise the integrals for INT 1
	integ = np.zeros([notime,len(isentropes),PV_bins,noLons]) # initialise the integrals to 0
	integral = np.zeros([notime,len(isentropes),PV_bins])
	q = np.zeros([notime,len(isentropes),PV_bins])
	integ1=0

	# initialise the integrals for INT 2
	int2 = np.zeros([notime,len(isentropes),PV_bins,noLats])
	phiM = np.zeros([notime,len(isentropes),PV_bins])
	integ_zonmean = 0
	Q_interp=np.zeros([notime,len(isentropes), noLats])

	### INT 1 ###
	for t in range (0,notime):
		for i in range(0, len(isentropes)):
			#prescribed PV contours (in PVU) for each isentropic surface from min to max of PV range of values
			# NOTE: here q are set at each timestep, remove the t dependence if you want Qs to be constant for all times!
			q[t,i]=np.linspace(np.nanmin(PV[t,i,:,:]), np.nanmax(PV[t,i,:,:]), num=PV_bins, endpoint=True)

			for j in range(0, PV_bins):
				for h in range(0,noLons):		
					for k in range(0,noLats):
			
						if (PV[t,i,k,h]-q[t,i,j]) > 0:
							#print 'a'
							integ1 = sigma[t,i,k,h]*a**2*dphi*dlambda*np.cos(lats[k]*np.pi/180)
				
						integ[t,i,j,h] += integ1
						integ1=0
	
				integral[t,i,j]=np.sum(integ[t,i,j,:]) #integrate in dlambda
		
		### INT 2 ###
		# compute zonal mean 
		sigma_zonal=np.zeros([notime,len(isentropes),noLats])
		for k in range(0,noLats):
			sigma_zonal[t,:,k]=np.mean(sigma[t,:,k,:], axis=1)

		for i in range (0, len(isentropes)):
			for j in range(0, PV_bins):
				for k in range(0,noLats-1):
					if np.sum(int2[t,i,j,:]) < integral[t,i,j]:	
						# compute the eulerian mean then integrate in dphi until int1=int2
						integ_zonmean = 2*np.pi*a**2*sigma_zonal[t,i,k]*dphi*np.cos(lats[k]*np.pi/180)
						int2[t,i,j,k] += integ_zonmean
			
					else:
						break
	
				#interpolate to find the equivalent latitude
				phiM[t,i,j] = np.interp(integral[t,i,j], [np.sum(int2[t,i,j,:(k-1)]), np.sum(int2[t,i,j,:k])], [lats[k-1], lats[k]])

			phiM[t,i,0]=lats[-1]
			phiM[t,i,-1]=lats[0]
		# now interpolate back to find the q values that corresponds to the original lats at given gridpoints
		
		for i in range (0, len(isentropes)):
			Q_interp[t,i,:]=np.interp(lats[:], phiM[t,i,:], q[t,i,:])

	return Q_interp, PV_bins, phiM



# compute wave activity

# lats = array with latitudes at grid points that are also the equivalent latitudes (noLats)
# Q_interp = array with the corresponding PV contours for each isentropic surface (time,noLev,noLats)
# sigma = isentropic layer density (time, noIsentropes,noLats, noLons)

def local_wave_activity_trap(isentropes, omega, sigma, lons, lats, Q, PV , PV_bins):

	""" 
	isentropes = set of isentropes as vertical level (theta)
	omega = absolute vorticity in isentropic coord. Shape is [time,theta,lat,lon].
	sigma = isentropic layer density in isentropic coord. Shape is [time,theta,lat,lon].
	lons = longitudes from grid
	lats = latitudes from grid 
	Q = prescribed PV contours. Shape must be [time,theta,PV_bins]
	PV = potential vorticity in isentropic coord. Shape must be [time,theta,lat,lon]
	PV_bins= scalar number equals to the number of prescribed PV contours

	returns
	integ = Local Wave activity (LWA). Shape is [time,theta,lat,lon]
	
	NOTE: in this version of the code LWA is computed setting eq lats = grid lats
		  version in which the integrals are computed using the trapezoidal rule
	"""
	noLats = PV.shape[2]
	noLons = PV.shape[3]
	notime = PV.shape[0]

	phiM=lats #since eq lats and grid lats are coincident
	
	integ1=integ2=integ3=integ4=integ5=integ6=0

	integ = np.zeros([notime,len(isentropes),noLats,noLons])

	for t in range (0, notime):
		for i in range (0, len(isentropes)):
			for j in range (0,PV_bins): #loop for equivalent latitudes, remeber lats=eq lats
				for h in range(0,noLons):		
					for k in range(0,noLats-1): #physical lat
			
						# compute the integrals on the area south to Phi_M									
			
						if  lats[k] <= lats[j] and (PV[t,i,k,h]-Q[t,i,j]) > 0 and (PV[t,i,k+1,h]-Q[t,i,j]) > 0 :  
							#print 'a'
							integ1 = (1/np.cos(phiM[j]*np.pi/180))*0.5*((sigma[t,i,k,h]*(PV[t,i,k,h]-Q[t,i,j]))+(sigma[t,i,k+1,h]*(PV[t,i,k+1,h]-Q[t,i,j])))*(abs((lats[k]-lats[k+1])*np.pi/180))*a*np.cos((lats[k]+lats[k+1])*np.pi/360)
				
						if lats[k] <= lats[j] and (PV[t,i,k,h]-Q[t,i,j])*(PV[t,i,k+1,h]-Q[t,i,j]) < 0 and (PV[t,i,k,h]-Q[t,i,j])<(PV[t,i,k+1,h]-Q[t,i,j]):
							#print 'b'
							phistar = np.interp(0, [(PV[t,i,k,h]-Q[t,i,j]), (PV[t,i,k+1,h]-Q[t,i,j])], [lats[k], lats[k+1]])
							#sigmastar = np.interp(phistar, [lats[k+1], lats[k]], [sigma[i,k+1,h], sigma[i,k,h]])
							integ2 = (1/np.cos(phiM[j]*np.pi/180))*0.5*(sigma[t,i,k+1,h]*(PV[t,i,k+1,h]-Q[t,i,j]))*(abs((phistar-lats[k+1])*np.pi/180))*a*np.cos((lats[k]+lats[k+1])*np.pi/360)

						if lats[k] <= lats[j] and (PV[t,i,k,h]-Q[t,i,j])*(PV[t,i,k+1,h]-Q[t,i,j]) < 0 and (PV[t,i,k,h]-Q[t,i,j])>(PV[t,i,k+1,h]-Q[t,i,j]):
							#print 'c'
							phistar = np.interp(0, [(PV[t,i,k+1,h]-Q[t,i,j]), (PV[t,i,k,h]-Q[t,i,j])], [lats[k+1], lats[k]])
							#sigmastar = np.interp(phistar, [lats[k+1], lats[k]], [sigma[i,k+1,h], sigma[i,k,h]])
							integ3 = (1/np.cos(phiM[j]*np.pi/180))*0.5*(sigma[t,i,k,h]*(PV[t,i,k,h]-Q[t,i,j]))*(abs((lats[k]-phistar)*np.pi/180))*a*np.cos((lats[k]+lats[k+1])*np.pi/360)
				
						# then compute the integrals on the area north to Phi_M

				
						if  lats[k] > lats[j] and (PV[t,i,k,h]-Q[t,i,j]) <= 0 and (PV[t,i,k+1,h]-Q[t,i,j]) <= 0 :
							#print 'd'
							integ4 = (1/np.cos(phiM[j]*np.pi/180))*0.5*((sigma[t,i,k,h]*(Q[t,i,j]-PV[t,i,k,h]))+(sigma[t,i,k+1,h]*(Q[t,i,j]-PV[t,i,k+1,h])))*(abs((lats[k]-lats[k+1])*np.pi/180))*a*np.cos((lats[k]+lats[k+1])*np.pi/360)
					
						if  lats[k] > lats[j] and (Q[t,i,j]-PV[t,i,k,h])*(Q[t,i,j]-PV[t,i,k+1,h]) < 0 and (Q[t,i,j]-PV[t,i,k,h])<=(Q[t,i,j]-PV[t,i,k+1,h]):
							#print 'e'
							phistar = np.interp(0, [(Q[t,i,j]-PV[t,i,k,h]), (Q[t,i,j]-PV[t,i,k+1,h])], [lats[k], lats[k+1]])
							#sigmastar = np.interp(phistar, [lats[k+1], lats[k]], [sigma[i,k+1,h], sigma[i,k,h]])
							integ5 = (1/np.cos(phiM[j]*np.pi/180))*0.5*(sigma[t,i,k+1,h]*(Q[t,i,j]-PV[t,i,k+1,h]))*(abs((phistar-lats[k+1])*np.pi/180))*a*np.cos((lats[k]+lats[k+1])*np.pi/360)
					
						if  lats[k] > lats[j] and (Q[t,i,j]-PV[t,i,k,h])*(Q[t,i,j]-PV[t,i,k+1,h]) < 0 and (Q[t,i,j]-PV[t,i,k,h])>=(Q[t,i,j]-PV[t,i,k+1,h]):
							#print 'f'
							phistar = np.interp(0, [(Q[t,i,j]-PV[t,i,k+1,h]), (Q[t,i,j]-PV[t,i,k,h])], [lats[k+1], lats[k]])
							#sigmastar = np.interp(phistar, [lats[k+1], lats[k]], [sigma[i,k+1,h], sigma[i,k,h]])
							integ6 = (1/np.cos(phiM[j]*np.pi/180))*0.5*(sigma[t,i,k,h]*(Q[t,i,j]-PV[t,i,k,h]))*(abs((lats[k]-phistar)*np.pi/180))*a*np.cos((lats[k]+lats[k+1])*np.pi/360)
	
						integ[t,i,j,h] += integ1+integ2+integ3+integ4+integ5+integ6
				
						integ1=integ2=integ3=integ4=integ5=integ6=0

	#return the integral which is the local FAWA [time,theta,lat,lon]
	return integ 

# lats = array with latitudes at grid points that are also the equivalent latitudes (noLats)
# Q_interp = array with the corresponding PV contours for each isentropic surface (time,noLev,noLats)
# sigma = isentropic layer density (time, noIsentropes,noLats, noLons)

def local_wave_activity_box(isentropes, omega, sigma, lons, lats, Q, PV , PV_bins):

	""" 
	isentropes = set of isentropes as vertical level (theta)
	omega = absolute vorticity in isentropic coord. Shape is [time,theta,lat,lon].
	sigma = isentropic layer density in isentropic coord. Shape is [time,theta,lat,lon].
	lons = longitudes from grid
	lats = latitudes from grid 
	Q = prescribed PV contours. Shape must be [time,theta,PV_bins]
	PV = potential vorticity in isentropic coord. Shape must be [time,theta,lat,lon]
	PV_bins= scalar number equals to the number of prescribed PV contours

	returns
	integ = Local Wave activity (LWA). Shape is [time,theta,lat,lon]
	
	NOTE: in this version of the code LWA is computed setting eq lats = grid lats
		  version in which the integrals are computed using the rectangular rule (conditional box counting)
	"""
	noLats = PV.shape[2]
	noLons = PV.shape[3]
	notime = PV.shape[0]
	dphi=abs(lats[0]-lats[1])*np.pi/180

	phiM=lats #since eq lats and grid lats are coincident
	
	integ1=integ2=0

	integ = np.zeros([notime,len(isentropes),noLats,noLons])

	for t in range (0, notime):
		for i in range (0, len(isentropes)):
			for j in range (0,PV_bins): #loop for equivalent latitudes, remeber lats=eq lats
				for h in range(0,noLons):		
					for k in range(0,noLats): #physical lat
			
						# compute the integrals on the area south to Phi_M									
			
						if  lats[k] <= lats[j] and (PV[t,i,k,h]-Q[t,i,j]) > 0 :  
							#print 'a'
							integ1 = (1/np.cos(phiM[j]*np.pi/180))*(sigma[t,i,k,h]*(PV[t,i,k,h]-Q[t,i,j]))*dphi*a*np.cos(lats[k]*np.pi/180)
				
						
						# then compute the integrals on the area north to Phi_M

				
						if  lats[k] > lats[j] and (PV[t,i,k,h]-Q[t,i,j]) <= 0:
							#print 'd'
							integ2 = (1/np.cos(phiM[j]*np.pi/180))*(sigma[t,i,k,h]*(Q[t,i,j]-PV[t,i,k,h]))*dphi*a*np.cos(lats[k]*np.pi/180)
					
					
	
						integ[t,i,j,h] += integ1+integ2
				
						integ1=integ2=0

	#return the integral which is the local FAWA [time,theta,lat,lon]
	return integ 

"""""""""""""""""""""""""""
		Plotting
"""""""""""""""""""""""""""
def plotPV(PV, isentropes, lats, lons, day1, time):

	print ("Producing and saving the PV plots")

	levels=[-8,-6,-4,-2,-1,-0.5,0.5,1,2,4,6,8]

	dt_day=(time[1]-time[0])/24

	# add cyclic point at lon 360
	PV, lons = add_cyclic_point(PV, coord=lons)

	for i in range (0, len (isentropes)):
		validtime=day1
		for j in range (0,len(time)):

			ax = plt.axes(projection=ccrs.PlateCarree())

			ax.set_global()

			cs=plt.contourf(lons, lats, PV[j,i,:,:]*10**6, levels, cmap=plt.get_cmap('plasma'), extend ="both")

			ax.coastlines()
			ax.gridlines()

			# Add Colorbar
			cb = plt.colorbar(cs, orientation='horizontal', ticks=levels)#, labelsize=18)
			cb.ax.tick_params(labelsize=14)
			cb.set_label("PVU", fontsize=16)

			# Add Title
			plt.title('PV at %s K - %s UTC %s' %(int(isentropes[i]), validtime.strftime("%H%M"), validtime.strftime("%d %b %Y")), fontsize=12)

			plt.savefig(plotdir + 'PV/PV_%sK_%sh.png' %(int(isentropes[i]), "%03d" %int(time[j])), bbox_inches='tight', dpi=350)
		
			plt.close()
			
			validtime+=timedelta(days=dt_day)

	print ('figure saved')

def plotLWA(A, isentropes, lats, lons, day1, time, title, name_savefig):

	print(("Producing and saving the", title, "plots"))

	levels=[0,50,75,100,150,200]

	dt_day=(time[1]-time[0])/24

	# add cyclic point at lon 360
	A, lons = add_cyclic_point(A, coord=lons)

	for i in range (0, len (isentropes)):
		validtime=day1
		for j in range (0,len(time)):

			ax = plt.axes(projection=ccrs.PlateCarree())

			ax.set_global()

			cs=plt.contourf(lons, lats, A[j,i,:,:], levels, cmap=colormapLWA(), extend='max')

			ax.coastlines()
			ax.gridlines()

			# Add Colorbar
			cb = plt.colorbar(cs, orientation='horizontal')#, labelsize=18)
			cb.ax.tick_params(labelsize=14)
			cb.set_label(r"m s$^{-1}$", fontsize=16)

			# Add Title
			plt.title(title + ' at %s K - %s UTC %s' %(int(isentropes[i]), validtime.strftime("%H%M"), validtime.strftime("%d %b %Y")), fontsize=12)

			plt.savefig(plotdir + 'LWA/' + name_savefig +'_%sK_%sh.png' %(int(isentropes[i]), "%03d" %int(time[j])), bbox_inches='tight', dpi=350)
		
			plt.close()
			
			validtime+=timedelta(days=dt_day)

	print ('figure saved')


"""""""""""""""""""""""""""
		MAIN
"""""""""""""""""""""""""""

def main():
		
	# set the isentropic levels 
	# ideally it should intersect the tropopause in the midlatitudes

	isentropes_old = [325]

	# get the data from file #
	"""
	index of array variables u, v, temperature are [time,pressure,lat,lon]
	specify the start and end timestep in the format yr,month,date,time
	sample file contanis 10-14 Apr 2011 with 6 hourly data on a 1x1 degree grid
	"""
	lats, lons, pressure, u, v, temperature, time, day1 = read_data(space_res=2, time_res=1, y1=2011, m1=4, d1=10, t1=00, y2=2011, m2=4, d2=14, t2=00) 

	potential_temp=potential_temperature(temperature, pressure)
	isentropes = remove_underground_isentropes(isentropes_old, potential_temp)

	u_isen, v_isen = velocities_isen(lats, lons, u, v, potential_temp, pressure, isentropes)
	omega = absolutevorticity(u_isen, v_isen, lons, lats, isentropes)
	sigma = compute_sigma(pressure, potential_temp, isentropes)
	PV = compute_PV (omega, sigma)
	Q_interp, PV_bins, phiM = PV_contours_trap(PV, sigma, lons, lats, isentropes) 

	A = local_wave_activity_trap(isentropes, omega, sigma, lons, lats, Q_interp, PV, PV_bins)

	# zonal filtering: choose the filter you prefer #

	filter_type = "fourier"

	"""  
	method where the dominant zonal wavenumber is computed with fourier analysis at a latitude circle 
	in this way the filter is linear in longitude and commutes with the gradient operator. This property
	is useful to preserve tthe conservation of LWA in its flux form (local E-P relation).

	"""

	if filter_type=="fourier":

		maxWNRunMean = zonalWN_fourier(v_isen, lats, lons) #dominant zonal wavenumber with Fourier analysis using meridional wind on isentropes
		smoothA = Hann_convolution(A, lats, lons, maxWNRunMean, calibration = 1)

		"""  
		method where the local zonal wavenumber k is computed with wavelet analysis
		and the width of Hann window depends on k. Note that the filtering operator is non linear and		
		does not commute with the gradient operator! 
		"""
	
	elif filter_type=="wavelet":

		k_2D = dom_wavenumber_2D(v_isen, lons)	#LOCAL zonal wavenumber with wavelet analysis using meridional wind on isentropes
		smoothA = HannSmoothing_time_2D (A, lons, k_2D)

	else:

		print("filter type not recognized!")

	# plotting #

	#plotPV(PV, isentropes, lats, lons, day1, time)
	plotLWA(A, isentropes, lats, lons, day1, time, title='LWA', name_savefig='LWA')
	plotLWA(smoothA, isentropes, lats, lons, day1, time, title='Filtered LWA', name_savefig='filtLWA')


runthisprogram=main()
