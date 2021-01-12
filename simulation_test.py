# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 12:35:59 2020

Simulation in order to test the galaxy/stars identification process

@author: Katherine
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from astropy.modeling.functional_models import Sersic2D, Moffat2D
from scipy.optimize import curve_fit

def sersic(R, Ie, Re, n):
    """ Ie is intensity at half light radius Re; Ie, Re are constant parameters. """
    if Re > 0 and n > 0:
        b = (2*n)-1/3
        I = Ie * np.exp(-b * (abs(R/Re)**(1/n) - 1) )
    else:
        I = 0
    return I

def moffat(r, A, beta, alpha):
    return A*2*(beta-1)/alpha**2 *(1+(r**2/alpha**2))**(-beta)

params = {
   'axes.labelsize':20,
   'axes.labelpad':7,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'font.family': "Times New Roman",
   'legend.fontsize': 15,
   'axes.titlesize':10,
   'errorbar.capsize':0,
   'axes.formatter.use_mathtext': True
}
plt.rcParams.update(params)

b = np.random.rand(4611,2570)
# fig,[ax1,ax2,ax3] = plt.subplots(1,3, sharex=True, sharey=True)
a = np.random.rand(4611,2570)

#creating artificial galaxies (g) and stars (s) with different parameters
g1 = Sersic2D(amplitude = 1000, r_eff = 30, n=4, x_0=500, y_0=2000,
               ellip=.5, theta=-1)
g2 = Sersic2D(amplitude = 150, r_eff = 20, n=4, x_0=1000, y_0=2000,
               ellip=.7, theta=-6)
g3 = Sersic2D(amplitude = 5000, r_eff = 20, n=4, x_0=2000, y_0=2000,
               ellip=0, theta=-90)
s1 = Moffat2D(amplitude = 1000,x_0=250,y_0=4000,gamma=20,alpha=1)
s2 = Moffat2D(amplitude = 130,x_0=2100,y_0=4000,gamma=20,alpha=1)
s3 = Moffat2D(amplitude = 5000,x_0=1100,y_0=4000,gamma=20,alpha=1)
x,y = np.meshgrid(np.arange(2570), np.arange(4611))
a = a + g1(x,y) + g2(x,y) + g3(x,y) + s1(x,y) + s2(x,y) + s3(x,y)
a2 = a + g1(x,y) + g2(x,y) + g3(x,y) + s1(x,y) + s2(x,y) + s3(x,y)
a = a.astype(np.float64)
a2 = a2.astype(np.float64)


#threshold image to remove background noise
a_threshold = cv2.threshold(a,100,255, cv2.THRESH_BINARY)[1]
a_threshold = a_threshold.astype(np.uint8)

#contour detection
contour, hierarchy = cv2.findContours(a_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = cv2.drawContours(b, contour,-1,(255,255,255),3)

#same process as find_galaxy.py module
resultlist = []
x_star=[]
y_star=[]
x_galaxy=[]
y_galaxy=[]

for i in range(0, len(contour)):
    # Generate a list of points on the contour as an nx2 array so
    # that we can use slices
    arrs=np.array(contour[i])
    sh=arrs.shape
    arrs.shape=(sh[0],sh[2])

    xmax = max(arrs[:,0])
    xmin = min(arrs[:,0])
    ymax = max(arrs[:,1])
    ymin = min(arrs[:,1])

    # calculating local background
    background = []
    xleft = xmin-10
    xright = xmax+10
    yleft = ymin-10
    yright = ymax+10
    for m in arrs[:,1]:
        background.append(a2[m,xleft])
        background.append(a2[m,xright])
    for n in arrs[:,0]:
        background.append(a2[yleft,n])
        background.append(a2[yright,n])
    median_background = np.median(background)

    # centre of object
    xcentre = int((xmax+xmin)/2)
    ycentre = int((ymax+ymin)/2)

    lux = 0
    pixels = 0
    intensity = []
    r = []

    for j in range(xmin,xmax+1):
        mask = np.in1d(arrs[:,0], j)
        subarr = arrs[mask]
        for k in range(min(subarr[:,1]),max(subarr[:,1])+1):
            lux += int(a2[k,j])
            pixels += 1
            if a2[k,j] < 1e4:
                intensity.append(a2[k,j]-median_background)
                r.append(np.sqrt( (j - xcentre)**2 + (k - ycentre)**2 ))

    r = np.array(r)
    intensity = np.array(intensity)
    profile = 'unsure'
    
    #fitting the pixel clusters
    if len(r) > 10:
        try:
            guess_a = np.median(intensity)
            guess_r = np.median(r)
            p_sersic, cov_sersic = curve_fit(sersic, r, intensity, p0 = [guess_a,guess_r,1], maxfev = 10000)
        except RuntimeError:
            print('no sersic fit')
            p_sersic = [0,1,4]

        try:
            guess_a = np.median(intensity-median_background)
            p_moffat, cov_moffat = curve_fit(moffat,  r,  intensity, p0 = [100,10,1], maxfev = 10000)
        except RuntimeError:
            print("no moffat fit")
            p_moffat = [0,0,3]

        u = np.linspace(min(r), max(r), 500)
        fig, ax = plt.subplots(1,1)

        #plotting the fits
        ax.plot(r, intensity, 'x', label='data')
        ax.plot(u, moffat(u, p_moffat[0], p_moffat[1], p_moffat[2]), label ='Moffat')
        ax.plot(u, sersic(u, p_sersic[0], p_sersic[1], p_sersic[2]), label ='Sersic')
        ax.legend()
        ax.grid()
        fitintensity_moffat = moffat(r, p_moffat[0], p_moffat[1], p_moffat[2])
        delta_intensity_moffat =  intensity - fitintensity_moffat
        GoF_moffat = sum(delta_intensity_moffat**2)

        fitintensity_sersic =sersic(r, p_sersic[0], p_sersic[1], p_sersic[2])
        delta_intensity_sersic =  intensity - fitintensity_sersic
        GoF_sersic = sum(delta_intensity_sersic**2)

        if GoF_sersic > GoF_moffat:
            profile = 'star'
            x_star.append(arrs[:,0])
            y_star.append(arrs[:,1])
            print('star')

        if GoF_sersic < GoF_moffat:
            profile = 'galaxy'
            x_galaxy.append(arrs[:,0])
            y_galaxy.append(arrs[:,1])
            print('galaxy')

        if GoF_sersic == GoF_moffat:
            profile = 'unsure'
            print('equal')

    resultlist.append([xcentre, ycentre, lux, pixels, profile])

#plotting the contours of the stars and galaxies on the image
fig, (ax1,ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)
ax1.imshow(np.log(a), cmap='gray')
ax1.set_xlabel('image (log scale)')

ax2.imshow(a_threshold, cmap='gray')
ax2.set_xlabel('threshold')

ax3.imshow(np.log(a), cmap='gray')
ax3.plot(np.concatenate(x_galaxy),np.concatenate(y_galaxy), '.', ms=0.2, label = 'galaxy', color='g')
ax3.plot(np.concatenate(x_star),np.concatenate(y_star), '.', ms=0.2, label = 'star', color = 'yellow')
ax3.axis('scaled')
ax3.legend( bbox_to_anchor=(1.02, 0), loc='lower left')
ax3.set_xlabel('galaxies and stars')
plt.show()
