# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:01:33 2020

@author: Katherine and Erin
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.optimize import curve_fit
from astropy.io import fits

def sersic(R, Ie, Re, n):
    """ Ie is intensity at half light radius Re; Ie, Re are constant parameters. """
    if Re > 0 and n > 0:
        b = (2*n)-1/3
        I = Ie * np.exp(-b * (((R)/Re)**(1/n)) - 1)
    else:
        I = 0
    return I

def moffat(r, A, beta, alpha):
    return A*2*(beta-1)/alpha**2 *(1+((r)**2/alpha**2))**(-beta)

# load original and masked image data

hdulist = fits.open("masked.fits")
hdulist0 = fits.open("A1_mosaic.fits")

original = hdulist0[0].data
imdata = hdulist[0].data
#create a second copy because some opencv methods are destructive
imdata2 = hdulist[0].data
imdata2 = imdata2.astype(np.float64)
hdr = hdulist[0].header #header data contains the info about the file
hdulist.close()
#%%
# plot original image with adjusted colour scale so more features can be seen
fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharex=True, sharey=True)
ax1.imshow(original, cmap='gray', vmin=3300,vmax=4000)
ax1.set_xlabel('Original')

# Apply median blur to reduce salt and pepper noise in the image
median = cv2.medianBlur(imdata,3)

# threshold image to identify positions of sources
# creates a binary image suitable for countour detection
threshold = cv2.threshold(median,3456,255, cv2.THRESH_BINARY)[1]
threshold = threshold.astype(np.uint8)
ax2.imshow(threshold, cmap='gray')
ax2.set_xlabel('Thresholded')

#detect contours (outlines of each shape)
contour, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
blank = np.zeros((4611,2570))
outlines = cv2.drawContours(blank, contour,-1,(255,255,255),1)
ax3.imshow(outlines)
ax3.set_xlabel('Contours')
#%%
resultlist = []
galaxylist = []
p=0
"""We have identified all the stars/ galaxies and have a list of contours,
each containing all the points bounding the star/galaxy. We next need to find
the flux and magnitude, categorise as star/galaxy and append to catalogue"""


for i in range(0, len(contour)):
    # Generate a list of points on the contour as an nx2 array so
    # that we can use slices
    arrs=np.array(contour[i])
    sh=arrs.shape
    arrs.shape=(sh[0],sh[2])

    # maximum and minimum extent of each contour
    xmax = max(arrs[:,0])
    xmin = min(arrs[:,0])
    ymax = max(arrs[:,1])
    ymin = min(arrs[:,1])

    """Estimate background intensity. We assume that this represents light
    bleed, e.g. for a terrestrial telescope there will be light reflected off
    dust, water droplets, etc. in the air. The approach is to form a square 10
    pixels outside the galaxy and take the median intensity of the pixels on
    that square. We deduct this from the brightness values for the galaxy."""
    background = []
    xleft = xmin-10
    xright = xmax+10
    yleft = ymin-10
    yright = ymax+10
    for m in arrs[:,1]:
        background.append(imdata2[m,xleft])
        background.append(imdata2[m,xright])
    for n in arrs[:,0]:
        background.append(imdata2[yleft,n])
        background.append(imdata2[yright,n])
    median_background = np.median(background)

    xcoords = []
    ycoords= []
    for l in range(xmin,xmax+1):
        indices = np.where(arrs[:,0]==l)
        #print(indices)
        if arrs[:,1][indices].size > 0:
            left = min(arrs[:,1][indices])
            right = max(arrs[:,1][indices])
            yvals = np.arange(left,right+1,1)
            xvals = np.full(len(yvals),l)
            ycoords.append(yvals)
            xcoords.append(xvals)
    if (len(ycoords) > 0) and (len(xcoords) > 0):
        yarr = np.concatenate(ycoords)
        xarr = np.concatenate(xcoords)
        xcentre = sum(xarr*imdata2[yarr, xarr])/sum(imdata2[yarr, xarr])
        ycentre = sum(yarr*imdata2[yarr, xarr])/sum(imdata2[yarr, xarr])

    lux = 0
    pixels = 0
    intensity = []
    r = []

    # finds all the pixels within each contour, the total intensity, and the
    # intensity and radius of each pixel
    for j in range(xmin,xmax+1):
        mask = np.in1d(arrs[:,0], j)
        subarr = arrs[mask]
        for k in range(min(subarr[:,1]),max(subarr[:,1])+1):
            lux += int(imdata2[k,j])
            pixels += 1
            intensity.append(imdata2[k,j]-median_background)
            r.append(np.sqrt( (j - xcentre)**2 + (k - ycentre)**2 ))

    lux_subtracted = lux - median_background*pixels

    r = np.array(r)
    intensity = np.array(intensity)
    profile = 'unsure'

    #sersic and moffat fits to identify as galaxy or star
    if len(r) > 10:
        try:
            guess_a = np.mean(intensity)
            guess_r = np.mean(r)
            p_sersic, cov_sersic = curve_fit(sersic, r, intensity,
                                             p0 = [guess_a,guess_r,10], maxfev=10000)
        except RuntimeError:
            print('no sersic fit')
            p_sersic = [0,2,0]

        try:
            p_moffat, cov_moffat = curve_fit(moffat,  r,  intensity,
                                             p0 = [2,0,3], maxfev = 10000)
        except RuntimeError:
            print("no moffat fit")
            p_moffat = [0,0,3]

        fitintensity_moffat = moffat(r, p_moffat[0], p_moffat[1], p_moffat[2])
        delta_intensity_moffat =  intensity - fitintensity_moffat
        GoF_moffat = sum(delta_intensity_moffat**2)

        fitintensity_sersic =sersic(r, p_sersic[0], p_sersic[1], p_sersic[2])
        delta_intensity_sersic =  intensity - fitintensity_sersic
        GoF_sersic = sum(delta_intensity_sersic**2)


        if GoF_sersic > GoF_moffat:
            profile = 'star'
            print('star')

        if GoF_sersic < GoF_moffat:
            profile = 'galaxy'
            print('galaxy')

        if GoF_sersic == GoF_moffat:
            profile = 'unsure'
            print('equal')

    resultlist.append([xcentre, ycentre, lux, pixels, median_background,
                       lux_subtracted, profile])

    # find magnitude for galaxies
    if profile == 'galaxy' and lux_subtracted > 0:
        p += 1
        magnitude = 25.3 - (2.5*np.log10(lux_subtracted))
        magerr = (2.5/np.log(10))*(1/np.sqrt(lux_subtracted))
        galaxylist.append([xcentre, ycentre, lux, pixels, median_background,
                           lux_subtracted, magnitude, magerr, p_sersic[0],
                           p_sersic[1], p_sersic[2]])


resultarr = np.concatenate(resultlist)
resultarr.shape=(i+1,7)

galaxyarr = np.concatenate(galaxylist)
galaxyarr.shape=(p,11)

np.savetxt('results.csv', resultarr,
            fmt='%s', delimiter=',')
np.savetxt('galaxies.csv', galaxyarr,
            fmt='%s', delimiter=',')
