# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:05:33 2020

creates array of all  pixel values in the image
Plots histogram of the background and fits a Gaussian to it to find
the mean and standard deviation of the background noise

@author: Katherine and Erin
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

#open file
hdulist = fits.open("A1_mosaic.fits")
imdata = hdulist[0].data
hdulist.close()

def gaussian(x, A, mu, sigma):
    "Gaussian function"
    return A*(1 / (np.sqrt(2 * np.pi)*sigma) *
            np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))

data_bits = []
for l in imdata:
    for k in l:
        data_bits.append(k)

no_bright_data = []
for i in data_bits:
    if i <= 3700: #only interested in background
        no_bright_data.append(i)


plt.figure()

#plot Histogram of the background
hist = plt.hist(no_bright_data, bins=700, color='b')
counts, intensity_bins = hist[0], hist[1]
bin_centre = (intensity_bins[:-1] + intensity_bins[1:]) / 2

#fit Gaussian to background
bins = np.linspace(3200, 3700, 501)
p, cov = curve_fit(gaussian, bin_centre, counts, p0=[20000000, 3421, 10])
plt.plot(bins, gaussian(bins, *p), 'r-', label='Gaussian fit')

#extract parameters of the Gaussian fit
print('Amplitude=', p[0], 'Mean=', p[1], 'Sigma=', p[2])
plt.xlabel("Pixel value")
plt.ylabel("Number of pixels")
plt.legend()
plt.show()
