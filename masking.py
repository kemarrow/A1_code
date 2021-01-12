"""
Created on Mon Dec  7 11:05:33 2020

Overlays a mask onto the FITS file by creating circular and rectangular arrays
of zeros, then writes the results into a new FITS file.

@author: Erin
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

hdulist = fits.open("\A1_mosaic.fits")
data = hdulist[0].data
hdulist.close()

def rectangle_mask(alldata,y0, y1, x0, x1):
    alldata[x0:x1, y0:y1] = 0
    return alldata

def circular_mask(alldata, y0,x0, radius):
    x = np.arange(0, np.shape(alldata)[0], 1)
    y = np.arange(0, np.shape(alldata)[1], 1)
    for l in x:
        for m in y:
            if (l - x0)**2 + (m - y0)**2 < radius**2:
                alldata[l,m] = 0
    return alldata

def save_new_fits(data):
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    hdul.writeto(fp+r'\masked.fits')

circles = [[1436, 3214, 350], [775, 3319, 70], [558, 4095, 40],
           [905, 2284, 50], [2089, 1424, 80], [2133, 3758, 40],
           [2466, 3412, 20], [980, 2772, 53], [1455, 4032, 20],
           [2131, 2308, 40]]

rectangles = [[0, 100, 0, 4611], [2470, 2570, 0, 4611], [0, 2570, 0, 100],
         [0, 2570, 4511, 4611], [1198, 1654, 426, 490],
         [1000, 1686, 314, 362], [1390, 1486, 218, 260],
         [1310, 1545, 114, 170], [1134, 1654, 0, 50],
         [1418, 1454, 3236, 4315], [1418, 1454, 4315, 4610],
         [1418, 1454, 1977, 3200], [1420, 1454, 0, 1977],
         [900, 910, 2220, 2358], [767, 784, 3200, 3416],
         [960, 990, 2700, 2840], [2125, 2140, 2284, 2334],
         [2460, 2470, 3382, 3440], [2127, 2140, 3705, 3805],
         [554, 563, 4080, 4118], [1100,1200,420,430]]


for i in circles:
    data = circular_mask(data, i[0], i[1], i[2])

for i in rectangles:
    data = rectangle_mask(data, i[0], i[1], i[2], i[3])


masked = np.ma.masked_where(data < 10, data) #1st argument (data <10) is the condition to mask --> can be changed

plt.figure()

plt.imshow(masked)
plt.gca().invert_yaxis()
plt.show()

# if you want to save file, activate this line:
# save_new_fits(data)
