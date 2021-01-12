# A1_code

This contains the following .py files:
- background.py which creates an array of all the pixel values in the image, then plots a histogram of the background and fits a Gaussian to it to find the mean and standard deviation of the background noise
- masking.py which allows you to mask over unwanted areas of the image and create a new masked FITS file
- find_galaxies.py which takes the masked image, identifies the galaxies and stars in the image and returns two catalogues, one of all the sources and one of only the galaxies.
- simulation_test.py which tests the galaxy identification code on galaxies and stars simulated using astropy's Sersic2D and Moffat2D models
- number_count.py which plots the logarithm of the number count, N(m) as a function of magnitude, m, and then fits a straight line to it

