# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:51:46 2020

This takes the galaxy catalogue and plots log(N(m)) versus m

@author: Katherine
"""

import numpy as np
import matplotlib.pyplot as plt

params = {
    'axes.labelsize':20,
    'axes.labelpad':7,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'font.family': "Times New Roman",
    'legend.fontsize': 15,
    'axes.titlesize':10,
    'errorbar.capsize':0,
    'axes.formatter.use_mathtext': True
}
plt.rcParams.update(params)

#cumulative functions for non binned and binned data
def smallerthan(x):
    q = []
    for a in x:
        l = x[x<=a]
        q.append(len(l))
    return q

def smallerthan_bins(x):
    q = []
    qerr = []
    bins = np.arange(min(x),max(x), 0.5)
    for a in bins:
        l = x[x<=a]
        q.append(len(l))
        qerr.append(np.std(l)/np.sqrt(len(l)))
    return bins, q, qerr

#load catalogue
mag, magerr= np.loadtxt("galaxies.csv", unpack=True, skiprows=1, delimiter=',', usecols=(6,7))

mag = np.sort(mag)
magerr = np.sort(magerr)

#Non-binned data
N = smallerthan(mag)
N = np.array(N)
log = np.log10(N)
logerr = (np.sqrt(N))/(N*np.log(10))

#select data to fit straight line
a = 0
b = 1200
mag_linear = mag[a:b]
log_linear = log[a:b]
logerr_linear = logerr[a:b]

#plot non binned data
fig, ax = plt.subplots(1,1)
ax.errorbar(mag, log, xerr=magerr, yerr=logerr, fmt='none', label='data')

#fit linear section
fit, cov = np.polyfit(mag_linear, log_linear, w=1/logerr_linear, deg=1, cov=True)
bfl = np.poly1d(fit)
ax.plot(mag_linear, bfl(mag_linear), label='weighted fit')
ax.set_xlabel("Magnitude, m")
ax.set_ylabel("log N(m)")
ax.grid()
ax.legend(loc='lower right')

#binned data
bins, N, Nerr = smallerthan_bins(mag)
N = np.array(N)
log = np.log10(N)
logerr = (np.sqrt(N))/(N*np.log(10))

#select data to fit straight line
a = 1
b = 17
mag_linear = mag[a:b]
log_linear = log[a:b]
logerr_linear = logerr[a:b]
bins_linear = bins[a:b]

fig, ax = plt.subplots(1,1)
ax.errorbar(bins[1:len(bins)], log[1:len(log)], xerr=Nerr[1:len(Nerr)], yerr=logerr[1:len(logerr)], fmt='none', label='data')

#fit binned data
fitb, covb = np.polyfit(bins_linear, log_linear, w=1/logerr_linear, deg=1, cov=True)
bfl = np.poly1d(fitb)
ax.plot(bins_linear, bfl(bins_linear),label='weighted fit')

ax.set_xlabel("Magnitude, m")
ax.set_ylabel("log N(m)")
ax.grid()
ax.legend(loc='lower right')

#extract  fitting parameters
print('Non-binned fit')
print('gradient of fit =', fit[0], '±',np.sqrt(cov[0,0]))
print('Intercept=',fit[1],'±', np.sqrt(cov[1,1]))

print('binned fit')
print('gradient of fit =',fitb[0],'±', np.sqrt(covb[0,0]))
print('intercept=',fitb[1],'±', np.sqrt(covb[1,1]))
plt.show()
