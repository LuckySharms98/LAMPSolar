import numpy as np
import matplotlib.pyplot as plt

import matplotlib as matplotlib
font = {'size'   : 14}
matplotlib.rc('font', **font)

max_n_times_move = 3

data = {}
labels = []

def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    return results

latitudeMax = 50
latitudeMin = 2
for ind in range(max_n_times_move):
    data[ind] = np.loadtxt('Optimize'+str(ind+1),delimiter=',')
    fig, ax = plt.subplots()
    for j in range(ind+1):
        inds = data[ind][:, 0] <= latitudeMax
        plt.plot(data[ind][inds,0], data[ind][inds,1+j])
        results = polyfit(data[ind][inds,0], data[ind][inds,1+j], 1)
        inds = (data[ind][:, 0] <= latitudeMax) & (data[ind][:, 0] >= latitudeMin)
        plt.plot(data[ind][inds,0], data[ind][inds,1+j]/data[ind][inds,0])
        results2 = polyfit(data[ind][inds,0], data[ind][inds,1+j]/data[ind][inds,0], 1)

plt.show(block=True)