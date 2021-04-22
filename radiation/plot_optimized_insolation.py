import numpy as np
import matplotlib.pyplot as plt

import matplotlib as matplotlib
font = {'size'   : 14}
matplotlib.rc('font', **font)

max_n_times_move = 3

data = {}
labels = []
for ind in range(max_n_times_move):
    data[ind] = np.loadtxt('Optimize'+str(ind+1),delimiter=',')
    plt.plot(data[ind][:,0], data[ind][:,-1])
    #np.polyfit(data[ind][:,0], data[ind][:,])
    if ind == 0:
        labels = [str(ind + 1) + ' Tilt per Year']
    else:
        labels += [str(ind+1)+' Tilts per Year']
#labels = ['Incident']
plt.legend(labels,frameon=False)


plt.xlabel('Latitude (' + u'\N{DEGREE SIGN}' + ')')
plt.ylabel('Annual Average Total Insolation (kW-h/m$^2$/day)')

labels = []
plt.figure(2)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for ind in np.arange(1,max_n_times_move):
    plt.plot(data[ind][:, 0], 100*(data[ind][:, -1]/data[ind-1][:, -1]-1), color=colors[ind])
    labels += [str(ind + 1) + ' Tilts per Year']

plt.legend(labels,frameon=False)


plt.xlabel('Latitude (' + u'\N{DEGREE SIGN}' + ')')
plt.ylabel('Increase in Total Insolation (%)')
plt.show(block=True)