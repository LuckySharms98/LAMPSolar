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
        labels = ['Fixed']
    else:
        labels += [str(ind+1)+' Tilts per Year']
#labels = ['Incident']

files =['1daxisNS', '1daxisEW', '2d']
labelsTracking = ['1D Tracking NS', '1D Tracking EW', '2D Tracking']
for ind in range(len(files)):
    data[max_n_times_move+ind] = np.loadtxt(files[ind], delimiter=',')
    plt.plot(data[max_n_times_move+ind][:, 0], data[max_n_times_move+ind][:, -1])

labels += labelsTracking

plt.legend(labels,frameon=False)


plt.xlim(0, 60)

plt.xlabel('Latitude (' + u'\N{DEGREE SIGN}' + ')')
plt.ylabel('Annual Average Total Insolation \n(kW-h/m$^2$/day)')

#labels = []
plt.figure(2)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for ind in np.arange(1,max_n_times_move+len(files)):
    plt.plot(data[ind][:, 0], 100*(data[ind][:, -1]/data[0][:, -1]-1), color=colors[ind])
    #labels += [str(ind + 1) + ' Tilts per Year']

plt.legend(labels[1:],frameon=False)


plt.xlabel('Latitude (' + u'\N{DEGREE SIGN}' + ')')
plt.ylabel('Increase in Total Insolation (%)')
plt.xlim(0, 60)
plt.show(block=True)

