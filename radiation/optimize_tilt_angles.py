from scipy.optimize import minimize
from radiation import ureg
from radiation import Radiation
import numpy as np

from display_months import display_months
import matplotlib.dates as dates
import matplotlib.ticker as ticker
from scipy.optimize import Bounds

import matplotlib.pyplot as plt

latitudes = np.arange(0, 61, 1) * ureg.deg
gammas = 0 * ureg.degree

n_times_move = 1

days_in_year = 365
spring_equinox = 81
#beta_fraction0 = np.arange(1/n_times_move, 1, 1/n_times_move)
betas0 = np.zeros((n_times_move))
days0 = spring_equinox + days_in_year*np.arange(0, 1, 1/n_times_move)

x0 = np.concatenate([betas0, days0])

beta_lower_bound = -30
beta_upper_bound = 90

days_lower_bound = 0
days_upper_bound = 365

#lb = (beta_lower_bound,)*n_times_move+(days_lower_bound,)*n_times_move
#ub = (beta_upper_bound,)*n_times_move+(days_upper_bound,)*n_times_move
#lb = np.concatenate([beta_lower_bound*np.ones((n_times_move, 1)), days_lower_bound*np.ones((n_times_move, 1))])
#ub = np.concatenate([beta_upper_bound*np.ones((n_times_move, 1)), days_upper_bound*np.ones((n_times_move, 1))])

bounds = ((beta_lower_bound, beta_upper_bound),)*n_times_move + ((days_lower_bound, days_upper_bound),)*n_times_move
#(days_lower_bound, days_upper_bound),*3)

#ub = np.concatenate(beta_upper_bound*np.ones((n_times_move, 1)), days_upper_bound*np.ones((n_times_move, 1)))
#bounds = (lb, ub)

#bounds = Bounds(lb, ub)


def calc_ITmA(x,n_times_move, latitude, gamma):
    days_per_interval = round(100/n_times_move)
    days = np.zeros((n_times_move, days_per_interval))*ureg.days
    betas = x[0:n_times_move]*ureg.degree
    days_move = np.sort(x[n_times_move:])
    days_in_year = 365
    #rb = []
    rb = n_times_move*[None]
    sum = 0
    for ind in range(n_times_move):
        if ind != 0:
            days[ind,:] = np.linspace(days_move[ind-1],days_move[ind], days_per_interval)*ureg.day
        else:
            days[ind, :] = np.linspace(days_move[-1]-days_in_year, days_move[ind], days_per_interval)*ureg.day
        rb[ind] = Radiation(latitude, days[ind,:], betas[ind], gamma)
        sum += rb[ind].I_tmA
    return -sum.magnitude.item()

beta_optimum = np.zeros((len(latitudes), n_times_move*2))
opt_val = np.zeros((len(latitudes), 1))
ind = 0
#out = calc_ITmA(x0, n_times_move, latitudes[ind], gammas)
res = []
method = 'L-BFGS-B'
#res = minimize(calc_ITmA, x0, args = (n_times_move, latitudes[ind], gammas), method = 'nelder-mead')
res = minimize(calc_ITmA, x0, args = (n_times_move, latitudes[ind], gammas), method = method, options={'disp': True}, bounds = bounds)
beta_optimum[ind, :] = res.x
opt_val[ind] = -res.fun

for ind in range(1,len(latitudes)):
    res = minimize(calc_ITmA, beta_optimum[ind-1, :] , args = (n_times_move, latitudes[ind], gammas), method = method, options={'disp': True},  bounds = bounds)
    beta_optimum[ind, :] = res.x
    opt_val[ind] = -res.fun

# for ind in range(1,len(latitudes)):
#     res = minimize(calc_ITmA, beta_optimum[ind-1, :] , args = (n_times_move, latitudes[ind], gammas), method = 'nelder-mead')
#     beta_optimum[ind, :] = res.x
#     opt_val[ind] = -res.fun

for ind in range(n_times_move):
    #y_plot = np.take(y_module.magnitude, ind_line, axis=line_index)
    plt.plot(latitudes.magnitude, beta_optimum[:, ind])
plt.xlim([0, np.max(latitudes.magnitude)])
#plt.ylim([0, np.max(latitudes.magnitude)])

plt.ylabel('Optimum Tilt (' + u'\N{DEGREE SIGN}' + ')')
plt.xlabel('Latitude (' + u'\N{DEGREE SIGN}' + ')')


beta_optimum[:, n_times_move:] = np.sort(beta_optimum[:, n_times_move:],1)

#plt.figure(2)
fig, ax = plt.subplots()
if n_times_move == 0:
    plt.fill([latitudes]+latitudes[::-1]),np.zeros((len(latitudes), n_times_move*2))+ 365*np.ones((len(latitudes), n_times_move*2))
else:
    for ind in range(n_times_move):
        if ind == 0:
            p = plt.fill(np.concatenate([latitudes.magnitude, latitudes[::-1].magnitude]), np.concatenate([np.zeros(len(latitudes)),beta_optimum[::-1, n_times_move]]))
        else:
            plt.fill(np.concatenate([latitudes.magnitude, latitudes[::-1].magnitude]), np.concatenate([beta_optimum[:, n_times_move+ind-1],beta_optimum[::-1, n_times_move+ind]]))
    plt.fill(np.concatenate([latitudes.magnitude, latitudes[::-1].magnitude]), np.concatenate([beta_optimum[:, -1],365*np.ones(len(latitudes))]), color = p[0].get_facecolor())

data = np.concatenate([latitudes.magnitude[:,None], beta_optimum,opt_val/365], axis=1)
header = "LATITUDE," + "BETA,"*n_times_move + "DAY,"*n_times_move + "OPTIMUM"
np.savetxt("Optimize"+str(n_times_move), data, delimiter = ",", header = header)



    #tick.label1.set_horizontalalignment('center')

#ax.set_xticklabels(labels)
#plt.setp(axis.get_xticklabels(minor=True), visible=True)
#imid = len(r) // 2
#h = ax.set_ylabel(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
#h = plt.ylabel('Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
#h.set_rotation(0)
#ax.yaxis.set_minor_formatter(['Jan', 'Feb'])

#ax.xaxis.set_minor_formatter(dates.DateFormatter('%b'))


plt.xlim([0, np.max(latitudes.magnitude)])
display_months(ax, 'y_axis')

plt.xlabel('Latitude (' + u'\N{DEGREE SIGN}' + ')')



plt.show(block=True)






