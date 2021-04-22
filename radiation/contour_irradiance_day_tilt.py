from radiation import ureg
from radiation import Radiation
import numpy as np
from radiation import plt
from display_months import display_months

from matplotlib.legend_handler import HandlerBase
class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                                                color='#1f77b4')
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], color='#ff7f0e')
        return [l1, l2]



latitudes = 60 * ureg.degree
days = np.linspace(0, 365, 40) * ureg.day
betas = np.linspace(-10, 85, 301) * ureg.degree
#betas = 0 * ureg.degree
gammas = [0] * ureg.degree

rb = Radiation(latitudes, days, betas, gammas)
#fig, ax = plt.subplots()
[fig, ax] = rb.contour_irradiance('Days', 'Beta','TotalDay', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

display_months(ax, 'x_axis')


#rb.contour_irradiance('Beta', 'Gamma','TotalAnnualPerDay', [500 1000 1500 2000],'RadialTickSpacing', [10:10:90], 'RadLabels', 8);
#ax.xaxis.set_minor_formatter
#ax.xaxis.get_label().set_fontsize(10)
ax.tick_params(which ='minor', labelsize = 12)

plt.title('')
#plt.figure(2)
fig, ax = plt.subplots()
max_n_times_move = 2
data = {}
colors = ['#1f77b4', '#ff7f0e']
# https://matplotlib.org/devdocs/tutorials/colors/colors.html

line = plt.plot(days.magnitude,np.squeeze(rb.I_tD.magnitude[0,0,0,:]),color='k')
labels = ['Incident']
line_objects = line

for ind in range(max_n_times_move):
    n_times_move = ind + 1
    days_per_interval = round(100 / n_times_move)
    data[ind] = np.loadtxt('Optimize'+str(ind+1),delimiter=',')
    data_latitudes = data[ind][:,0]
    data_opt_beta = data[ind][:, 1:2+ind]
    data_days_move = data[ind][:, 2+ind:-1]
    opt_beta = np.squeeze(data_opt_beta[data_latitudes == latitudes.magnitude,:])*ureg.degree
    days_move =np.squeeze(data_days_move[data_latitudes == latitudes.magnitude,:])

    days_mat = np.zeros((n_times_move, days_per_interval)) * ureg.days
    rb = n_times_move * [None]
    days_in_year = 365
    #line_objects = n_times_move * [None]
#    days_move = data_opt_days
    #line_objects = [None]
    if n_times_move == 1:
        opt_beta_single = opt_beta
        rb = Radiation(latitudes, days, opt_beta, gammas)
        line = plt.plot(days.magnitude,np.squeeze(rb.I_tmD.magnitude),color='#2ca02c')
        labels += ['Module Tilt = ' + "{:.1f}".format(opt_beta.magnitude) + u'\N{DEGREE SIGN}']
        line_objects += line
    else:
        label_string = 'Module Tilt = '
        line = (n_times_move+1)*[None]
        for j_ind in range(n_times_move):
            #label_string = label_string + str(opt_beta[j_ind]) + u'\N{DEGREE SIGN}' + ','
            label_string = label_string + "{:.1f}".format(opt_beta[j_ind].magnitude) + u'\N{DEGREE SIGN}' + ', '

            if j_ind != 0:
                days_mat[j_ind,:] = np.linspace(days_move[j_ind-1],days_move[j_ind], days_per_interval)*ureg.day
            else:
                days_mat[j_ind, :] = np.linspace(0, days_move[j_ind], days_per_interval)*ureg.day
            rb[j_ind] = Radiation(latitudes, days_mat[j_ind,:], opt_beta[j_ind], gammas)
            line[j_ind] = plt.plot(days_mat[j_ind,:].magnitude,np.squeeze(rb[j_ind].I_tmD.magnitude),color=colors[j_ind])
        days_mat = np.linspace(days_move[-1], days_in_year, days_per_interval) * ureg.day
        rb = Radiation(latitudes, days_mat, opt_beta[0], gammas)
        line[n_times_move] = plt.plot(days_mat.magnitude, np.squeeze(rb.I_tmD.magnitude),color=colors[0])
        line_objects += [[plt.Line2D([0], [0.7], color='#1f77b4'), plt.Line2D([0], [0.3], color='#ff7f0e')]]
        labels += [label_string[:-2]]



betas = opt_beta_single + [-23.45, 23.45]*ureg.degree
line = []

#color_cycle = ax._get_lines.color_cycle
#color_cycle = ax._get_lines.prop_cycler
colors = ['#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for ind in range(len(betas)):
    rb = Radiation(latitudes, days, betas[ind], gammas)
    line = plt.plot(days, np.squeeze(rb.I_tmD.magnitude), color = colors[ind])
    labels += ['Module Tilt = ' + "{:.1f}".format(betas[ind].magnitude) + u'\N{DEGREE SIGN}']
    line_objects += line

#plt.legend([object], ['label'],
#           handler_map={object: AnyObjectHandler()})
plt.legend(line_objects, labels,handler_map={object: AnyObjectHandler()},frameon=False, loc = 'upper left')
#plt.legend(line_objects+[object], labels,handler_map={object: AnyObjectHandler()},frameon=False)
plt.ylabel('Total Insolation (kW-h/m$^2$/day)')
display_months(ax, 'x_axis')
plt.show(block=True)