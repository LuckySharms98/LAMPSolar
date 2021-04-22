from radiation import ureg
from radiation import Radiation
import numpy as np
from scipy import interpolate
from radiation import plt
from display_months import display_months

latitudes = [60] * ureg.degree
days = np.linspace(0, 365, 30)  * ureg.day

betas = np.linspace(0, 90, 20) * ureg.degree

gammas = np.linspace(-180, 180, 20) * ureg.degree

rb = Radiation(latitudes, days, betas, gammas)

#fig, ax = plt.subplots()
cs = rb.contour_irradiance('Beta','Gamma','TotalAnnualPerDay', [0, 1, 2, 3, 4, 5, 6, 7], 'polar')
#display_months(ax, 'x_axis')
#plt.clim(0, 8);
plt.title('')
#plt.clim(cs, 0, 10)
#plt.colorbar(cs)

plt.figure(2)


gammas = np.arange(0, 91, 30) * ureg.degree
betas = np.linspace(-20, 90, 50) * ureg.degree
days = np.linspace(0, 365, 30)  * ureg.day
labels = []
for ind in range(len(gammas)):
    rb = Radiation(latitudes, days, betas, gammas[ind])
    plt.plot(betas.magnitude, np.squeeze(rb.I_tmA_day_average.magnitude))
    if ind == 0:
        labels += ['$\gamma = $ ' + str(gammas[ind].magnitude) + u'\N{DEGREE SIGN}']
    else:
        labels += ['$\gamma = \pm$ ' + str(gammas[ind].magnitude) + u'\N{DEGREE SIGN}']

plt.xlabel('Tilt (' + u'\N{DEGREE SIGN}' + ')')
plt.ylabel('Total Insolation (kW-h/m$^2$/day)')
plt.ylim(0, 7)
plt.legend(labels,frameon=False)
plt.show(block=True)

