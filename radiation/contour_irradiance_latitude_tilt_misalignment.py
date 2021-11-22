from radiation import ureg
from radiation import Radiation
import numpy as np

from radiation import plt
#latitudes = np.concatenate([[0, 1], np.arange(2, 61, 2)]) * ureg.deg
#latitudes = np.arange(0, 61, 2) * ureg.deg
latitudes = np.concatenate([[0, 1], np.arange(2, 61, 2)]) * ureg.deg
#latitudes = np.arange(0, 61, 1) * ureg.deg
days = np.linspace(0, 365) * ureg.day
#days = np.linspace(0, 365, 10) * ureg.day

betas = np.linspace(-35, 70, 101)*ureg.deg
#betas = np.linspace(-35, 70, 301)*ureg.deg
#betaFraction = np.linspace(0, 1.8, 21)
gammas = 0 * ureg.deg

rb = Radiation(latitudes, days, betas, gammas)
rb.contour_irradiance('Latitude', 'Beta','TotalAnnualPerDay', [2, 3, 4, 5, 6, 7])

max_I_tmA_ind = np.argmax(rb.I_tmA, 1)
plt.plot(rb.latitudes.magnitude, rb.betas[max_I_tmA_ind].magnitude, linestyle = '--', color = 'w')


plt.figure(2)
misalignment = 30*ureg.deg
normalize_file = 1
rb.contour_tilt_misalignment(misalignment, normalize_file)
plt.ylim([.88, 1])
#plt.ylim([.94, 1])
plt.xlim([-30, 30])
plt.show(block=True)

#np.polyfit()
#slopes[i, intercepts[ind], r_values[ind], p_value, std_err = stats.linregress(latitudesFit, beta_optimum[latitudes <= latitudes_fit_cutoff, ind])





