from radiation import ureg
from radiation import Radiation
import numpy as np

from radiation import plt
latitudes = np.concatenate([[0, 1], np.arange(2, 61, 2)]) * ureg.deg
#latitudes = np.arange(0, 61, 2) * ureg.deg
days = np.linspace(0, 365) * ureg.day

betaFraction = np.linspace(0, 1.8, 301)
betaFractionFlag = 1
gammas = 0 * ureg.deg

rb = Radiation(latitudes, days, betaFraction, gammas, betaFractionFlag)
rb.contour_irradiance('Latitude', 'BetaFraction','TotalAnnualPerDay', [2, 3, 4, 5, 6, 7, 8]);

plt.title('')
max_I_tmA_ind = np.argmax(rb.I_tmA, 1)

plt.plot(rb.latitudes[1:].magnitude, rb.betas[max_I_tmA_ind[1:]].magnitude, linestyle = '--', color = 'w')

plt.show(block=True)