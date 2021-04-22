from radiation import ureg
from display_months import display_months
from radiation import Radiation
import numpy as np

import matplotlib.dates as dates
from radiation import plt

import matplotlib.ticker as ticker

latitudes = 40 * ureg.degree
days = np.linspace(0, 365, 40) * ureg.day
#betas = np.linspace(-10, 85, 301) * ureg.degree
#betas = [0, 15, 30] * ureg.degree
gammas = 0 * ureg.degree


rb = Radiation(latitudes, days, betas, gammas)
fig, ax = plt.subplots()
rb.plot_irradiance('Days', 'Betas', 'TotalD')

display_months(ax, 'x_axis')

plt.show(block=True)
