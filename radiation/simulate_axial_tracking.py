from radiation import ureg
from display_months import display_months
from radiation import Radiation
import numpy as np

import matplotlib.dates as dates
from radiation import plt

import matplotlib.pyplot as plt

latitudes = np.arange(0, 61, 1) * ureg.deg
days = np.linspace(0, 365) * ureg.day
betas = []*ureg.deg
gammas = []*ureg.deg
module_type='2d'
rb = Radiation(latitudes, days, betas, gammas, module_type)

fig, ax = plt.subplots()

plt.plot(latitudes.magnitude,np.squeeze(rb.I_tmA_day_average))
#rb.plot_irradiance('Days', 'Latitude', 'TotalD')
header = "LATITUDE, OPTIMUM"
data = np.concatenate([latitudes.magnitude[:,None], rb.I_tmA_day_average.magnitude.reshape(len(rb.I_tmA_day_average.magnitude),1)], axis=1)
np.savetxt(module_type, data, delimiter = ",", header = header)
