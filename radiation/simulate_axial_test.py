from radiation import ureg
from display_months import display_months
from radiation import Radiation
import numpy as np

import matplotlib.dates as dates
from radiation import plt

import matplotlib.pyplot as plt

latitudes = np.arange(0, 61, 2) * ureg.deg
days = np.linspace(0, 365) * ureg.day
betas = []
gammas = []

rb = Radiation(latitudes, days, betas, gammas)
