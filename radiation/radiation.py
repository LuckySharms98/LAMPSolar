import numpy as np
import math

from pint import UnitRegistry
ureg = UnitRegistry()
from scipy import interpolate
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
#matplotlib.use('macosx')
import matplotlib as matplotlib
from matplotlib.cm import ScalarMappable

font = {'size'   : 14}
matplotlib.rc('font', **font)

class Radiation:
    def __init__(self, latitudes, days, betas, gammas, module_type='fixed', beta_fraction_flag=0):
        # self.latitudes = latitudes # in degrees
        self.set_latitudes(latitudes)
        self.set_days(days)  # day of year, where 1 is January 1, 365 is Dec 31
        self.beta_fraction_flag = beta_fraction_flag  # 0 or 1
        self.set_betas(betas)  # tilt of module in degrees
        self.gammas = gammas  # azimuth of module in degrees

        self.set_module_type(module_type)

        _omega = np.linspace(-math.pi, math.pi, 100) * ureg.rad  # hour angle in radians
        self._omegas = _omega
        self.calculate_daily_insolation()

    def calculate_daily_insolation(self):
        # print('inside')
        deltas = Radiation.calculate_delta(self.days)
        # breakpoint()
        # set_trace()
        if self.module_type.lower() in 'fixed':
            [omega_v, latitude_v, delta_v, beta_v, gamma_v] = np.meshgrid(self._omegas, self.latitudes, deltas,
                                                                          self.betas, self.gammas)
        elif self.module_type.lower().str.match(pat = '(1daxisNS)|(1daxisEW)|(2d)'):
            [omega_v, latitude_v, delta_v] = np.meshgrid(self._omegas, self.latitudes, deltas)




        cos_theta_z_v = np.sin(delta_v) * np.sin(latitude_v) + np.cos(delta_v) * np.cos(latitude_v) * np.cos(omega_v)
        theta_z_v = np.arccos(cos_theta_z_v)
        alpha_s_v = np.arcsin(cos_theta_z_v)

        alpha_s_v[alpha_s_v < 0] = 0
        theta_z_v[theta_z_v > math.pi / 2] = math.pi / 2
        #cos_theta_z_v[theta_z_v > math.pi / 2] = 0
        i_b_v = 1.353 * 0.7 ** ((1 / cos_theta_z_v) ** 0.678) # warning here because it doesn't like 1/0
        i_b_v[alpha_s_v <= 0] = 0

        cos_gamma_s_v = (cos_theta_z_v * np.sin(latitude_v) - np.sin(delta_v)) / (                np.sin(theta_z_v) * np.cos(latitude_v))
        gamma_s_v = np.sign(omega_v) * np.arccos(cos_gamma_s_v)

        gamma_s_v[np.isnan(gamma_s_v)] = 0
        gamma_s_v[alpha_s_v <= 0] = 0

        if self.module_type.lower().str.match(pat='(fixed)'):
            if self.beta_fraction_flag:
                cos_theta_tilt_v = cos_theta_z_v * np.cos(beta_v * latitude_v) + np.sin(theta_z_v) * np.sin(
                    beta_v * latitude_v) * np.cos(gamma_s_v - gamma_v)
            else:
                cos_theta_tilt_v = cos_theta_z_v * np.cos(beta_v) + np.sin(theta_z_v) * np.sin(beta_v) * np.cos(
                    gamma_s_v - gamma_v)
        elif self.module_type.lower().str.match(pat='(1daxisNS)'):
            cos_theta_tilt_v = math.sqrt(1 - np.cos(delta_v)**2*np.sin(omega_v)**2)
            beta_v = np.arctan(np.tan(theta_z_v)*abs(np.cos(gamma_s_v)))
        elif self.module_type.lower().str.match(pat='(1daxisEW)'):
            cos_theta_tilt_v = math.sqrt(np.cos(theta_z_v)^2+ np.cos(delta_v) ** 2 * np.sin(omega_v) ** 2)
            gamma_v = np.zeros(gamma_s_v.shape )
            beta_v = np.arctan(tan(thetaz_v) * abs(cos(gamma_s_v)))
        elif self.module_type.lower().str.match(pat='(2d)'):
            cos_theta_tilt_v = 1


        i_b_tilt_v = i_b_v * cos_theta_tilt_v
        i_b_tilt_v[i_b_tilt_v < 0] = 0
        i_d_v = 0.1 * i_b_v

        if self.beta_fraction_flag:
            i_d_tilt_v = (1 + np.cos(beta_v * latitude_v)) / 2 * i_d_v
        else:
            i_d_tilt_v = (1 + np.cos(beta_v)) / 2 * i_d_v
        # print(I_b_v)
        #print(I_d_v.shape)
        #print(self._omegas.shape)
        self.I_dD = 12 / math.pi * np.trapz(i_d_v, self._omegas, axis=1)/ureg.rad
        self.I_dmD = 12 / math.pi * np.trapz(i_d_tilt_v, self._omegas, axis=1)/ureg.rad

        self.I_bD = 12 / math.pi * np.trapz(i_b_v, self._omegas, axis=1)/ureg.rad
        self.I_bmD = 12 / math.pi * np.trapz(i_b_tilt_v, self._omegas, axis=1)/ureg.rad

        self.I_tD = self.I_dD + self.I_bD
        self.I_tmD = self.I_dmD + self.I_bmD

        day_index = 3

        self.I_bA = np.trapz(self.I_bD, self.days, axis=day_index)/ureg.day
        self.I_dA = np.trapz(self.I_dD, self.days, axis=day_index)/ureg.day
        self.I_tA = self.I_bA + self.I_dA

        self.I_bmA = np.trapz(self.I_bmD, self.days, axis=day_index)/ureg.day
        self.I_dmA = np.trapz(self.I_dmD, self.days, axis=day_index)/ureg.day
        self.I_tmA = self.I_bmA + self.I_dmA

        self.I_bA_day_average = self.I_bA /(self.days[-1]-self.days[0])
        self.I_dA_day_average = self.I_dA /(self.days[-1]-self.days[0])
        self.I_tA_day_average = self.I_bA_day_average + self.I_dA_day_average

        self.I_bmA_day_average = self.I_bmA/(self.days[-1]-self.days[0])
        self.I_dmA_day_average = self.I_dmA/(self.days[-1]-self.days[0])
        self.I_tmA_day_average = self.I_bmA_day_average + self.I_dmA_day_average

        return self

    @staticmethod
    def _check_vars_plot(y, var_x_index, line_index):
        n_dim = len(y.shape)
        # varsNotPlot = setdiff(1:n_dim, [varXIndex lineIndex]);
        vars_not_plot = np.setdiff1d(np.array(range(n_dim)), [var_x_index, line_index])
        for ind in range(len(vars_not_plot)):
            if y.shape[vars_not_plot[ind]] != 1:
                raise ValueError("All other variables should be of size 1")

    def get_irradiance_variable(self, variable):
        if variable.lower() in 'beamday':
            var_plot = self.I_bD
            var_module_plot = self.I_bmD
            var_label = ('Direct Insolation (kW-h/m$^2$/day)')
        elif variable.lower() in 'diffuseday':
            var_plot = self.I_dD
            var_module_plot = self.I_dmD
            var_label = ('Diffuse Insolation (kW-h/m$^2$/day)')
        elif variable.lower() in 'totalday':
            var_plot = self.I_tD
            var_module_plot = self.I_tmD
            var_label = ('Total Insolation (kW-h/m$^2$/day)')
        elif variable.lower() in 'totalannualperday':
            var_plot = self.I_tA_day_average
            var_module_plot = self.I_tmA_day_average
            var_label = ('Total Insolation (kW-h/m$^2$/day)')
        elif variable.lower() in 'diffuseannualperday':
            var_plot = self.I_dA_day_average
            var_module_plot = self.I_dmA_day_average
            var_label = ('Diffuse Insolation (kW-h/m$^2$/day)')
        elif variable.lower() in 'beamannualperday':
            var_plot = self.I_bA_day_average
            var_module_plot = self.I_bmA_day_average
            var_label = ('Direct Insolation (kW-h/m$^2$/day)')
        elif variable.lower() in 'totalannual':
            var_plot = self.I_tA
            var_module_plot = self.I_tmA
            var_label = ('Total Insolation (kW-h/m$^2$/day)')
        elif variable.lower() in 'diffuseannual':
            var_plot = self.I_dA
            var_module_plot = self.I_dmA
            var_label = ('Diffuse Insolation (kW-h/m$^2$/day)')
        elif variable.lower() in 'beamannual':
            var_plot = self.I_bA
            var_module_plot = self.I_bmA
            var_label = ('Direct Insolation (kW-h/m$^2$/day)')
        else:
            raise ValueError(variable + " is not recognized")
        return var_plot, var_module_plot, var_label

    def get_variable(self, variable):
        if variable.lower() in 'latitude':
            var_index = 0
            var = self.latitudes
            var_label = 'Latitude (' + u'\N{DEGREE SIGN}' + ')'
            var_units = (u'\N{DEGREE SIGN}')
        elif variable.lower() in 'days':
            var_index = 1
            var = self.days
            var_label = 'Day'
            var_units = ('')
        elif variable.lower() in 'betas':
            var_index = 2
            var = self.betas
            var_label = 'Tilt (' + u'\N{DEGREE SIGN}' + ')'
            var_units = u'\N{DEGREE SIGN}'
        elif variable.lower() in 'betafraction':
            var_index = 2
            var = self.betas
            var_label = 'Tilt Fraction of Latitude'
            var_units = ''
        elif variable.lower() in 'gamma':
            var_index = 3
            var = self.gammas
            var_label = 'Azimuth'
            var_units = (u'\N{DEGREE SIGN}')
        else:
            raise ValueError(variable + " is not recognized")
        return var_index, var, var_label, var_units

    def get_num_omega(self):
        return len(self._omegas)

    def set_module_type(self, value):
        # if not(value.check('dimensionless')):
        #    raise ValueError("Latitude should be dimensionless")
        if not value.str.match(pat = '(fixed)|(1daxisNS)|(1daxisEW)|(2d)')
            raise ValueError('module type needs to be fixed, 1daxisNW, 1daxisEW, or 2d')




    def set_latitudes(self, value):
        # if not(value.check('dimensionless')):
        #    raise ValueError("Latitude should be dimensionless")
        if not (value.dimensionless):
            raise ValueError("Latitude should be dimensionless")
        if np.isscalar(value.magnitude):
            value = [value.magnitude] * value.units  # this makes value iterable
        if any(value < -90) or any(value > 90):
            raise ValueError("Latitude below 90 or above 90 degrees is not possible")
        self.latitudes = value

    def set_betas(self, value):
        if not (self.beta_fraction_flag):
            if not (value.dimensionless):
                raise ValueError("Betas should be dimensionless")
            if np.isscalar(value.magnitude):
                value = [value.magnitude] * value.units  # this makes value iterable
            self.betas = value
        else:
            if np.isscalar(value):
                value = [value]
            self.betas = ureg.Quantity(value)

    def set_gammas(self, value):
        if not (value.dimensionless):
            raise ValueError("Gammas should be dimensionless")
        if np.isscalar(value.magnitude):
            value = [value.magnitude] * value.units  # this makes value iterable
        self.gammas = value

    def set_days(self, value):
        # if not(value.check('dimensionless')):
        #    raise ValueError("Latitude should be dimensionless")
        if not (value.units == 'day'):
            raise ValueError("Days should be units of time")
        if np.isscalar(value.magnitude):
            value = [value.magnitude] * value.units  # this makes value iterable
        self.days = value

    def get_delta(self):
        return Radiation.calculate_delta(self.days)

    def get_num_beta(self):
        return len(self.betas)

    @staticmethod
    def calculate_delta(days):
        return 23.45 * np.sin(360 / (365) * (284 + days / ureg.day)*ureg.degree)*ureg.degree

    @classmethod
    def with_beta_fraction(cls, latitudes, days, beta_fraction, gammas):
        return cls(latitudes, days, beta_fraction, gammas, 1)

    def contour_tilt_misalignment(self, misalignment, normalize_file=0):
        num_misalignment_tilts = 100
        num_latitudes = len(self.latitudes)
        I_tmA_interp = np.zeros((num_misalignment_tilts, num_latitudes))
        tilt_misalignment_relative = np.linspace(-misalignment, misalignment, num_misalignment_tilts)

        if normalize_file:
            data = np.loadtxt('Optimize1', delimiter=',')
            opt_beta = data[:, 1]*ureg.degree
            max_I_tmA_day_average = data[:, 3]
            if len(max_I_tmA_day_average) != len(self.latitudes):
                raise ValueError("Length of latitudes and maxI_tmA should be the same")
        else:
            max_I_tmA_ind = np.argmax(self.I_tmA, 1)
            opt_beta_check = self.betas[max_I_tmA_ind]
            max_I_tmA_day_average = np.amax(self.I_tmA_day_average, 1).magnitude

        for ind in range(0, num_latitudes):
            tilt_misalignment = tilt_misalignment_relative + opt_beta[ind]
            I_tmA_interp[:,ind] = np.interp(tilt_misalignment, self.betas, np.squeeze(self.I_tmA_day_average[ind, :].magnitude))/max_I_tmA_day_average[ind]

        [tilt_misalignment_v, latitudes_v] = np.meshgrid(self.latitudes.magnitude, tilt_misalignment_relative.magnitude)
        plt.contourf(tilt_misalignment_v, latitudes_v, I_tmA_interp, 200, linestyles='none', cmap=plt.cm.jet)
        plt.xlabel('Latitude (' + u'\N{DEGREE SIGN}' + ')')
        plt.ylabel('Normalized Insolation')

        #plt.figure(plt.gcf().number+1)
        fig, ax = plt.subplots()
        avg_over_all_latitudes = np.mean(I_tmA_interp, 1)

        plt.plot(tilt_misalignment_relative.magnitude, avg_over_all_latitudes)
        plt.xlabel('Tilt Misalignment (' + u'\N{DEGREE SIGN}' + ')')
        plt.ylabel('Normalized Insolation')

        ax.tick_params(top=True, right=True, direction='in')






    def contour_irradiance(self, x_variable, y_variable, z_variable, lines, type_plot='Cartesian'):
        [var_x_index, var_x, x_axis_label, var_x_units] = self.get_variable(x_variable)
        [var_y_index, var_y, y_axis_label, var_y_units] = self.get_variable(y_variable)

        [z, z_m, z_axis_label] = self.get_irradiance_variable(z_variable)

        self._check_vars_plot(z, var_x_index, var_y_index)
        #plt.interactive(True)
        [x_v, y_v] = np.meshgrid(var_x, var_y)

        if type_plot.lower() in 'cartesian':
            fig, ax = plt.subplots()
            if var_x_index > var_y_index:
                z_plot = z_m.magnitude
            else:
                z_plot = np.transpose(z_m.magnitude)
            cs = plt.contourf(x_v.magnitude, y_v.magnitude, np.squeeze(z_plot), 200, vmin = min(lines), vmax = max(lines), linestyles='none', cmap = plt.cm.jet)
            cs2 = plt.contour(x_v.magnitude, y_v.magnitude, np.squeeze(z_plot), lines, vmin = min(lines), vmax = max(lines), colors = 'k', linewidths=1,linestyles='solid')

            ax.clabel(cs2, lines, inline = 1, fontsize = 14, inline_spacing = 5)
            # cs = plt.contour(x_v, y_v, np.squeeze(z_plot), lines, colors='k', linewidths=1, linestyles='solid')
            #fig.colorbar(cs,ticks=lines)
            fig.colorbar(
                ScalarMappable(norm=cs.norm, cmap=cs.cmap),
                ticks=lines)
            plt.xlabel(x_axis_label)
            plt.ylabel(y_axis_label)
            ax.tick_params(top=True, right=True, direction='in')
        else:
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
            #fig, ax = plt.subplots()
            ax.set_theta_zero_location("S")
            #cs = ax.contourf(y_v.to('rad').magnitude, x_v.magnitude, np.squeeze(np.transpose(z_m.magnitude)), linestyles='none', cmap=plt.cm.jet)
            cs = ax.contourf(y_v.to('rad').magnitude, x_v.magnitude, np.squeeze(np.transpose(z_m.magnitude)), 200, vmin = min(lines), vmax = max(lines), linestyles='none', cmap = plt.cm.jet)
            #cs2 = plt.contour(x_v, y_v, np.squeeze(z_m.magnitude), lines, colors='k', linewidths=1, linestyles='solid')
            #ax.clabel(cs2, lines, inline = 1, fontsize = 14, inline_spacing = 5)
            cbaxes = fig.add_axes([0.86, 0.1, 0.03, 0.8])
            fig.colorbar(
                ScalarMappable(norm=cs.norm, cmap=cs.cmap),
                ticks=lines, cax = cbaxes)
            #fig.colorbar(cs, ticks=lines)


        #plt.colorbar(cs2, )

        plt.title(z_axis_label)
        points = np.array((x_v.magnitude.flatten(), y_v.magnitude.flatten())).T
        values = np.squeeze(np.transpose(z_m.magnitude)).flatten()
        interpolate.griddata(points, values, (32.2, 90))
        max(np.squeeze(np.transpose(z_m.magnitude)).flatten())
        #interpolate.interp2d(x_v.magnitude, y_v.magnitude,np.squeeze(np.transpose(z_m.magnitude)), 44, 90)
        return fig, ax




    def plot_irradiance(self, x_variable, line_variable, y_variable):
        [var_x_index, var_x, x_axis_label, var_x_units] = self.get_variable(x_variable)
        [y, y_module, y_axis_label] = self.get_irradiance_variable(y_variable)
        [line_index, var_line, line_label, var_line_units] = self.get_variable(line_variable)

        self._check_vars_plot(y, var_x_index, line_index)

        num_line_var = len(var_line)
        #print(var_x.shape)
        #print(np.squeeze(y).shape)
        # set_trace()

        n_dim_y = len(y.shape)
        index_dict = {index: 0 for index, value in enumerate(range(n_dim_y))}
        index_dict[var_x_index] = len(var_x)

        # n_dim_y = len(y.shape)
        # reshape_dim = np.ones(n_dim_y).astype(int)
        # reshape_dim[var_x_index] = y.shape[var_x_index]
        # ind_take = np.array(range(y.shape[var_x_index]))
        # ind_take_reshaped = ind_take.reshape(reshape_dim)
        # y_plot = np.take(y.magnitude, ind_take_reshaped)
        y_plot = np.take(y.magnitude, 0, axis=line_index)
        #plt.interactive(True)
        plt.plot(np.squeeze(var_x.magnitude),np.squeeze(y_plot))

        labels = ['Incident']
        for ind_line in range(num_line_var):
            y_plot = np.take(y_module.magnitude, ind_line, axis=line_index)
            plt.plot(np.squeeze(var_x.magnitude), np.squeeze(y_plot))
            labels += ['Module, ' + line_label + ' = ' + str(var_line.magnitude[ind_line]) + var_line_units]

        plt.legend(labels,frameon=False)

        if not var_x_units:
            plt.xlabel(x_axis_label)
        else:
            plt.xlabel(x_axis_label + '(' + var_x_units + ')')
        #plt.legend(frameon=False)
        plt.ylabel(y_axis_label)



