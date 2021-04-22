from radiation import plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.dates as dates

def display_months(ax, param):
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_days_sum = np.concatenate([[0], np.cumsum(month_days)])

    if param in 'x_axis':
        plt.xlim([0, 365])
        ax.set_xticks(month_days_sum)
        ax.xaxis.set_major_formatter(ticker.NullFormatter())

        ax.xaxis.set_minor_locator(dates.MonthLocator(bymonthday=16))
        ax.xaxis.set_minor_formatter(dates.DateFormatter(fmt='%b'))

        for tick in ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('center')
        plt.xlabel('')
    elif param in 'y_axis':
        ax.set_yticks(month_days_sum)
        # plt.yticks(month_days_sum, ['', '', '', '', '', '', '', '', '', '', '', '', ''])
        ax.yaxis.set_major_formatter(ticker.NullFormatter())

        ax.yaxis.set_minor_locator(dates.MonthLocator(bymonthday=16))
        ax.yaxis.set_minor_formatter(dates.DateFormatter(fmt='%b'))

        # ax.xaxis.set_minor_formatter(dates.DateFormatter('%b'))

        for tick in ax.yaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)

        plt.ylim([0, 365])

    else:
        raise ValueError("Axis must be x_axis or y_axis")

