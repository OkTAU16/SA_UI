import Rbeast as rb
import numpy as np
from scipy import interpolate
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt


class SLM_tools:
    @staticmethod
    def interpolate_data_over_regular_time(data: np.array, sample_rate: int = 1):

        """interpolate data over a regularly spaced time vector
            inputs:
                data (np.array): time series data
                sample_rate (int): optional
            outputs:
                time_vec (np.array): regularly spaced time vector
                data_new (np.array): interpolated data over the regularly spaced time vector"""
        try:
            t = np.arange(0, np.shape(data)[1], 1 / sample_rate)
            data_new = interpolate.interp1d(t, data, kind="linear")
            return data_new.y, t
        except Exception as e:
            print(e)

    @staticmethod
    def downsample(data: np.array, time_vec: np.array, downsampling_factor: int):

        """downsample the data by a user input downsampling_factor
        inputs:
            data (np.array): data to downsample
            downsampling_factor (int): downsampling factor
        output:
            data (np.array): downsampled data"""

        try:
            downsampled_data = signal.decimate(data, downsampling_factor)
            downsampled_time = signal.resample(time_vec, downsampling_factor)
            return downsampled_data, downsampled_time
        except Exception as e:
            print(e)

    @staticmethod
    def extract_vertical_line_locs(plot_object: tuple):

        """extract vertical lines x location from plot_beast
        input:
            plot_object (tuple): output of rb.plot(o)
        output:
            vertical_line_locs (list): a sorted list of the x locations of the vertical lines """

        # TODO: talk with micheal to check this function,do we need the loc of the lines or the values?
        try:
            fig, axes = plot_object
            vertical_line_locs = [line.get_xdata()[0] for ax in axes for line in ax.get_lines()
                                  if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]]
            vertical_line_locs = list(set(vertical_line_locs))
            vertical_line_locs.sort()
            return vertical_line_locs
        except Exception as e:
            print(e)

    @staticmethod
    def beast(data: np.array):

        """call the BEAST algorithm and extract the change points
        input:
            data (np.array):time series data for analysis
        outputs:
            change_points (list): sorted list of change points index
            ver_locs (list): sorted list of the vertical lines in the plot"""
        # TODO: check that beast plot doesn't pop
        try:
            o = rb.beast(data, 0, tseg_minlength=0.1 * data.shape[1], season="none")
            plt.switch_backend('Agg')
            x = rb.plot(o)
            ver_locs = SLM_tools.extract_vertical_line_locs(x)
            cp = np.sort(o.trend.cp[0:int(o.trend.ncp_median)])
            cp = cp[~np.isnan(cp)]
            cp.insert(0, 0)  # TODO: check with micheal if were adding the firts index or time 0 sec?
            plt.switch_backend('default')
            return cp, ver_locs
        except Exception as e:
            print(e)


if __name__ == "__main__":
    print('hello world')
