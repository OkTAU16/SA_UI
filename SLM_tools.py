import Rbeast as rb
import numpy as np
from scipy import interpolate
from scipy import signal
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd


class SLM_tools:
    @staticmethod
    def load_data(data_path: str, target_num, data_type='csv', time_vec_exists=False):
        # TODO: incomplete, try with more then two targets
        try:
            if data_type == 'csv':
                values_vec = pd.read_csv(data_path, usecols=[0]).to_numpy()
                if time_vec_exists:
                    time_vec = pd.read_csv(data_path, usecols=[1]).to_numpy()
                    distance_columns = list(range(2, 2 + target_num))
                    distance = pd.read_csv(data_path, usecols=distance_columns).to_numpy()
                    return values_vec, time_vec, distance
                distance_columns = list(range(1, 1 + target_num))
                distance = pd.read_csv(data_path, usecols=distance_columns).to_numpy()
                return values_vec, distance
            elif data_type == 'excel':
                values_vec = pd.read_excel(data_path, usecols=[0]).to_numpy()
                if time_vec_exists:
                    time_vec = pd.read_excel(data_path, usecols=[1]).to_numpy()
                    distance_columns = list(range(2, 2 + target_num))
                    distance = pd.read_excel(data_path, usecols=distance_columns).to_numpy()
                    return values_vec, time_vec, distance
                distance_columns = list(range(1, 1 + target_num))
                distance = pd.read_excel(data_path, usecols=distance_columns)
                return values_vec, distance
            elif data_type == '.mat':
                dict_data = sio.loadmat(data_path)
                keys_lst = list(dict_data.keys())
                key = keys_lst[-1]
                values_vec = dict_data[key][:, 0]
                if time_vec_exists:
                    time_vec = dict_data[key][:, 1]
                    distance = dict_data[key][:, 2:2 + target_num]
                    return values_vec, time_vec, distance
                distance = dict_data[key][:, 1:1 + target_num]
                return values_vec, distance
            else:
                raise Exception("file type is not supported")
        except Exception as e:
            print(e)

    @staticmethod
    def interpolate_data_over_regular_time(data: np.array, sample_rate: int = 1):

        """interpolate data over a regularly spaced time vector
            inputs:
                data (np.array): time series data
                sample_rate [Hz] (int): optional
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
            downsampled_data = signal.decimate(data, downsampling_factor, ftype='fir')
            downsampled_time = signal.decimate(time_vec, downsampling_factor, ftype='fir')
            return downsampled_data, downsampled_time
        except Exception as e:
            print(e)

    @staticmethod
    def extract_vertical_line_locs(plot_object: tuple):

        """
        extract vertical lines x location from plot_beast
        input:
            plot_object (tuple): output of rb.plot(o)
        output:
            vertical_line_locs (list): a sorted list of the x locations of the vertical lines
        """

        # TODO: talk with micheal to check this function,do we need the loc of the lines or the values? -->the values
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
        try:
            o = rb.beast(data, 0, tseg_minlength=0.1 * data.shape[1], season="none", torder_minmax=[1, 1.01])
            mean_trend = o.trend.Y
            cp = np.sort(o.trend.cp[0:int(o.trend.ncp_median)])
            cp = cp[~np.isnan(cp)]
            cp.insert(0, 0)
            plt.switch_backend('default')
            return o, cp, mean_trend
        except Exception as e:
            print(e)

    @staticmethod
    def segment_data(data: np.array, mean_trend: np.array, cp: np.array):
        # TODO: complete according to giv_vecs(),missing distance
        mean_trend = np.floor(mean_trend)
        mu = np.zeros_like(data)
        std = np.zeros_like(data)
        median = np.zeros_like(data)
        skew = np.zeros_like(data)
        trend = np.zeros_like(data)
        time = np.zeros_like(data)
        for i in range(0, data.shape[1]):
            mu[i] = np.mean(data[:, cp[i]:cp[i + 1]])
            std[i] = np.std(data[:, cp[i]:cp[i + 1]])
            median[i] = np.median(data[:, cp[i]:cp[i + 1]])
            skew[i] = (mu[i] - median) / (3 * std[i])
            fitted_trend = np.polyfit([cp[i], cp[i + 1]], mean_trend[cp[i]: cp[i + 1]],
                                      1)  # TODO: check if we need this
            trend[i] = fitted_trend[1]
            time[i] = cp[i + 1] - cp[i]
        return mu, std, median, skew, trend, time


if __name__ == "__main__":
    print('hello world')
