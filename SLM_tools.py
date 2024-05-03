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
                values_vec = dict_data['foo'][:, 0]
                if time_vec_exists:
                    time_vec = dict_data['foo'][:, 1]
                    distance = dict_data['foo'][:, 2:2 + target_num]
                    return values_vec, time_vec, distance
                distance = dict_data['foo'][:, 1:1 + target_num]
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
    def downsample(data: np.array,  downsampling_factor: int,time_vec: np.array = None):

        """downsample the data by a user input downsampling_factor
        inputs:
            data (np.array): data to downsample
            downsampling_factor (int): downsampling factor
        output:
            data (np.array): downsampled data"""

        try:
            downsampled_data = signal.decimate(data, downsampling_factor, ftype='fir')
            downsampled_time = signal.decimate(time_vec, downsampling_factor, ftype='fir') if time_vec else None
            return downsampled_data, downsampled_time if time_vec else None
        except Exception as e:
            print(e)

    # @staticmethod
    # def extract_vertical_line_locs(plot_object: tuple):
    #     # TODO: check is this function is necessary
    #     """
    #     extract vertical lines x location from plot_beast
    #     input:
    #         plot_object (tuple): output of rb.plot(o)
    #     output:
    #         vertical_line_locs (list): a sorted list of the x locations of the vertical lines
    #     """
    #
    #     # TODO: talk with micheal to check this function,do we need the loc of the lines or the values? -->the values
    #     try:
    #         fig, axes = plot_object
    #         vertical_line_locs = [line.get_xdata()[0] for ax in axes for line in ax.get_lines()
    #                               if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]]
    #         vertical_line_locs = list(set(vertical_line_locs))
    #         vertical_line_locs.sort()
    #         return vertical_line_locs
    #     except Exception as e:
    #         print(e)

    @staticmethod
    def beast(data: np.array):
        # TODO: check this function
        """call the BEAST algorithm and extract the change points
        input:
            data (np.array):time series data for analysis
        outputs:
            change_points (np.array): sorted np.array of change points index
            mean_trend (np.array): mean trend from BEAST"""
        try:
            o = rb.beast(data, 0, tseg_minlength=0.1 * data.shape[1], season="none", torder_minmax=[1, 1.01])
            mean_trend = o.trend.Y
            cp = np.sort(o.trend.cp[0:int(o.trend.ncp_median)])
            cp = cp[~np.isnan(cp)]
            cp = np.insert(cp, 0, 0)
            return o, cp, mean_trend
        except Exception as e:
            print(e)

    @staticmethod
    def segment_data(energy: np.array, distance: np.array, mean_trend: np.array, cp: np.array, N=None):
        # TODO: check this function
        len_cp = len(cp) - 1
        mu = np.zeros(len_cp)
        std = np.zeros(len_cp)
        skew = np.zeros(len_cp)
        trend_vec = np.zeros(len_cp)
        times_vec = np.zeros(len_cp)
        sa_vec = np.zeros(len_cp)
        cumulated_time_vec = np.zeros(len_cp)
        assembly_mat = np.zeros((energy.shape[1], 2 if N is None else N))
        mean_trend = np.floor(mean_trend)
        cp_int = np.vectorize(int)(cp)
        for i in range(2 if N is None else N):
            assembly = np.zeros(energy.shape[1])
            mimic = np.where(distance[:, i] == 0)
            assembly[mimic] = 1
            assembly_mat[:, i] = assembly

        for i in range(len_cp):
            mu[i] = np.nanmean(energy[:, cp_int[i]:cp_int[i + 1]])
            std[i] = np.std(energy[:, cp_int[i]:cp_int[i + 1]])
            median = np.median(energy[:, cp_int[i]:cp_int[i + 1]])
            skew[i] = (mu[i] - median) / 3 * std[i]
            abc = np.polyfit(range(cp_int[i], cp_int[i + 1]), mean_trend[cp_int[i]:cp_int[i + 1]], 1)  # TODO: is this necessary?
            trend_vec[i] = abc[0]
            times_vec[i] = cp_int[i + 1] - cp_int[i]
            for j in range(2 if N is None else N):
                sa_vec[i] += np.sum(assembly_mat[cp_int[i]:cp_int[i + 1], j])
        sa_vec[sa_vec > 1] = 1

        for i in range(len_cp):
            temp1 = np.where(sa_vec == 1)[0]
            if i not in temp1:
                try:
                    temp2 = np.where(temp1 > i)[0][0]
                    omega = np.cumsum(times_vec[i:temp1[temp2]])
                    cumulated_time_vec[i] = omega[-1]
                except Exception:
                    cumulated_time_vec[i] = np.nan
            else:
                cumulated_time_vec[i] = 0

        return mu, std, skew, trend_vec, times_vec, sa_vec, cumulated_time_vec


if __name__ == "__main__":
    print('hello world')
