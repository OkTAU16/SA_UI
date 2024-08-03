import pickle
import time
from datetime import datetime

import Rbeast as rb
import numpy as np
import scipy.stats
from scipy import interpolate, stats, signal, ndimage
from scipy.stats import linregress
import scipy.io as sio
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle as pkl
import os


class SLM_tools:
    @staticmethod
    def load_data(data_path: str, data_variable_name, target_num, data_type='csv', time_vec_exists=False):
        try:
            if data_type == 'csv':
                if time_vec_exists:
                    time_vec = pd.read_csv(data_path, usecols=[0]).to_numpy()
                    values_vec = pd.read_csv(data_path, usecols=[1], header=None).to_numpy()
                    distance_columns = list(range(2, 2 + target_num))
                    distance = pd.read_csv(data_path, usecols=distance_columns, header=None).to_numpy()
                    return values_vec, time_vec, distance
                else:
                    values_vec = pd.read_csv(data_path, usecols=[0], header=None).to_numpy()
                    distance_columns = list(range(1, 1 + target_num))
                    distance = pd.read_csv(data_path, usecols=distance_columns, header=None).to_numpy()
                    return values_vec, distance
            elif data_type == 'excel':
                if time_vec_exists:
                    time_vec = pd.read_excel(data_path, usecols=[0], header=None).to_numpy()
                    values_vec = pd.read_excel(data_path, usecols=[1], header=None).to_numpy()
                    distance_columns = list(range(2, 2 + target_num))
                    distance = pd.read_excel(data_path, usecols=distance_columns, header=None).to_numpy()
                    return values_vec, time_vec, distance
                else:
                    values_vec = pd.read_excel(data_path, usecols=[0], header=None).to_numpy()
                    distance_columns = list(range(1, 1 + target_num))
                    distance = pd.read_excel(data_path, usecols=distance_columns, header=None).to_numpy()
                    return values_vec, distance
            elif data_type == '.mat':
                dict_data = sio.loadmat(data_path)
                if time_vec_exists:
                    time_vec = dict_data[data_variable_name][:, 0]
                    values_vec = dict_data[data_variable_name][:, 1]
                    distance = dict_data[data_variable_name][:, 2:2 + target_num]
                    return values_vec, time_vec, distance
                else:
                    values_vec = dict_data[data_variable_name][:, 0]
                    distance = dict_data[data_variable_name][:, 1:1 + target_num]
                    return values_vec, distance
            else:
                raise Exception("file type is not supported")
        except Exception as e:
            raise e

    @staticmethod
    def interpolate_data_over_regular_time(data: np.array, sample_rate: int = 1):
        # checked
        """interpolate data over a regularly spaced time vector
            inputs:
                data (np.array): time series data
                sample_rate [Hz] (int): optional
            outputs:
                time_vec (np.array): regularly spaced time vector
                data_new (np.array): interpolated data over the regularly spaced time vector"""
        try:
            t = np.arange(0, np.shape(data)[0], 1 / sample_rate)
            data_new = interpolate.interp1d(t, data, kind="linear")
            return data_new.y, t
        except Exception as e:
            raise e

    @staticmethod
    def downsample(data: np.array, distance: np.array, downsampling_factor: int):
        """downsample the data by a user input downsampling_factor
        inputs:
            data (np.array): data to downsample
            downsampling_factor (int): downsampling factor
        output:
            data (np.array): downsampled data"""

        try:
            length = data.shape[0]
            time_vec = np.linspace(0, length - 1, length // downsampling_factor, dtype=int)
            data = data[time_vec]
            downsampled_distance = distance[time_vec, :]
            downsampled_data = np.reshape(data, (len(data), 1))
            return downsampled_data, downsampled_distance, time_vec
        except Exception as e:
            raise e

    @staticmethod
    def beast(data: np.array):
        """call the BEAST algorithm and extract the change points
        input:
            data (np.array):time series data for analysis
        outputs:
            change_points (np.array): sorted np.array of change points index
            mean_trend (np.array): mean trend from BEAST"""

        try:
            o = rb.beast(data, 0, tseg_minlength=0.01 * data.shape[0], season="none",
                         print_options=False, print_progress=False, tcp_minmax=[0, 10000], sorder_minmax=[1, 5],
                         )
            mean_trend = o.trend.Y
            cp = np.sort(o.trend.cp[0:int(o.trend.ncp_median)])
            cp = cp[~np.isnan(cp)]
            cp = np.insert(cp, 0, 0)
            return o, cp, mean_trend
        except Exception as e:
            raise e

    @staticmethod
    def feature_extraction(energy: np.array, distance: np.array, mean_trend: np.array, cp: np.array, Number_of_targets=None):
        # checked, original give_vecs.m
        Number_of_targets = 2 if Number_of_targets is None else Number_of_targets
        len_cp = len(cp) - 1
        mu = np.zeros(len_cp)
        std = np.zeros(len_cp)
        skew = np.zeros(len_cp)
        trend_vec = np.zeros(len_cp)
        times_vec = np.zeros(len_cp)
        sa_vec = np.zeros(len_cp)
        cumulated_time_vec = np.zeros(len_cp)
        assembly_mat = np.zeros((energy.shape[0], Number_of_targets))
        mean_trend = np.floor(mean_trend)
        cp_int = np.vectorize(int)(cp)
        for i in range(Number_of_targets):
            assembly = np.zeros(energy.shape[0])
            mimic = np.where(distance[:, i] == 0)[0]
            assembly[mimic] = 1
            assembly_mat[:, i] = assembly
        for i in range(len_cp):
            start = cp_int[i]
            end = cp_int[i + 1]
            segment = energy[start:end, :]
            mu[i] = np.nanmean(segment, axis=0)
            std[i] = np.nanstd(segment, axis=0)
            median = np.nanmedian(segment, axis=0)
            skew[i] = (mu[i] - median) / 3 * std[i]
            x = np.arange(start, end)
            abc = np.polyfit(x, mean_trend[cp_int[i]:cp_int[i + 1]], 1)
            trend_vec[i] = abc[0]
            times_vec[i] = end - start
            for j in range(Number_of_targets):
                sa_vec[i] += np.any(assembly_mat[start:end, j])
        sa_vec[sa_vec > 1] = 1

        for i in range(len_cp):
            temp1 = np.where(sa_vec == 1)[0]
            if i not in temp1:
                try:
                    temp2 = np.where(temp1 > i)[0][0]
                    omega = np.cumsum(times_vec[i:temp1[temp2]])
                    cumulated_time_vec[i] = omega[-1]
                except IndexError:
                    cumulated_time_vec[i] = np.nan
            else:
                cumulated_time_vec[i] = 0
        if not np.any(sa_vec == 1):
            return []
        else:
            mu = np.reshape(mu, (len(mu), 1))
            std = np.reshape(std, (len(std), 1))
            skew = np.reshape(skew, (len(skew), 1))
            cumsum_vec = np.cumsum(times_vec)
            cumsum_vec = np.reshape(cumsum_vec, (len(cumsum_vec), 1))
            trend_vec = np.reshape(trend_vec, (len(trend_vec), 1))
            sa_vec = np.reshape(sa_vec, (len(sa_vec), 1))
            cumulated_time_vec = np.reshape(cumulated_time_vec, (len(cumulated_time_vec), 1))
            return [mu, std, skew, cumsum_vec, trend_vec, sa_vec, cumulated_time_vec]


    @staticmethod
    def feature_selection(segment_data_aggregated_output: np.array):
        c_reduced = []
        for list in segment_data_aggregated_output:
            if len(list) == 0:
                continue
            x = list[0]  # mean_vec
            y = list[1]  # std_vec
            z = list[4]  # trend
            w = list[2]  # skew
            v = list[3]  # total_trajectory_time
            c = list[6]  # time_to_self_assembly
            ending_theme = np.where(c == 0)[0]
            if len(ending_theme) > 0:
                ending_theme = ending_theme[0]
            x = x[:ending_theme]
            y = y[:ending_theme]
            z = z[:ending_theme]
            c = c[:ending_theme]
            w = w[:ending_theme]
            v = v[:ending_theme]
            d_reduced = np.column_stack((x, y, z, w, v, c))
            c_reduced.append(d_reduced)
        c_reduced = np.vstack(tuple(c_reduced))
        return c_reduced

    @staticmethod
    def pca(c_reduced: np.array, n_components: int = 3):
        data = c_reduced[:, :n_components]
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        norm_data = (data - data_mean) / data_std
        pca = PCA(n_components=n_components)
        score = pca.fit_transform(norm_data)
        principal_components = pca.components_
        latent = pca.explained_variance_
        score_cov = np.cov(score, rowvar=False)
        score_std = np.diag(score_cov)
        score = score / (score_std[:, None] * np.ones((n_components, score.shape[0]))).T
        return principal_components, score, latent

    @staticmethod
    def data_preparation(score: np.array, c_reduced: np.array, n_components: int = 3):
        a_reduced = []
        if n_components == 3:
            for i in range(score.shape[0]):
                x = score[i, 0]  # mean_vec
                y = score[i, 1]  # std_vec
                z = score[i, 2]  # trend
                c = c_reduced[i, 5]
                b_reduced = np.array([x, y, z, c])
                a_reduced.append(b_reduced)
        else:  # n_components == 5
            for i in range(score.shape[0]):
                x = score[i, 0]  # mean_vec
                y = score[i, 1]  # std_vec
                z = score[i, 4]  # trend
                w = score[i, 2]  # skew
                v = score[i, 3]  # total_trajectory_time
                c = c_reduced[i, 5]
                b_reduced = np.array([x, y, z, w, v, c])
                a_reduced.append(b_reduced)
        a_reduced = np.array(a_reduced)
        return a_reduced

    @staticmethod
    def replace_nan_with_rounded_mean(array):
        mean_val = np.nanmean(array)
        array[np.isnan(array)] = np.round(mean_val)
        return array

    @staticmethod
    def model_training_with_cv(a_reduced: np.array, n_components: int = 3, cv_num: int = 10):
        # from LogTfas....
        np.random.seed(42)
        idn = np.where(a_reduced[:, n_components] != 0)[0]
        mapx = a_reduced[idn, :]
        mapx[:, n_components] = np.log(mapx[:, n_components])
        yo = np.sort(mapx[:, n_components])
        aop = int(np.ceil((0.16 * len(yo)))) - 1
        train_index = int(np.floor(0.6 * mapx.shape[0]))
        validation_index = int(np.floor(0.8 * mapx.shape[0]))
        size_validation_set = len(range(train_index, validation_index))
        tfas_predict_mat = np.zeros((cv_num, size_validation_set))
        tfas_actually_mat = np.zeros((cv_num, size_validation_set))
        mean_error_mat = np.zeros((cv_num, size_validation_set))
        for i in range(cv_num):
            random_x = mapx[np.random.permutation(mapx.shape[0]), :]  # line 89 matlab
            train_index = int(np.floor(0.6 * random_x.shape[0]))
            validation_index = int(np.floor(0.8 * random_x.shape[0]))
            training_set = random_x[:train_index, :]
            validation_set = random_x[train_index:validation_index, :]
            min_1 = np.nanmin(training_set[:, 0])
            max_1 = np.nanmax(training_set[:, 0])
            min_2 = np.nanmin(training_set[:, 1])
            max_2 = np.nanmax(training_set[:, 1])
            min_3 = np.nanmin(training_set[:, 2])
            max_3 = np.nanmax(training_set[:, 2])
            d1 = np.linspace(min_1, max_1, 150)
            d2 = np.linspace(min_2, max_2, 150)
            d3 = np.linspace(min_3, max_3, 150)
            if n_components == 5:
                min_4 = np.nanmin(training_set[:, 3])
                max_4 = np.nanmax(training_set[:, 3])
                min_5 = np.min(training_set[:, 4])
                max_5 = np.max(training_set[:, 4])
                d4 = np.linspace(min_4, max_4, 10)
                d5 = np.linspace(min_5, max_5, 10)
                x0, y0, z0, w0, v0 = np.meshgrid(d1, d2, d3, d4, d5, indexing='ij')
                XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel(), w0.ravel(), v0.ravel()))
            else:
                x0, y0, z0 = np.meshgrid(d1, d2, d3, indexing='ij')
                XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel()))
            X = training_set[:, :n_components]
            Y = training_set[:, n_components]
            YI = scipy.interpolate.griddata(X, Y, XI, method='linear')
            YI = YI.astype(float)
            if n_components == 3:
                k = np.ones((n_components, n_components, n_components))
                k[1, 1, 1] = 0
                k /= k.sum()
            elif n_components == 5:
                k = np.ones((n_components, n_components, n_components, n_components, n_components))
                k[2, 2, 2, 2, 2] = 0
                k /= k.sum()
            YI = np.reshape(YI, x0.shape)
            YI = scipy.signal.convolve(YI, k, mode='same', method='direct')
            YI = scipy.signal.convolve(YI, k, mode='same', method='direct')
            if np.isnan(YI).all():
                raise Exception("Model building failed! reason: not enough "
                                "input data or not enough times in the dataset in which the target was reached")
            Ix = np.digitize(validation_set[:, 0], d1) - 1
            Iy = np.digitize(validation_set[:, 1], d2) - 1
            Iz = np.digitize(validation_set[:, 2], d3) - 1
            Ix[np.isnan(Ix)] = np.round(np.nanmean(Ix))
            Iy[np.isnan(Iy)] = np.round(np.nanmean(Iy))
            Iz[np.isnan(Iz)] = np.round(np.nanmean(Iz))
            if n_components == 5:
                Iw = np.digitize(validation_set[:, 3], d4) - 1
                Iv = np.digitize(validation_set[:, 4], d5) - 1
                Iw[np.isnan(Iw)] = np.round(np.nanmean(Iw))
                Iv[np.isnan(Iv)] = np.round(np.nanmean(Iv))
            tfas_real = np.zeros((validation_set.shape[0]))
            tfas_predict = np.zeros((validation_set.shape[0]))
            for j in range(validation_set.shape[0]):
                tfas_real[j] = validation_set[j, n_components]
                if n_components == 3:
                    tfas_predict[j] = YI[int(Ix[j]), int(Iy[j]), int(Iz[j])]
                elif n_components == 5:
                    tfas_predict[j] = YI[int(Ix[j]), int(Iy[j]), int(Iz[j]), int(Iw[j]), int(Iv[j])]
                if np.isnan(tfas_predict[j]):
                    tfas_predict[j] = np.nanmean(Y)

            tfas_predict_mat[i, :] = tfas_predict
            tfas_actually_mat[i, :] = tfas_real
            mean_error_mat[i, :] = np.abs(tfas_real - tfas_predict)
        return YI, tfas_predict_mat, tfas_actually_mat, mean_error_mat,random_x

    @staticmethod
    def draw_stochastic_landscape_2d(random_x: np.array, save_path,n_components):
        train_index = int(np.floor(0.8 * random_x.shape[0]))
        training_set = random_x[:train_index, :]
        min_1 = np.nanmin(training_set[:, 0])
        max_1 = np.nanmax(training_set[:, 0])
        min_2 = np.nanmin(training_set[:, 1])
        max_2 = np.nanmax(training_set[:, 1])
        d1 = np.linspace(min_1, max_1, 150)
        d2 = np.linspace(min_2, max_2, 150)
        x0, y0 = np.meshgrid(d1, d2, indexing='ij')
        X = training_set[:, :2]
        Y = training_set[:, n_components]
        XI = np.column_stack((x0.ravel(), y0.ravel()))
        YI = scipy.interpolate.griddata(X, Y, XI, method='linear')
        YI = YI.astype(float)
        k = np.ones((2, 2))
        k[1, 1] = 0
        k /= k.sum()
        YI = np.reshape(YI, x0.shape)
        YI = scipy.signal.convolve(YI, k, mode='same', method='direct')
        YI = scipy.signal.convolve(YI, k, mode='same', method='direct')
        YI[np.isnan(YI)] = np.log(5 * 10 ** 3)
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(YI, cmap='jet')
        fig.colorbar(im, ax=ax)
        ax.set_title('Predictor Space in 2D')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(os.path.join(save_path, 'stochastic_landscape_2d.png'))
        plt.close(fig)

    @staticmethod
    def model_eval(tfas_predict_mat, tfas_actually_mat, cv_num, save_path):
        bin_width = 0.5
        min_of_all = np.nanmin(np.nanmin(tfas_predict_mat, axis=1))
        max_of_all = np.nanmax(np.nanmax(tfas_predict_mat, axis=1))
        n_bins = int(np.ceil(max_of_all / bin_width) - np.floor(min_of_all / bin_width) + 1)
        hist_space = np.linspace(bin_width * np.floor(min_of_all / bin_width),
                                 bin_width * np.ceil(max_of_all / bin_width),
                                 n_bins)
        mean = np.zeros((cv_num, len(hist_space) - 1))
        std = np.zeros((cv_num, len(hist_space) - 1))
        color_map = plt.cm.jet(np.linspace(0, 1, cv_num))
        color_map = color_map[:, :-1]
        fig_1 = plt.figure(1, figsize=(10, 7))  # grap 1
        a = fig_1.add_subplot(2, 1, 1)  # grap 1 subplot 1
        b = fig_1.add_subplot(2, 1, 2)  # grap 1 subplot 2
        x_ticks = np.arange(min_of_all, max_of_all + bin_width, bin_width)
        max_labels = np.nanmax(np.nanmax(tfas_actually_mat, axis=0))
        if max_of_all > max_labels:
            y_ticks = np.arange(min_of_all, max_of_all + 3, 3)
        else:
            y_ticks = np.arange(min_of_all, max_labels + 3, 3)
        for i in range(cv_num):
            x = np.squeeze(tfas_predict_mat[i, :])
            y = np.squeeze(tfas_actually_mat[i, :])
            sorted_indices = np.argsort(x)
            x = x[sorted_indices]
            y = y[sorted_indices]
            a.scatter(x, y, color=color_map[i, :], marker='o')  # grap 1 subplot 1
            a.set_xticks(x_ticks)
            a.set_yticks(y_ticks)
            a.set_xlim(x_ticks[0], x_ticks[-1])
            a.set_ylim(y_ticks[0], y_ticks[-1])
            a.set_ylabel("True Value")
            a.set_xlabel("Predicted Value")
            for n in range(1, len(x_ticks)):
                a.axvline(x_ticks[n - 1], color='k', linestyle='--', linewidth=0.5)
            std_hista = np.zeros(len(hist_space) - 1)
            mean_hista = np.zeros(len(hist_space) - 1)
            for j in range(1, len(hist_space)):
                Ind = np.where((hist_space[j - 1] < x) & (x < (hist_space[j])))[0]
                if len(Ind) < np.floor(0.01 * len(x)):
                    std_hista[j - 1] = np.nan
                    mean_hista[j - 1] = np.nan
                else:
                    yo = np.sort(y[Ind])
                    mean_hista[j - 1] = np.nanmedian(y[Ind])
                    aop = int(np.ceil(0.16 * len(yo)))
                    std_hista[j - 1] = mean_hista[j - 1] - yo[aop]
            x_hist_space = (hist_space[1:] + hist_space[:-1]) / 2
            i_4 = len(mean_hista) - 1
            b.scatter(x_hist_space[:i_4], mean_hista[:i_4], marker='s', color=color_map[i, :],
                      zorder=1)  # grap 1 subplot 2
            b.set_xticks(x_ticks)
            b.set_yticks(y_ticks)
            b.set_xlim(x_ticks[0], x_ticks[-1])
            b.set_ylim(y_ticks[0], y_ticks[-1])
            mean[i, :len(mean_hista)] = mean_hista
            std[i, :len(std_hista)] = std_hista
            i_5 = np.argmin(np.abs(x - x_hist_space[i_4]))
            a.scatter(x[:i_5], y[:i_5], color=color_map[i, :], marker='o')
        mean_vec = np.zeros(len(hist_space) - 1)
        std_vec = np.zeros(len(hist_space) - 1)
        mean[mean == 0] = np.nan
        std[std == 0] = np.nan
        for row in range(mean.shape[1]):
            mean_row = mean[:, row]
            std_row = std[:, row]
            nan = ~np.isnan(mean_row)
            mean_row = mean_row[nan]
            std_row = std_row[nan]
            if np.sum(~np.isnan(mean_row)) > 0:
                mean_vec[row] = np.nanmean(mean_row)
                std_vec[row] = np.nanstd(std_row)
            else:
                mean_vec[row] = np.nan
                std_vec[row] = np.nan
        b.plot(x_hist_space[:i_4], x_hist_space[:i_4], linestyle='--', color='k', linewidth=0.5, zorder=2,label="Perfect Predictor")
        b.errorbar(x_hist_space, mean_vec, yerr=std_vec, ecolor='b', zorder=3,label="Mean Predictor across iterations")
        for k in range(1, len(x_ticks)):
            b.axvline(x_ticks[k - 1], color='k', linestyle='--', linewidth=0.5)
            x_pos = (x_ticks[k - 1] + x_ticks[k]) / 2
            y_pos = y_ticks[-1] - 2
            b.text(x_pos, y_pos, f"Bin {k}", ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7),
                   zorder=4)
        b.set_ylabel("Scatter Plot Average for each bin")
        b.set_xlabel("Center of Predictor bin")
        b.legend(loc='upper right')
        fig_1.savefig(os.path.join(save_path, 'predictions_scatter_and_hist.png'))
        plt.close(fig_1)
        return mean_vec, std_vec, hist_space, x_hist_space, x_ticks, y_ticks

    @staticmethod
    def train_again_on_validation_and_test(a_reduced, n_components=3):
        np.random.seed(42)
        idn = np.where(a_reduced[:, n_components] != 0)[0]
        mapx = a_reduced[idn, :]
        mapx[:, n_components] = np.log(mapx[:, n_components])
        random_x = mapx[np.random.permutation(mapx.shape[0]), :]
        validation_index = int(np.ceil(0.9 * mapx.shape[0]))
        training_set = random_x[:validation_index, :]
        validation_set = random_x[validation_index:, :]
        min_1 = np.nanmin(training_set[:, 0])
        max_1 = np.nanmax(training_set[:, 0])
        min_2 = np.nanmin(training_set[:, 1])
        max_2 = np.nanmax(training_set[:, 1])
        min_3 = np.nanmin(training_set[:, 2])
        max_3 = np.nanmax(training_set[:, 2])
        d1 = np.linspace(min_1, max_1, 150)
        d2 = np.linspace(min_2, max_2, 150)
        d3 = np.linspace(min_3, max_3, 150)
        if n_components == 5:
            min_4 = np.nanmin(training_set[:, 3])
            max_4 = np.nanmax(training_set[:, 3])
            min_5 = np.min(training_set[:, 4])
            max_5 = np.max(training_set[:, 4])
            d4 = np.linspace(min_4, max_4, 10)
            d5 = np.linspace(min_5, max_5, 10)
            x0, y0, z0, w0, v0 = np.meshgrid(d1, d2, d3, d4, d5, indexing='ij')
            XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel(), w0.ravel(), v0.ravel()))
        else:
            x0, y0, z0 = np.meshgrid(d1, d2, d3, indexing='ij')
            XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel()))
        X = training_set[:, :n_components]
        Y = training_set[:, n_components]
        YI = scipy.interpolate.griddata(X, Y, XI, method='linear')
        YI = YI.astype(float)
        if n_components == 3:
            k = np.ones((n_components, n_components, n_components))
            k[1, 1, 1] = 0
            k /= k.sum()
        elif n_components == 5:
            k = np.ones((n_components, n_components, n_components, n_components, n_components))
            k[2, 2, 2, 2, 2] = 0
            k /= k.sum()
        YI = np.reshape(YI, x0.shape)
        YI = scipy.signal.convolve(YI, k, mode='same', method='direct')
        YI = scipy.signal.convolve(YI, k, mode='same', method='direct')
        if np.isnan(YI).all():
            raise Exception("Model building failed! reason: not enough "
                            "input data or not enough times in the dataset in which the target was reached")
        Ix = np.digitize(validation_set[:, 0], d1) - 1
        Iy = np.digitize(validation_set[:, 1], d2) - 1
        Iz = np.digitize(validation_set[:, 2], d3) - 1
        Ix[np.isnan(Ix)] = np.round(np.nanmean(Ix))
        Iy[np.isnan(Iy)] = np.round(np.nanmean(Iy))
        Iz[np.isnan(Iz)] = np.round(np.nanmean(Iz))
        if n_components == 5:
            Iw = np.digitize(validation_set[:, 3], d4) - 1
            Iv = np.digitize(validation_set[:, 4], d5) - 1
            Iw[np.isnan(Iw)] = np.round(np.nanmean(Iw))
            Iv[np.isnan(Iv)] = np.round(np.nanmean(Iv))
        tfas_actually_mat_2 = np.zeros((validation_set.shape[0]))
        tfas_predict_mat_2 = np.zeros((validation_set.shape[0]))
        for j in range(validation_set.shape[0]):
            tfas_actually_mat_2[j] = validation_set[j, n_components]
            if n_components == 3:
                tfas_predict_mat_2[j] = YI[int(Ix[j]), int(Iy[j]), int(Iz[j])]
            elif n_components == 5:
                tfas_predict_mat_2[j] = YI[int(Ix[j]), int(Iy[j]), int(Iz[j]), int(Iw[j]), int(Iv[j])]
            if np.isnan(tfas_predict_mat_2[j]):
                tfas_predict_mat_2[j] = np.nanmean(Y)

        mean_error_mat_2 = np.abs(tfas_actually_mat_2 - tfas_predict_mat_2)
        tfas_predict_mat_2 = np.reshape(tfas_predict_mat_2, (len(tfas_predict_mat_2), 1))
        tfas_actually_mat_2 = np.reshape(tfas_actually_mat_2, (len(tfas_actually_mat_2), 1))
        mean_error_mat_2 = np.reshape(mean_error_mat_2, (len(mean_error_mat_2), 1))

        return tfas_predict_mat_2, tfas_actually_mat_2, mean_error_mat_2

    @staticmethod
    def multiple_boxplot(input_data, x_labels, legend_label, ax):
        L = len(input_data)
        positions = np.arange(1, L + 1)

        # Create box plot
        bp1 = ax.boxplot(input_data, positions=positions, patch_artist=True, widths=0.6)

        # Set x-axis ticks and labels
        ax.set_xticks(positions)
        # ax.set_xticklabels([f"{x:.2f}" for x in x_labels], rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xticklabels([f"{x:.2f}" for x in x_labels])
        ax.set_xlim(0.5, L + 0.5)

        # Customize box plots
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp1[element], color='black')

        for box in bp1['boxes']:
            box.set(facecolor='white', edgecolor='red', linewidth=2)

        for flier in bp1['fliers']:
            flier.set(marker='o', color='red', alpha=0.7)

        # Set labels and title
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel('True Value')

        # Add legend entries
        bp1["boxes"][0].set_label(legend_label)
        ax.plot(positions, x_labels, linestyle='--', color='k', label='Perfect Predictor')

        # Add legend
        ax.legend(loc='upper right')
        plt.gcf().subplots_adjust(bottom=0.3)
        # Adjust y-axis limits
        all_values = [val for sublist in input_data for val in sublist]
        y_min, y_max = min(all_values), max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    @staticmethod
    def plot_helper(ax, data, box_positions, color, legend_label, n_boxes_last_group):
        bp = ax.boxplot(data, positions=np.round(box_positions, 2), patch_artist=True)
        # Apply colors
        if n_boxes_last_group > 0:
            for i, box in enumerate(bp['boxes'][:n_boxes_last_group], start=n_boxes_last_group):
                box.set(facecolor='white', edgecolor=color, linewidth=2)
        else:
            for i, box in enumerate(bp['boxes']):
                box.set(facecolor='white', edgecolor=color, linewidth=2)

        for i, element in enumerate(
                bp['medians'] + bp['whiskers'] + bp['caps']):
            element.set_color('black')

        for i, element in enumerate(bp['fliers']):
            element.set_markeredgecolor('black')
        bp["boxes"][n_boxes_last_group - 1].set_label(legend_label)

    @staticmethod
    def multiple_boxplot_2(data, x_labels, legend_labels, colormap, ax, x_hist_space):
        # Get sizes
        M, L = len(data), len(data[0])
        w = 0.25  # width of boxes
        positions = np.arange(1, M * L * w + 1 + w * L, w)
        positions = positions[:(M * L)]
        YY = x_labels[0] - 1 + positions - w / 2

        # Extract data and label it in the group correctly
        x = []
        group = []
        for i in range(M):
            for j in range(L):
                aux = data[i][j]
                x.extend(aux)
                group.extend([j + i * M] * len(aux))
        x_lims = [np.min(YY) - 0.5, np.max(YY) + 0.5]
        box_positions1 = YY[::2]
        box_positions2 = YY[1::2]
        data1 = data[0]
        data2 = data[1]
        legend_label1 = legend_labels[0]
        legend_label2 = legend_labels[1]
        SLM_tools.plot_helper(ax, data1, box_positions1, 'red', legend_label1, 0)
        SLM_tools.plot_helper(ax, data2, box_positions2, 'blue', legend_label2, len(data1))

        all_values = [val for sublist in data for val in sublist][0]
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Predictor Error")
        ax.axhline(y=0, linestyle='--', color='black')
        ax.scatter(x_hist_space, np.ones(x_hist_space.shape) * (np.max(all_values) + 1), color=colormap, s=100,
                   edgecolors='black', label='Relative Weight of Bin')
        ax.legend(loc='lower left', fontsize=6)
        ax.set_xlim(x_lims)

    @staticmethod
    def normalize_array(arr):
        median = np.median(arr)
        return arr - 2 * median if median < 0 else arr

    @staticmethod
    def cv_bias_correction(tfas_predict_mat_2, tfas_actually_mat_2, hist_space, mean_vec, x_hist_space,
                           x_ticks, y_ticks, save_path):
        bin_width = 0.5
        median_fit_vec = np.nanmedian(np.nanmedian(tfas_predict_mat_2))
        x = np.squeeze(tfas_predict_mat_2)
        y = np.squeeze(tfas_actually_mat_2)
        tfas_predict_mat_2_sorted_indices = np.argsort(x)
        x = x[tfas_predict_mat_2_sorted_indices]
        y = y[tfas_predict_mat_2_sorted_indices]
        std_hista_origin = np.full(len(hist_space) - 1, np.nan)
        mean_hista_origin = np.full(len(hist_space) - 1, np.nan)
        std_hista = np.full(len(hist_space) - 1, np.nan)
        mean_hista = np.full(len(hist_space) - 1, np.nan)
        cutofflength = len(mean_vec)
        occurrence_probability = np.zeros((cutofflength, 1))
        cv_offset = mean_vec - x_hist_space
        cv_offset[np.isnan(cv_offset)] = 0
        cv_corrected_x = np.zeros_like(x)
        new_y = np.zeros(y.shape)
        fig_2 = plt.figure(2,figsize=(7, 14))
        a = fig_2.add_subplot(3, 1, 1)
        b = fig_2.add_subplot(3, 1, 2)
        c = fig_2.add_subplot(3, 1, 3)
        scat_1 = a.scatter(x, y, color=(0.7, 0.7, 0.7), marker='s',)
        a.set_ylabel("True Value")
        a.set_xlabel("Predicted Value")
        a.set_xticks(x_ticks)
        a.set_yticks(y_ticks)
        a.set_xlim(x_ticks[0], x_ticks[-1])
        a.set_ylim(y_ticks[0], y_ticks[-1])
        for i in range(1, cutofflength - 1):
            Ind = np.where((hist_space[i - 1] < x) & (x < (hist_space[i])))[0]
            if len(Ind) < np.floor(0.01 * len(x)):
                std_hista[i - 1] = np.nan
                mean_hista[i - 1] = np.nan
                occurrence_probability[i - 1] = len(Ind)
                cv_corrected_x[Ind] = x[Ind] + cv_offset[i - 1]
                new_y[Ind] = y[Ind]
                scat_1 = a.scatter(cv_corrected_x[Ind], y[Ind], color='k', marker='s')
            else:
                yo = np.sort(y[Ind])
                z0 = -(np.exp(median_fit_vec) - yo)  # predictor error
                z1 = -x_hist_space[i - 1] - yo + cv_offset[i - 1]  # cv_corrected predictor error
                mean_hista[i - 1] = np.median(z1)
                std_hista[i - 1] = np.sqrt(np.sum((z1 - np.median(z1)) ** 2) / len(yo))
                mean_hista_origin[i - 1] = np.median(z0)
                std_hista_origin[i - 1] = np.sqrt(np.sum((z0 - np.median(z0)) ** 2) / len(yo))
                occurrence_probability[i - 1] = len(Ind)
                cv_corrected_x[Ind] = x[Ind] + cv_offset[i - 1]
                scat_2 = a.scatter(cv_corrected_x[Ind], y[Ind], color='k', marker='s')
                new_y[Ind] = y[Ind]
        sorted_indices = np.argsort(cv_corrected_x)
        new_y_sorted = new_y[sorted_indices]
        cv_corrected_x_sorted = cv_corrected_x[sorted_indices]
        slope, intercept, r_value, p_value, std_err = linregress(cv_corrected_x_sorted, new_y_sorted)
        R_x = np.array([cv_corrected_x_sorted[0], new_y_sorted[-1]])
        R_y = intercept + slope * R_x
        a.plot(R_x, R_y, color='b', linestyle='--', label='Linear Regression')
        scat_1.set_label("Original Predictor")
        scat_2.set_label("CV Bias Corrected Predictor")
        a.legend()
        new_x = np.copy(cv_corrected_x_sorted)
        min_of_all2 = np.min(new_x)
        max_of_all2 = np.max(new_x)
        hist_space_2 = np.linspace(bin_width * np.floor(min_of_all2 / bin_width),
                                   bin_width * np.ceil(max_of_all2 / bin_width),
                                   int(np.ceil(max_of_all2 / bin_width) - np.floor(min_of_all2 / bin_width) + 1))
        x_hist_space_2 = (hist_space_2[1:] + hist_space_2[:-1]) / 2
        cutofflength2 = len(hist_space_2) - 1
        mean_hista_last_fig2 = np.full(cutofflength2, np.nan)
        mean_histaria = np.full(cutofflength2, np.nan)
        mean_histaria_origin = np.full(cutofflength2, np.nan)
        std_histaria_origin = np.full(cutofflength2, np.nan)
        std_histaria = np.full(cutofflength2, np.nan)
        occurrence_probability_r = np.full(cutofflength2, np.nan)
        yo_lst = []
        z1_lst = []
        z0_lst = []
        mega_kde = []
        for i in range(1, cutofflength2 + 1):
            Ind = np.where((hist_space_2[i - 1] < new_x) & (new_x < (hist_space_2[i])))[0]
            if len(Ind) < 5:
                occurrence_probability_r[i - 1] = len(Ind)
            else:
                yo = np.sort(new_y_sorted[Ind])
                mean_hista_last_fig2[i - 1] = np.median(yo)
                z0 = -(median_fit_vec - yo)  # trivial predictor error
                z1 = -((x_hist_space_2[i - 1]) - yo)  # post information predictor error
                mean_histaria[i - 1] = np.median(z1)
                mean_histaria_origin[i - 1] = np.median(z0)
                std_histaria_origin[i - 1] = np.sqrt(np.sum((z0 - np.median(z0)) ** 2) / len(yo))
                std_histaria[i - 1] = np.sqrt(np.sum((z1 - np.median(z1)) ** 2) / len(yo))
                occurrence_probability_r[i - 1] = len(Ind)
                mega_kde.append([z1, z0, yo, i - 1])
                z1_lst.append(z1)
                z0_lst.append(z0)
                yo_lst.append(yo)
        occurrence_probability_r = occurrence_probability_r / np.sum(occurrence_probability_r)
        I2 = (~np.isnan(x_hist_space_2)) & (~np.isnan(mean_hista_last_fig2))
        SLM_tools.multiple_boxplot(input_data=yo_lst, x_labels=x_hist_space_2[I2],
                                   legend_label="CV Bias Corrected Predictor", ax=b)
        z0_lst = [SLM_tools.normalize_array(z0) for z0 in z0_lst]
        z1_lst = [SLM_tools.normalize_array(z1) for z1 in z1_lst]
        data = [z0_lst, z1_lst]
        ind = np.round(occurrence_probability_r * 1000) + 1
        ind[ind < 1] = 1
        ind = np.ceil(1000 * np.log(ind) / np.log(1000))
        ind[ind < 1] = 1
        ind = np.squeeze(ind)
        colormap = plt.cm.get_cmap('cool')
        colormap = colormap(np.linspace(0, 1, 1000))
        colormap = colormap[ind.astype(int), :]
        SLM_tools.multiple_boxplot_2(data=data,
                                     x_labels=x_hist_space_2[I2],
                                     legend_labels=["CV Bias Corrected Predictor Mean Error",
                                                    "Naive (median) Predictor Mean Error"],
                                     colormap=colormap,
                                     ax=c,
                                     x_hist_space=x_hist_space_2
                                     )
        fig_2.colorbar(plt.cm.ScalarMappable(cmap='cool'), ax=c)
        fig_2.savefig(os.path.join(save_path, 'predictor_eval.png'), bbox_inches='tight')
        plt.close(fig_2)

    @staticmethod
    def create_and_evaluate_stochastic_landscape(dir_path, n_components, downsampling_factor, target_num,
                                                 time_vec_exists, cv_num, save_path, data_variable_name):
        current_datetime = datetime.now().strftime("%d_%m_%H_%M")
        total_data_files = len(os.listdir(dir_path))
        counters = {"load": [], "extraction": [], "segmentation": [],
                    "feature_selection": 0, "N_extracted_features": [],
                    "total_features_extracted": 0, "features_selected": 0,
                    "total_runtime": 0, "files_with_target": 0, "log": [],
                    }
        C = []
        for i, file in enumerate(os.listdir(dir_path)):
            file_path = os.path.abspath(os.path.join(dir_path, file))
            if os.path.isfile(file_path):
                file_name, file_extension = os.path.splitext(file)
                start1 = time.time()
                if not data_variable_name:
                    data_variable_name = 'foo'
                values_vec, distance_data = SLM_tools.load_data(file_path,data_variable_name, int(target_num), file_extension, time_vec_exists)
                if downsampling_factor:
                    values_vec, distance_data, time_vec = SLM_tools.downsample(values_vec, distance_data,
                                                                               downsampling_factor)
                values_vec = np.reshape(values_vec, (len(values_vec), 1))
                distance_data = np.reshape(distance_data, (len(values_vec), int(target_num)))
                loading_time = round(time.time() - start1, 3)
                log1 = f"\nFile {file_name}, {i + 1}/{total_data_files}  loaded, Loading time: {loading_time} seconds"
                counters["log"].append(log1)
                counters["load"].append(loading_time)
                print(log1)
                start2 = time.time()
                o, cp, mean_trend = SLM_tools.beast(values_vec)
                segmentation_time = round(time.time() - start2, 3)
                log2 = f"\nFinished Segmentation, runtime: {segmentation_time} seconds"
                counters["log"].append(log2)
                counters["segmentation"].append(segmentation_time)
                print(log2)
                start3 = time.time()
                A = SLM_tools.feature_extraction(values_vec, distance_data, mean_trend, cp)
                extraction_time = round(time.time() - start3, 3)
                log = f"\nFinished feature extraction {file_name}, {i + 1}/{total_data_files}, runtime: {extraction_time} seconds"
                counters["log"].append(log)
                counters["extraction"].append(extraction_time)
                print(log)
                log = f"\nTarget Reached? : {len(A) > 0}"
                counters["log"].append(log)
                print(log)
                if len(A) > 0:
                    counters["files_with_target"] += 1
                    counters["total_features_extracted"] += A[0].shape[0]
                    counters["N_extracted_features"].append(A[0].shape[0])
                    log3 = f"\nfiles_with_target: {counters['files_with_target']}/{total_data_files}"
                    counters["log"].append(log3)
                    log4 = f"\nTotal features_extracted: {counters['total_features_extracted']}"
                    counters["log"].append(log4)
                else:
                    log5 = f"\nTarget Not Reached, No features extracted"
                    counters["log"].append(log5)
                C.append(A)
                counters["total_runtime"] += (loading_time + extraction_time + segmentation_time)
        log6 = f"\nFinished pre-processing,total runtime {counters['total_runtime']} for {total_data_files} files"
        counters["log"].append(log6)
        print(log6)
        start_feature_selection = time.time()
        c_reduced = SLM_tools.feature_selection(C)
        principal_components, score, latent = SLM_tools.pca(c_reduced, n_components)
        a_reduced = SLM_tools.data_preparation(score, c_reduced, n_components)
        counters["selected_features"] = a_reduced.shape[0]
        runtime_feature_selection = round(time.time() - start_feature_selection, 3)
        counters["feature_selection"] = runtime_feature_selection
        counters["total_runtime"] += runtime_feature_selection
        log9 = f"\nFinished Feature Selection runtime {runtime_feature_selection} seconds"
        counters["log"].append(log9)
        print(log9)
        selected_features_path = os.path.join(save_path,f"selected_features_{current_datetime}.mat")
        sio.savemat(selected_features_path, {'selected_features': a_reduced})
        log10 = f"\nSaved all selected_features"
        print(log10)
        counters["log"].append(log10)
        counters["log"].append("SUMMARY")
        log11 = (f"\nFiles {total_data_files}"
                 f"\nFiles Where the Target was Reached {counters['files_with_target']}"
                 f"\nMean Extracted Features per file {np.mean(counters['N_extracted_features'])} "
                 f"\nTotal Features Extracted {counters['total_features_extracted']}"
                 f"\n{counters['selected_features']} Features Selected"
                 f"\nMean loading time per file {np.mean(counters['load'])} seconds"
                 f"\nMean segmentation time per file {np.mean(counters['segmentation'])} seconds"
                 f"\nMean feature extraction time per file {np.mean(counters['extraction'])} seconds"
                 f"\nFeature selection time {counters['feature_selection']} seconds"
                 f"\nTotal Runtime {counters['total_runtime']} seconds ")
        counters["log"].append(log11)
        print(log11)
        log_path = os.path.join(save_path, f"preprocessing_log_{current_datetime}.txt")
        with open(log_path, 'w') as f:
            for line in counters["log"]:
                f.write(line)
        print("Training start")
        YI, tfas_predict_mat, tfas_actually_mat, mean_error_mat,random_x = SLM_tools.model_training_with_cv(a_reduced,
                                                                                                            n_components,
                                                                                                            int(cv_num))
        SLM_tools.draw_stochastic_landscape_2d(random_x, save_path, n_components)
        print("Eval 1 start")
        mean_vec, std_vec, hist_space, x_hist_space, x_ticks, y_ticks = SLM_tools.model_eval(tfas_predict_mat,
                                                                                             tfas_actually_mat, int(cv_num),
                                                                                             save_path)
        tfas_predict_mat_2, tfas_actually_mat_2, mean_error_mat_2 = SLM_tools.train_again_on_validation_and_test(a_reduced, n_components)
        print("Eval 2 start")
        SLM_tools.cv_bias_correction(tfas_predict_mat_2, tfas_actually_mat_2,hist_space, mean_vec, x_hist_space,x_ticks, y_ticks, save_path)