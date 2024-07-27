import pickle
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


class SLM_Manager:
    def __init__(self, *args):  # user inputs
        pass


class SLM_tools:
    @staticmethod
    def load_data(data_path: str, target_num, data_type='csv', time_vec_exists=False):
        # TODO: incomplete, try with more then two targets
        try:
            if data_type == 'csv':
                values_vec = pd.read_csv(data_path, usecols=[0], header=None).to_numpy()
                if time_vec_exists:
                    time_vec = pd.read_csv(data_path, usecols=[1]).to_numpy()
                    distance_columns = list(range(2, 2 + target_num))
                    distance = pd.read_csv(data_path, usecols=distance_columns, header=None).to_numpy()
                    return values_vec, time_vec, distance
                distance_columns = list(range(1, 1 + target_num))
                distance = pd.read_csv(data_path, usecols=distance_columns, header=None).to_numpy()
                return values_vec, distance
            elif data_type == 'excel':
                values_vec = pd.read_excel(data_path, usecols=[1], header=None).to_numpy()
                if time_vec_exists:
                    time_vec = pd.read_excel(data_path, usecols=[0], header=None).to_numpy()
                    distance_columns = list(range(2, 2 + target_num))
                    distance = pd.read_excel(data_path, usecols=distance_columns, header=None).to_numpy()
                    return values_vec, time_vec, distance
                distance_columns = list(range(1, 1 + target_num))
                distance = pd.read_excel(data_path, usecols=distance_columns, header=None).to_numpy()
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
            print(e)

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
            # downsampled_data = signal.decimate(data, downsampling_factor, ftype='fir')
            # downsampled_time = signal.decimate(time_vec, downsampling_factor, ftype='fir') if time_vec else None
            # downsampled_distance = signal.decimate(distance, downsampling_factor, ftype='fir')
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
                         print_options=False, print_progress=False)
            mean_trend = o.trend.Y
            cp = np.sort(o.trend.cp[0:int(o.trend.ncp_median)])
            cp = cp[~np.isnan(cp)]
            cp = np.insert(cp, 0, 0)
            return o, cp, mean_trend
        except Exception as e:
            print(e)

    @staticmethod
    def segment_data(energy: np.array, distance: np.array, mean_trend: np.array, cp: np.array, Number_of_targets=None):
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
            std[i] = np.std(segment, axis=0)
            median = np.median(segment, axis=0)
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

        # TODO: all returns of segment_data of all files should v_stack before post processing and model building

    @staticmethod
    def post_beast_processing(segment_data_aggregated_output: np.array):
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
            else:
                ending_theme = len(c)
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
        return principal_components, score, latent

    @staticmethod
    def post_pca_processing(score: np.array, c_reduced: np.array, n_components: int = 3):
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

    # @staticmethod
    # def trajectory_plot_vecs(mean_vec, std_vec, trend, time_to_self_assembly, save_path: str, bottom=0, top=3000):
    #     # not checked
    #     sz = 40
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(mean_vec.T, std_vec.T, trend.T, s=sz, c=time_to_self_assembly, cmap='jet', marker='o')
    #     ax.set_xlim(bottom, top)
    #     ax.set_ylim(bottom, top)
    #     ax.set_zlim(bottom, top)
    #     d = (np.vstack((mean_vec[1:], std_vec[1:], trend[1:])) - np.vstack(
    #         (mean_vec[:-1], std_vec[:-1], trend[:-1]))) / 2
    #     color_triplet = np.random.rand(1, 3)
    #     ax.quiver(mean_vec[:-1], std_vec[:-1], trend[:-1], d[0], d[1], d[2], color=color_triplet, length=0.1,
    #               normalize=True)
    #     ax.scatter(mean_vec[0], std_vec[0], trend[0], s=1 * sz, c=time_to_self_assembly[0], marker='^',
    #                label='Start Point')
    #     ax.scatter(mean_vec[-1], std_vec[-1], trend[-1], s=1 * sz, c=time_to_self_assembly[-1], marker='d',
    #                label='Finish Point')
    #     ax.legend(['Points', 'Arrows', 'Trajectory', 'Start Point', 'Finish Point'])
    #     ax.set_xlabel('Mean')
    #     ax.set_ylabel('Standard Deviation')
    #     ax.set_zlabel('Trend')
    #     ax.set_title('Object Position')
    #     ax.grid(True)
    #     plt.show()
    #     # plt.savefig(save_path, bbox_inches='tight')

    @staticmethod
    def replace_nan_with_rounded_mean(array):
        mean_val = np.nanmean(array)
        array[np.isnan(array)] = np.round(mean_val)
        return array

    @staticmethod
    def model_training_with_cv(a_reduced: np.array, n_components: int = 3, cv_num: int = 3):
        # from LogTfas....
        np.random.seed(42)
        idn = np.where(a_reduced[:, n_components] != 0)[0]
        mapx = a_reduced[idn, :]
        mapx[:, n_components] = np.log(mapx[:, n_components])
        median_fit_vec = np.median(mapx[:, n_components], axis=0)  # line 54 matlab
        # xw = mapx[:, n_components]
        # points = np.linspace(min, max, 1000)
        # f_x = scipy.stats.gaussian_kde(points)  # line 60 matlab
        # p_x = f_x[xw]
        yo = np.sort(mapx[:, n_components])
        aop = int(np.ceil((0.16 * len(yo)))) - 1
        std_fit = np.nanmedian(mapx[:, n_components], axis=0) - yo[aop]
        x = np.copy(mapx)
        train_index = int(np.floor(0.6 * x.shape[0]))
        validation_index = int(np.floor(0.8 * x.shape[0]))
        size_validation_set = len(range(train_index, validation_index))
        tfas_predict_mat = np.zeros((cv_num, size_validation_set))
        tfas_actually_mat = np.zeros((cv_num, size_validation_set))
        mean_error_mat = np.zeros((cv_num, size_validation_set))
        for i in range(cv_num):
            random_x = x[np.random.permutation(x.shape[0]), :]  # line 89 matlab
            train_index = int(np.floor(0.6 * random_x.shape[0]))
            validation_index = int(np.floor(0.8 * random_x.shape[0]))
            training_set = random_x[:train_index, :]
            validation_set = random_x[train_index:validation_index, :]
            min_1 = np.min(training_set[:, 0])
            max_1 = np.max(training_set[:, 0])
            min_2 = np.min(training_set[:, 1])
            max_2 = np.max(training_set[:, 1])
            min_3 = np.min(training_set[:, 2])
            max_3 = np.max(training_set[:, 2])
            d1 = np.linspace(np.floor(min_1), np.ceil(max_1), 150)
            d2 = np.linspace(np.floor(min_2), np.ceil(max_2), 150)
            d3 = np.linspace(np.floor(min_3), np.ceil(max_3), 150)
            if n_components == 5:
                min_4 = np.min(training_set[:, 3])
                max_4 = np.max(training_set[:, 3])
                min_5 = np.min(training_set[:, 4])
                max_5 = np.max(training_set[:, 4])
                d4 = np.linspace(np.floor(min_4), np.ceil(max_4), 10)
                d5 = np.linspace(np.floor(min_5), np.ceil(max_5), 10)
                x0, y0, z0, w0, v0 = np.meshgrid(d1, d2, d3, d4, d5, indexing='ij')
                X = training_set[:, :n_components]
                Y = training_set[:, n_components]
                XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel(), w0.ravel(), v0.ravel()))
            else:
                x0, y0, z0 = np.meshgrid(d1, d2, d3, indexing='ij')
                X = training_set[:, :3]
                Y = training_set[:, 3]
                XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel()))
            YI = scipy.interpolate.griddata(X, Y, XI, method='linear')
            YI = YI.astype(float)
            non_nan_mask = ~np.isnan(YI)
            values = YI[non_nan_mask]
            intergal_dist = 3
            k = np.ones(intergal_dist) / intergal_dist
            k[0] = 0
            averageIntensities = scipy.ndimage.convolve(values, k, mode='constant', cval=0.0)
            averageIntensities = scipy.ndimage.convolve(averageIntensities, k, mode='constant', cval=0.0)
            YI[non_nan_mask] = averageIntensities
            # averageIntensities[~mask] = np.nan
            YI = np.reshape(YI, x0.shape)
            Ix = pd.cut(validation_set[:, 0], bins=d1, labels=False, include_lowest=True)
            Iy = pd.cut(validation_set[:, 1], bins=d2, labels=False, include_lowest=True)
            Iz = pd.cut(validation_set[:, 2], bins=d3, labels=False, include_lowest=True)
            Ix = SLM_tools.replace_nan_with_rounded_mean(Ix.astype(float))
            Iy = SLM_tools.replace_nan_with_rounded_mean(Iy.astype(float))
            Iz = SLM_tools.replace_nan_with_rounded_mean(Iz.astype(float))
            if n_components == 5:
                Iw = pd.cut(validation_set[:, 3], bins=d4, labels=False, include_lowest=True)
                Iv = pd.cut(validation_set[:, 4], bins=d5, labels=False, include_lowest=True)
                Iw = SLM_tools.replace_nan_with_rounded_mean(Iw.astype(float))
                Iv = SLM_tools.replace_nan_with_rounded_mean(Iv.astype(float))
            tfas_real = np.zeros((validation_set.shape[0]))
            tfas_predict = np.zeros((validation_set.shape[0]))
            for j in range(validation_set.shape[0]):
                tfas_real[j] = validation_set[j, n_components]
                if n_components == 3:
                    tfas_predict[j] = YI[int(Ix[j]), int(Iy[j]), int(Iz[j])]
                else:
                    tfas_predict[j] = YI[int(Ix[j]), int(Iy[j]), int(Iz[j]), int(Iw[j]), int(Iv[j])]
                if np.isnan(tfas_predict[j]):
                    tfas_predict[j] = np.mean(Y)

            tfas_predict_mat[i, :] = tfas_predict
            tfas_actually_mat[i, :] = tfas_real
            mean_error_mat[i, :] = np.abs(tfas_real - tfas_predict)
            print(f"finished iteration {i+1}/{cv_num}")
        return YI, tfas_predict_mat, tfas_actually_mat, mean_error_mat, train_index, random_x, validation_index, median_fit_vec

    @staticmethod
    def model_eval(tfas_predict_mat, tfas_actually_mat, cv_num, save_path):
        bin_width = 0.5
        min_of_all = int(np.min(np.nanmin(tfas_predict_mat,axis=0)))
        max_of_all = int(np.max(np.nanmax(tfas_predict_mat,axis=0)))
        n_bins = int(np.ceil(max_of_all/bin_width)-np.floor(min_of_all/bin_width)+1)
        # bin_width = abs(min_of_all-max_of_all)/num_bins
        hist_space = np.linspace(int(bin_width*np.floor(min_of_all/bin_width)),
                                 int(bin_width*np.ceil(max_of_all/bin_width)),
                                 n_bins)
        mean = np.zeros((cv_num, len(hist_space) - 1))
        std = np.zeros((cv_num, len(hist_space) - 1))
        color_map = plt.cm.jet(np.linspace(0, 1, cv_num))
        color_map = color_map[:, :-1]
        fig_1 = plt.figure(1,figsize=(10, 7))  # grap 1
        a = fig_1.add_subplot(2, 1, 1)  # grap 1 subplot 1
        b = fig_1.add_subplot(2, 1, 2)  # grap 1 subplot 2
        # x_ticks = np.arange(min_of_all, max_of_all-2, 0.5)
        # y_ticks = np.arange(min_of_all, max_of_all+3, 3)
        x_ticks = np.arange(min_of_all, max_of_all - bin_width, bin_width)
        y_ticks = np.arange(min_of_all, max_of_all + 3, 3)
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
            for n in range(1,len(x_ticks)):
                a.axvline(x_ticks[n-1], color='k', linestyle='--',linewidth=0.5)
            std_hista = np.zeros(len(hist_space) - 1)
            mean_hista = np.zeros(len(hist_space) - 1)
            for j in range(1, len(hist_space)):
                Ind = np.where((hist_space[j-1] < x) & (x < (hist_space[j])))[0]
                if len(Ind) < 5:
                    std_hista[j-1] = np.nan
                    mean_hista[j-1] = np.nan
                else:
                    yo = np.sort(y[Ind])
                    mean_hista[j-1] = np.median(y[Ind])
                    aop = int(np.ceil(0.16 * len(yo)))
                    std_hista[j-1] = mean_hista[j] - yo[aop]
            x_hist_space = (hist_space[1:] + hist_space[:-1]) / 2
            # x_hist_space_all[0, cv_num] = x_hist_space
            # i_9 = np.where(~np.isnan(mean_hista) & (x_hist_space > 0))[0][0] if np.any(
            #     ~np.isnan(mean_hista) & (x_hist_space > 0)) else len(mean_hista)
            # i_2 = np.where(np.isnan(mean_hista) & (x_hist_space > 0))[0][0] if np.any(
            #     np.isnan(mean_hista) & (x_hist_space > 0)) else len(mean_hista)
            # i_3 = np.where(np.isnan(std_hista) & (x_hist_space > 0))[0][0] if np.any(
            #     np.isnan(std_hista) & (x_hist_space > 0)) else len(mean_hista)
            # i_4 = min(i_2, i_3)
            # if i_4 is None:
            #     i_4 = len(mean_hista)-1
            # if i_9 < i_4:
            #     i_4 = min(i_2, i_3)
            # else:
            #     i_2 = np.where(np.isnan(mean_hista) & (x_hist_space > 0), i_9)[0][-1] if np.any(
            #         np.isnan(mean_hista) & (x_hist_space > 0)) else len(mean_hista)
            #     i_3 = np.where(np.isnan(std_hista) & (x_hist_space > 0), i_9)[0][-1] if np.any(
            #         np.isnan(std_hista) & (x_hist_space > 0)) else len(mean_hista)
            #     i_4 = min(i_2, i_3)
            # if i_4 is None:
            #     i_4 = len(mean_hista) - 1
            i_4 = len(mean_hista) - 1  # TODO: check in testing
            b.scatter(x_hist_space[:i_4], mean_hista[:i_4], marker='s', color=color_map[i, :])  # grap 1 subplot 2
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
                mean_vec[row] = np.mean(mean_row)
                std_vec[row] = np.std(std_row)
            else:
                mean_vec[row] = np.nan
                std_vec[row] = np.nan
        b.plot(x_hist_space[:i_4], x_hist_space[:i_4], linestyle='--', color='k')
        b.errorbar(x_hist_space, mean_vec, yerr=std_vec, xerr=std_vec,
                   ecolor='k')  # TODO: might crash, yerr and xerr are different in matlab
        for k in range(1, len(x_ticks)):
            b.axvline(x_ticks[k - 1], color='k', linestyle='--', linewidth=0.5)
            x_pos = (x_ticks[k-1] + x_ticks[k])/2
            y_pos = y_ticks[0] + 2
            b.text(x_pos, y_pos, f"Bin {k}", ha='center', va='bottom')
        b.set_ylabel("Scatter Plot Average for each bin")
        b.set_xlabel("Center of Predictor bin")
        fig_1.savefig(os.path.join(save_path, 'fig_1.png'))
        return mean_vec, std_vec, hist_space, x_hist_space, fig_1, a, b, x_ticks, y_ticks, bin_width

    @staticmethod
    def train_again_on_validation_and_test(random_x, validation_index, n_components=3):
        np.random.seed(42)
        random_x = random_x[np.random.permutation(random_x.shape[0]), :]  # line 89 matlab
        training_set = random_x[:validation_index - 1, :]
        test_set = random_x[validation_index:, :]
        min_1 = np.min(training_set[:, 0])
        max_1 = np.max(training_set[:, 0])
        min_2 = np.min(training_set[:, 1])
        max_2 = np.max(training_set[:, 1])
        min_3 = np.min(training_set[:, 2])
        max_3 = np.max(training_set[:, 2])
        d1 = np.linspace(np.floor(min_1), np.ceil(max_1), 150)
        d2 = np.linspace(np.floor(min_2), np.ceil(max_2), 150)
        d3 = np.linspace(np.floor(min_3), np.ceil(max_3), 150)
        if n_components == 5:
            min_4 = np.min(training_set[:, 3])
            max_4 = np.max(training_set[:, 3])
            min_5 = np.min(training_set[:, 4])
            max_5 = np.max(training_set[:, 4])
            d4 = np.linspace(np.floor(min_4), np.ceil(max_4), 10)
            d5 = np.linspace(np.floor(min_5), np.ceil(max_5), 10)
            x0, y0, z0, w0, v0 = np.meshgrid(d1, d2, d3, d4, d5, indexing='ij')
            X = training_set[:, :n_components]
            Y = training_set[:, n_components]
            XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel(), w0.ravel(), v0.ravel()))
        else:
            x0, y0, z0 = np.meshgrid(d1, d2, d3, indexing='ij')
            X = training_set[:, :3]
            Y = training_set[:, 3]
            XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel()))
        YI = scipy.interpolate.griddata(X, Y, XI, method='linear')
        YI = YI.astype(float)
        non_nan_mask = ~np.isnan(YI)
        values = YI[non_nan_mask]
        intergal_dist = 3
        k = np.ones(intergal_dist) / intergal_dist
        k[0] = 0
        averageIntensities = scipy.ndimage.convolve(values, k, mode='constant', cval=0.0)
        averageIntensities = scipy.ndimage.convolve(averageIntensities, k, mode='constant', cval=0.0)
        YI[non_nan_mask] = averageIntensities
        # averageIntensities[~mask] = np.nan
        YI = np.reshape(YI, x0.shape)
        Ix = pd.cut(test_set[:, 0], bins=d1, labels=False, include_lowest=True)
        Iy = pd.cut(test_set[:, 1], bins=d2, labels=False, include_lowest=True)
        Iz = pd.cut(test_set[:, 2], bins=d3, labels=False, include_lowest=True)
        Ix = SLM_tools.replace_nan_with_rounded_mean(Ix.astype(float))
        Iy = SLM_tools.replace_nan_with_rounded_mean(Iy.astype(float))
        Iz = SLM_tools.replace_nan_with_rounded_mean(Iz.astype(float))
        if n_components == 5:
            Iw = pd.cut(test_set[:, 3], bins=d4, labels=False, include_lowest=True)
            Iv = pd.cut(test_set[:, 4], bins=d5, labels=False, include_lowest=True)
            Iw = SLM_tools.replace_nan_with_rounded_mean(Iw.astype(float))
            Iv = SLM_tools.replace_nan_with_rounded_mean(Iv.astype(float))
        tfas_real = np.zeros((test_set.shape[0]))
        tfas_predict = np.zeros((test_set.shape[0]))
        for j in range(test_set.shape[0]):
            tfas_real[j] = test_set[j, n_components]
            if n_components == 3:
                tfas_predict[j] = YI[int(Ix[j]), int(Iy[j]), int(Iz[j])]
            else:
                tfas_predict[j] = YI[int(Ix[j]), int(Iy[j]), int(Iz[j]), int(Iw[j]), int(Iv[j])]
            if np.isnan(tfas_predict[j]):
                tfas_predict[j] = np.mean(Y)

        tfas_predict_mat_2 = tfas_predict
        tfas_actually_mat_2 = tfas_real
        mean_error_mat_2 = np.abs(tfas_real - tfas_predict)
        return tfas_predict_mat_2, tfas_actually_mat_2, mean_error_mat_2

    def cellboxplotchange(self, cell_data, labell, Y, legendlabel, ax):
        data = []
        grp = []

        for i in range(len(cell_data)):
            data.extend(cell_data[i])
            grp.extend(np.ones(len(cell_data[i])) * i)

        X = labell
        col = np.array([255, 0, 0, 0]) / 255

        self.multiple_boxplot(cell_data, ax, X, [legendlabel], np.expand_dims(col, axis=0).T)

    @staticmethod
    def multiple_boxplot(data, ax, xlab=None, Mlab=None, colors=None):
        if not isinstance(data, list):
            raise ValueError('Input data is not even a cell array!')
        M = len(data[0])  # Number of data for the same group
        L = len(data)  # Number of groups
        # Check optional inputs
        if colors is not None:
            if colors.shape[1] != M:
                raise ValueError('Wrong amount of colors!')
        if xlab is not None:
            if len(xlab) != L:
                raise ValueError('Wrong amount of X labels given')
        if M > 1:
            w = 0.25
        else:
            w = 0.25
        # Calculate the positions of the boxes
        if (np.sum(np.mean(np.diff(xlab)) == np.diff(xlab)) != len(np.diff(xlab))):
            if M > 1:
                YY = np.sort(np.concatenate([xlab - w / 2, xlab + w / 2]))
                positions = YY
            else:
                positions = xlab
                YY = positions
        else:
            if M > 1:
                positions = np.arange(1, M * L * w + 1 + w * L, w)
                positions = positions[:M * L]  # Remove excess positions if M*L*w+1 > M*L
                YY = xlab[0] - 1 + positions - w / 2
            else:
                positions = np.arange(1, M * L * w + 1 + w * L, w)
                positions = positions[:M * L]  # Remove excess positions if M*L*w+1 > M*L
                YY = xlab[0] - 1 + positions - w
        x = []
        group = []
        for ii in range(L):
            for jj in range(M):
                aux = np.array(data[ii][jj])
                x.extend(aux.flatten())
                group.extend(np.ones(aux.size) * (jj + ii * M + 1))
        ax.boxplot(x, positions=YY)
        labelpos = np.sum(np.reshape(positions, (M, -1)), axis=0) / M
        ax.set_xticks(labelpos)
        ax.set_xticklabels(xlab if xlab is not None else [str(i) for i in range(1, L + 1)])
        if colors is None:
            cmap = plt.get_cmap('hsv')
            colors = np.vstack([cmap(np.linspace(0, 1, M)), np.ones(M) * 0.5]).T
        color = np.tile(colors, (1, L))
        for idx, box in enumerate(ax.get_children()):
            if isinstance(box, plt.matplotlib.patches.PathPatch):
                box.set_facecolor(color[:3, idx])
                box.set_edgecolor(color[:3, idx])
                box.set_alpha(color[3, idx])
        if Mlab is not None:
            ax.legend(reversed(Mlab), fontsize=6, loc='upper right')

    @staticmethod
    def cv_bias_correction(tfas_predict_mat_2, tfas_actually_mat_2, hist_space,bin_width, mean_vec, x_hist_space,
                           median_fit_vec, x_ticks, y_ticks, save_path):
        tfas_predict_mat_2_sorted = np.sort(tfas_predict_mat_2)
        tfas_predict_mat_2_sorted_indices = np.argsort(tfas_predict_mat_2)
        x = tfas_predict_mat_2_sorted
        y = tfas_actually_mat_2[tfas_predict_mat_2_sorted_indices]
        std_hista_origin = np.full(len(hist_space) - 1, np.nan)
        mean_hista_origin = np.full(len(hist_space) - 1, np.nan)
        mean_hista_last_fig = np.full(len(hist_space) - 1, np.nan)
        std_hista = np.full(len(hist_space) - 1, np.nan)
        mean_hista = np.full(len(hist_space) - 1, np.nan)
        mean_vec_pre = np.append(mean_vec[1:], np.nan)
        cutofflength = len(mean_vec)
        occurrence_probability = np.zeros((1, cutofflength))
        cv_offset = mean_vec - x_hist_space
        cv_offset = np.nan_to_num(cv_offset)
        cv_corrected_x = np.zeros_like(x)
        fig_2 = plt.figure()
        a = fig_2.add_subplot(3, 1, 1)
        b = fig_2.add_subplot(3, 1, 2)
        c = fig_2.add_subplot(3, 1, 3)
        a.scatter(x, y, color=(0.7, 0.7, 0.7))  # TODO: set marker size to 1
        a.set_xticks(x_ticks)
        a.set_yticks(y_ticks)
        a.set_xlim(x_ticks[0], x_ticks[-1])
        a.set_ylim(y_ticks[0], y_ticks[-1])
        for i in range(1,cutofflength):
            Ind = np.where((hist_space[i-1] < x) & (x < (hist_space[i])))[0]  # TODO: Ind = []
            if 0 < len(Ind) < int(0.01 * len(x)):
                std_hista[i-1] = np.nan
                mean_hista[i-1] = np.nan
                occurrence_probability[i-1] = len(Ind)
                cv_corrected_x[i-1] = x[Ind] + cv_offset[i-1]
                new_y = y[Ind]
                a.scatter(x[Ind] + cv_offset[i-1], y[Ind], color='k', marker='s')
            else:
                yo = np.sort(y[Ind])
                aop = np.ceil(0.16 * len(yo)).astype(int)
                aop2 = np.ceil(0.84 * len(yo)).astype(int)
                z0 = -(np.exp(median_fit_vec) - yo)  # predictor error
                z1 = x_hist_space[i-1] - yo + cv_offset[i-1]  # cv_corrected predictor error
                z11 = np.sort(z1)
                mean_hista[i-1] = np.median(z1)
                std_hista[i-1] = np.sqrt(np.sum((z1 - np.median(z1)) ** 2) / len(yo))
                mean_hista_origin[i-1] = np.median(z0)
                std_hista_origin[i-1] = np.sqrt(np.sum((z1 - np.median(z1)) ** 2) / len(yo))
                occurrence_probability[i-1] = len(Ind)
                a.scatter(x[Ind] + cv_offset[i-1], y[Ind], color='k', marker='s')
                cv_corrected_x[i-1] = x[Ind] + cv_offset[i-1]
                new_y = y[Ind]
                mean_hista_last_fig = np.median(yo)
        cv_corrected_x_sorted = np.sort(cv_corrected_x)
        sorted_indices = np.argsort(cv_corrected_x)
        new_y_sorted = new_y[sorted_indices]
        slope, intercept, r_value, p_value, std_err = linregress(cv_corrected_x_sorted, new_y_sorted)
        R_x = np.array([cv_corrected_x_sorted[0], new_y_sorted[-1]])
        R_y = intercept + slope * R_x
        slope = round(slope, 2)
        b_c = round(intercept, 2)
        a.plot(R_x, R_y, color='r', linestyle='--')
        new_x = np.array(cv_corrected_x_sorted)
        min_of_all2 = np.min(cv_corrected_x_sorted)
        max_of_all2 = np.max(cv_corrected_x_sorted)
        hist_space_2 = np.linspace(bin_width * np.floor(min_of_all2 / bin_width),
                                   bin_width * np.ceil(max_of_all2 / bin_width),
                                   int(np.ceil(max_of_all2 / bin_width) - np.floor(min_of_all2 / bin_width) + 1))
        x_hist_space_2 = (hist_space_2[1:] + hist_space_2[:-1]) / 2
        cutofflength2 = len(hist_space_2) - 1
        mean_hista_last_fig2 = np.full(cutofflength2, np.nan)
        mean_histaria = np.full(cutofflength2, np.nan)
        mean_histaria_origin = np.full(cutofflength2, np.nan)
        mean_last_fig_histaria = np.full(cutofflength2, np.nan)
        cutofflength2 = len(hist_space_2) - 1
        std_histaria_origin = np.full(cutofflength2, np.nan)
        std_histaria_origin2 = np.full(cutofflength2, np.nan)
        std_histaria = np.full(cutofflength2, np.nan)
        std_histaria2 = np.full(cutofflength2, np.nan)
        occurrence_probability_r = np.full(cutofflength2, np.nan)
        mega_kde = []
        for i in range(1,cutofflength2):
            Ind = np.where((hist_space_2[i-1] < x) & (x < (hist_space_2[i])))[0]
            if len(Ind) < 5:
                occurrence_probability_r[i-1] = len(Ind)
                ok_comp = 1
            else:
                yo = np.sort(new_y_sorted[Ind])
                aop = int(np.ceil(0.16 * len(yo)))
                aop2 = int(np.ceil(0.84 * len(yo)))
                mean_hista_last_fig2[i-1] = np.median(yo)
                z0 = -(median_fit_vec - yo)
                z1 = -((x_hist_space_2[i-1]) - yo)
                mean_histaria[i-1] = np.median(z1)
                mean_histaria_origin[i-1] = np.median(z0)
                std_histaria_origin[i-1] = np.sqrt(np.sum((z0 - np.median(z0)) ** 2) / len(yo))
                std_histaria[i-1] = np.sqrt(np.sum((z1 - np.median(z1)) ** 2) / len(yo))
                occurrence_probability_r[i-1] = len(Ind)
                mega_kde.append([z1, z0, yo, i-1])

        sorted_indices = np.argsort(x_hist_space_2)
        x_hista = x_hist_space_2[sorted_indices]
        mean_hista = mean_hista[sorted_indices]
        std_hista = std_hista[sorted_indices]
        std_hista2 = std_hista[sorted_indices]
        mean_hista_origin = mean_hista_origin[sorted_indices]
        std_hista_origin = std_hista_origin[sorted_indices]
        std_hista_origin2 = std_hista_origin[sorted_indices]
        occurrence_probability = occurrence_probability[sorted_indices]
        mean_hista_last_fig = mean_hista_last_fig[sorted_indices]
        occurrence_probability = occurrence_probability / np.sum(occurrence_probability)
        occurrence_probability_r = occurrence_probability_r / np.sum(occurrence_probability_r)
        std_histaria2 = std_histaria
        I2 = (~np.isnan(x_hist_space_2)) & (~np.isnan(mean_hista_last_fig2))
        SLM_tools.cellboxplotchange(mega_kde, x_hist_space_2[I2], x_hist_space_2[I2], r'$\hat{Y}_{BC}$', ax=b)
        b.plot(x_hist_space_2[I2], x_hist_space_2[I2], linestyle='--', color='k')
        b.set_xticks(x_ticks)
        b.set_yticks(y_ticks)
        b.set_xlim(x_ticks[0], x_ticks[-1])
        b.set_ylim(y_ticks[0], y_ticks[-1])
        labels = [str(label) for label in x_ticks[1:]]
        b.set_xticklabels(labels, rotation=45)
        X = x_hist_space_2[I2]
        col = np.array([[255, 0, 0, 0], [100, 100, 100, 0]]) / 255
        A = np.column_stack((mega_kde[:, 1], mega_kde[:, 0]))
        B = A.copy()
        for i1 in range(A.shape[0]):
            for i2 in range(A.shape[1]):
                med = np.median(A[i1, i2])
                if med < 0:
                    B[i1, i2] = A[i1, i2] - 2 * med
        SLM_tools.multiple_boxplot(B, c, X, [r'$\Delta\hat{Y}_{BC}$', r'$\Delta\hat{Y}_{M}$'], col.T)
        c.set_xlim(x_ticks[0], x_ticks[-1])
        c.set_xticks(x_ticks[1:])
        c.set_lables(x_ticks[1:])
        c.set_xticklabels(labels, rotation=45)
        fig_2.savefig(os.path.join(save_path, 'fig_2.png'))
