import Rbeast as rb
import numpy as np
import scipy.stats
from scipy import interpolate, stats, signal, ndimage
import scipy.io as sio
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle as pkl
import os


class SLM_tools:
    # def __init__(self): # TODO: think if and how should we save data along the process
    #     self.data = None
    #     self.input_directory_path = None
    #     self.output_directory_path = None
    #     self.files = []
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
        # checked
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
    def downsample(data: np.array, downsampling_factor: int, time_vec: np.array = None):
        # checked
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

    @staticmethod
    def beast(data: np.array):
        # checked
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
        # checked, original give_vecs.m
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
            abc = np.polyfit(range(cp_int[i], cp_int[i + 1]), mean_trend[cp_int[i]:cp_int[i + 1]], 1)
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
        # if not np.any(sa_vec == 1):  # TODO: uncomment after testing
        #     return []
        # else:
        return np.column_stack(
            (mu, std, skew, np.cumsum(times_vec), trend_vec, sa_vec, cumulated_time_vec))

        # TODO: all returns of segment_data of all files should v_stack before post processing and model building

    @staticmethod
    def post_beast_processing(segment_data_aggregated_output: np.array):
        # not checked
        c_reduced = []
        for i in range(segment_data_aggregated_output.shape[0]):
            x = segment_data_aggregated_output[i][0]  # mean_vec
            y = segment_data_aggregated_output[i][1]  # std_vec
            z = segment_data_aggregated_output[i][4]  # trend
            w = segment_data_aggregated_output[i][2]  # skew
            v = segment_data_aggregated_output[i][3]  # total_trajectory_time
            c = segment_data_aggregated_output[i][5]  # time_to_self_assembly
            ending_theme = np.where(c == 0)[0][0] if np.any(c == 0) else len(c)
            x = x[:ending_theme]
            y = y[:ending_theme]
            z = z[:ending_theme]
            c = c[:ending_theme]
            w = w[:ending_theme]
            v = v[:ending_theme]
            d_reduced = np.vstack((x, y, z, w, v, c)).T
            c_reduced.append(d_reduced)
        c_reduced = np.vstack(tuple(c_reduced))
        return c_reduced

    @staticmethod
    def pca(c_reduced: np.array, n_components: int = 3):
        # not checked
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
        # not checked
        a_reduced = []
        if n_components == 3:
            for i in range(score.shape[0]):
                x = score[i, 0]  # mean_vec
                y = score[i, 1]  # std_vec
                z = score[i, 2]  # trend
                c = c_reduced[i, 6]  # TODO: should be 3?
                b_reduced = list(np.array([x, y, z, c]).T)
                a_reduced.append(b_reduced)
            # a_reduced = np.array(a_reduced)
            return a_reduced
        else:  # n_components == 5
            for i in range(score.shape[0]):
                x = score[i, 0]  # mean_vec
                y = score[i, 1]  # std_vec
                z = score[i, 4]  # trend
                w = score[i, 2]  # skew
                v = score[i, 3]  # total_trajectory_time
                c = c_reduced[i, 6]  # TODO: what's in here?
                b_reduced = list(np.array([x, y, z, w, v, c]).T)
                a_reduced.append(b_reduced)
            # a_reduced = np.array(a_reduced)
            return a_reduced

    @staticmethod
    def trajectory_plot_vecs(mean_vec, std_vec, trend, time_to_self_assembly, save_path: str, bottom=0, top=3000):
        # not checked
        sz = 40
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mean_vec.T, std_vec.T, trend.T, s=sz, c=time_to_self_assembly, cmap='jet', marker='o')
        ax.set_xlim(bottom, top)
        ax.set_ylim(bottom, top)
        ax.set_zlim(bottom, top)
        d = (np.vstack((mean_vec[1:], std_vec[1:], trend[1:])) - np.vstack(
            (mean_vec[:-1], std_vec[:-1], trend[:-1]))) / 2
        color_triplet = np.random.rand(1, 3)
        ax.quiver(mean_vec[:-1], std_vec[:-1], trend[:-1], d[0], d[1], d[2], color=color_triplet, length=0.1,
                  normalize=True)
        ax.scatter(mean_vec[0], std_vec[0], trend[0], s=1 * sz, c=time_to_self_assembly[0], marker='^',
                   label='Start Point')
        ax.scatter(mean_vec[-1], std_vec[-1], trend[-1], s=1 * sz, c=time_to_self_assembly[-1], marker='d',
                   label='Finish Point')
        ax.legend(['Points', 'Arrows', 'Trajectory', 'Start Point', 'Finish Point'])
        ax.set_xlabel('Mean')
        ax.set_ylabel('Standard Deviation')
        ax.set_zlabel('Trend')
        ax.set_title('Object Position')
        ax.grid(True)
        plt.show()  # TODO: replace with plt.save()?
        # plt.savefig(save_path, bbox_inches='tight')

    @staticmethod
    def replace_nan_with_rounded_mean(array):
        mean_val = np.nanmean(array)
        array[np.isnan(array)] = np.round(mean_val)
        return array

    @staticmethod
    def model_training_with_cv(a_reduced: np.array, n_components: int = 3, cv_num: int = 3):
        # from LogTfas....
        np.random.seed(42)
        idn = np.where(a_reduced[:, n_components + 1] != 0)
        mapx = a_reduced[idn, :]
        mapx = np.hstack((mapx[:, 0:2], np.log(mapx[:, 3])))
        median_fit_vec = np.median(mapx[:, 3], axis=0)  # line 54 matlab
        min = np.min(mapx[:, 3], axis=0)
        max = np.max(mapx[:, 3], axis=0)
        xw = mapx[:, 3]
        points = np.linspace(min, max, 1000)
        f_x = scipy.stats.gaussian_kde(points)  # line 60 matlab
        p_x = f_x[xw]
        yo = np.sort(mapx[:, 3])
        aop = int(np.ceil((0.16 * len(yo)))) - 1
        std_fit = np.median(mapx[:, 3], axis=0) - yo[aop]
        x = np.copy(mapx)
        random_x = x[np.random.permutation(x.shape[0]), :]  # line 89 matlab
        train_index = int(np.floor(0.6 * random_x.shape[0]))
        validation_index = int(np.floor(0.8 * random_x.shape[0]))
        size_validation_set = len(range(train_index, validation_index))
        tfas_predict_mat = np.zeros((cv_num, size_validation_set))
        tfas_actually_mat = np.zeros((cv_num, size_validation_set))
        mean_error_mat = np.zeros((cv_num, size_validation_set))
        training_set = random_x[:train_index - 1, :]
        validation_set = random_x[train_index:validation_index, :]
        for i in range(cv_num):
            min_1 = np.min(random_x[:, 0])
            max_1 = np.max(random_x[:, 0])
            min_2 = np.min(random_x[:, 1])
            max_2 = np.max(random_x[:, 1])
            min_3 = np.min(random_x[:, 2])
            max_3 = np.max(random_x[:, 2])
            d1 = np.linspace(np.floor(min_1), np.ceil(max_1), 150)
            d2 = np.linspace(np.floor(min_2), np.ceil(max_2), 150)
            d3 = np.linspace(np.floor(min_3), np.ceil(max_3), 150)
            if n_components == 5:
                min_4 = np.min(random_x[:, 3])
                max_4 = np.max(random_x[:, 3])
                min_5 = np.min(random_x[:, 4])
                max_5 = np.max(random_x[:, 4])
                d4 = np.linspace(np.floor(min_4), np.ceil(max_4), 10)
                d5 = np.linspace(np.floor(min_5), np.ceil(max_5), 10)
                x0, y0, z0, w0, v0 = np.meshgrid(d1, d2, d3, d4, d5, indexing='ij')
                X = training_set[:, :4]
                Y = training_set[:, 5]
                XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel(), w0.ravel(), v0.ravel()))
            else:
                x0, y0, z0 = np.meshgrid(d1, d2, d3, indexing='ij')
                X = training_set[:, :2]
                Y = training_set[:, 3]
                XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel()))

            YI = scipy.interpolate.griddata(X, Y, XI, method='linear')
            YI.reshape(x0.shape)
            intergal_dist = 2  # TODO: ask michael!
            k = np.ones((intergal_dist, intergal_dist)) / (intergal_dist * intergal_dist - 1)
            k[intergal_dist // 2, intergal_dist // 2] = 0
            averageIntensities = scipy.ndimage.convolve(YI, k, mode='constant', cval=0.0)
            YI = averageIntensities
            Ix = pd.cut(validation_set[:, 0], bins=d1, labels=False, include_lowest=True)
            Iy = pd.cut(validation_set[:, 1], bins=d2, labels=False, include_lowest=True)
            Iz = pd.cut(validation_set[:, 2], bins=d3, labels=False, include_lowest=True)
            Ix = SLM_tools.replace_nan_with_rounded_mean(Ix.to_numpy(dtype=float))
            Iy = SLM_tools.replace_nan_with_rounded_mean(Iy.to_numpy(dtype=float))
            Iz = SLM_tools.replace_nan_with_rounded_mean(Iz.to_numpy(dtype=float))
            if n_components == 5:
                Iw = pd.cut(validation_set[:, 3], bins=d4, labels=False, include_lowest=True)
                Iv = pd.cut(validation_set[:, 4], bins=d5, labels=False, include_lowest=True)
                Iw = SLM_tools.replace_nan_with_rounded_mean(Iw.to_numpy(dtype=float))
                Iv = SLM_tools.replace_nan_with_rounded_mean(Iv.to_numpy(dtype=float))
            tfas_real = np.zeros((validation_set.shape[0], 1))
            tfas_predict = np.zeros((validation_set.shape[0], 1))
            for j in range(validation_set.shape[0]):
                tfas_real[j] = validation_set[j, n_components + 1]
                if n_components == 3:
                    tfas_predict[j] = YI[Ix[j], Iy[j], Iz[j]]
                else:
                    tfas_predict[j] = YI[Ix[j], Iy[j], Iz[j], Iw[j], Iv[j]]
                if np.isnan(tfas_predict[j]):
                    tfas_predict[j] = np.mean(Y)

            tfas_predict_mat[i, :] = tfas_predict
            tfas_actually_mat[i, :] = tfas_real
            mean_error_mat[i, :] = np.abs(tfas_real - tfas_predict)
        return YI, tfas_predict_mat, tfas_actually_mat, mean_error_mat, train_index, random_x, validation_index, median_fit_vec

    @staticmethod
    def save_model(YI, save_path, save_as_text=False, save_as_m=True):
        if save_as_m:
            sio.savemat(f"{os.path.join(save_path, 'YI.txt')}",{"model": YI})
        elif save_as_text:
            np.savetxt(f"{os.path.join(save_path, 'YI.txt')}", YI)
        else:
            np.save(f"{os.path.join(save_path, 'YI.npy')}", YI)

    @staticmethod
    def model_eval(tfas_predict_mat, tfas_actually_mat, train_index, save_path, cv_num: int = 3):
        bin_width = 0.5  # TODO: where are these values from?
        smooth_win = 0  # TODO: where are these values from?
        x_ticks = np.arange(3.5, 7.5 + 0.5, 0.5)  # TODO: where are these values from?
        y_ticks = np.arange(3, 9 + 2, 2)  # TODO: where are these values from?
        color_map = plt.cm.jet(np.linspace(0, 1, cv_num))
        red = len(range(1, train_index + 1))
        min_of_all = np.min(tfas_predict_mat)
        max_of_all = np.max(tfas_predict_mat)
        hist_space = np.linspace(bin_width * np.floor(min_of_all / bin_width),
                                 bin_width * np.ceil(max_of_all / bin_width),
                                 int((np.ceil(max_of_all / bin_width) - np.floor(min_of_all / bin_width)) + 1))
        mean = np.zeros((cv_num, len(hist_space) - 1))
        std = np.zeros((cv_num, len(hist_space) - 1))
        plt.figure()
        for i in range(cv_num):
            x = tfas_predict_mat[i, :]
            y = tfas_actually_mat[i, :]
            sorted_indices = np.argsort(x)
            x = x[sorted_indices]
            y = y[sorted_indices]
            plt.subplot(4, 1, 1)
            plt.scatter(x, y, edgecolors=color_map[i], facecolors=color_map[i])
            plt.gca().set_xticklabels([])
            plt.yticks(y_ticks)
            plt.ylabel(r'$\hat{Y}$', fontsize=24)
            plt.xlim([x_ticks[0], x_ticks[-1]])
            plt.ylim([y_ticks[0], y_ticks[-1]])
            plt.gca().tick_params(axis='both', which='both', length=0)
            plt.box(False)
            std_hista = np.zeros(len(hist_space) - 1)
            mean_hista = np.zeros(len(hist_space) - 1)
            for j in range(len(hist_space) - 1):
                indices = np.where((hist_space[j] - smooth_win < x) and (x < (hist_space[j + 1]) + smooth_win))[0]
                if len(indices) < 5:
                    std_hista[j] = np.nan
                    mean_hista[j] = np.nan
                else:
                    yo = np.sort(y[indices])
                    mean_hista[j] = np.median(y[indices])
                    aop = int(np.ceil(0.16 * len(yo)))
                    std_hista[j] = mean_hista[j] - yo[aop]

            x_hist_space = (hist_space[1:] + hist_space[:-1]) / 2
            i_9 = np.where(~np.isnan(mean_hista) & (x_hist_space > 0))[0][0] if np.any(
                ~np.isnan(mean_hista) & (x_hist_space > 0)) else len(mean_hista)
            i_2 = np.where(np.isnan(mean_hista) & (x_hist_space > 0))[0][0] if np.any(
                np.isnan(mean_hista) & (x_hist_space > 0)) else len(mean_hista)
            i_3 = np.where(np.isnan(std_hista) & (x_hist_space > 0))[0][0] if np.any(
                np.isnan(std_hista) & (x_hist_space > 0)) else len(mean_hista)
            i_4 = min(i_2, i_3)

            if i_9 < i_4:
                i_4 = min(i_2, i_3)
            else:
                i_2 = np.where(np.isnan(mean_hista) & (x_hist_space > 0), i_9)[0][-1] if np.any(
                    np.isnan(mean_hista) & (x_hist_space > 0)) else len(mean_hista)
                i_3 = np.where(np.isnan(std_hista) & (x_hist_space > 0), i_9)[0][-1] if np.any(
                    np.isnan(std_hista) & (x_hist_space > 0)) else len(mean_hista)
                i_4 = min(i_2, i_3)
            if i_4 is None:
                i_4 = len(mean_hista)

            plt.subplot(4, 1, 2)
            plt.scatter(x_hist_space[:i_4], mean_hista[:i_4], edgecolors=color_map[i], facecolors=color_map[i],
                        marker='s')
            plt.hold(True)
            # Store results
            mean[i, :len(mean_hista)] = mean_hista
            std[i, :len(std_hista)] = std_hista
            i_5 = np.argmin(np.abs(x - x_hist_space[i_4]))
            # Update first subplot
            plt.subplot(4, 1, 1)
            plt.scatter(x[:i_5], y[:i_5], edgecolors=color_map[i], facecolors=color_map[i])
            plt.hold(True)
            if i == 0:
                plt.ylabel(r'${Y}_{CV}$', fontsize=24, labelpad=20)
                plt.gca().set_xticklabels([])
                plt.yticks(y_ticks)
                plt.xlim([x_ticks[0], x_ticks[-1]])
                plt.ylim([y_ticks[0], y_ticks[-1]])
                plt.gca().tick_params(axis='both', which='both', length=0)
                plt.box(False)
        plt.show()  # TODO: replace with plt.save()?
        # plt.savefig(save_path, bbox_inches='tight')
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
                std_vec[row] = np.std(mean_row)
            else:
                mean_vec[row] = np.nan
                std_vec[row] = np.nan
        plt.subplot(4, 1, 2)
        plt.plot(x_hist_space[:i_4], x_hist_space[:i_4], 'k--')
        plt.errorbar(x_hist_space, mean_vec, yerr=std_vec, fmt='k', ecolor='k', elinewidth=1, capsize=2)
        plt.gca().set_xticklabels([])
        plt.yticks(y_ticks)
        plt.xlim([x_ticks[0], x_ticks[-1]])
        plt.ylim([y_ticks[0], y_ticks[-1]])
        plt.box(False)
        plt.gca().tick_params(axis='both', which='both', length=0)
        plt.gca().set_fontsize(24)
        plt.show()  # TODO: replace with plt.save()?
        # plt.savefig(save_path, bbox_inches='tight')
        return mean_vec, y_ticks, x_ticks, hist_space, x_hist_space

    @staticmethod
    def train_again_on_validation_and_test(random_x, validation_index, n_components=3):
        np.random.seed(42)
        random_x = random_x[np.random.permutation(random_x.shape[0]), :]
        test_set = random_x[validation_index:, :]
        training_set = random_x[:validation_index - 1, :]
        min_1 = np.min(random_x[:, 0])
        max_1 = np.max(random_x[:, 0])
        min_2 = np.min(random_x[:, 1])
        max_2 = np.max(random_x[:, 1])
        min_3 = np.min(random_x[:, 2])
        max_3 = np.max(random_x[:, 2])
        d1 = np.linspace(np.floor(min_1), np.ceil(max_1), 150)
        d2 = np.linspace(np.floor(min_2), np.ceil(max_2), 150)
        d3 = np.linspace(np.floor(min_3), np.ceil(max_3), 150)
        if n_components == 5:
            min_4 = np.min(random_x[:, 3])
            max_4 = np.max(random_x[:, 3])
            min_5 = np.min(random_x[:, 4])
            max_5 = np.max(random_x[:, 4])
            d4 = np.linspace(np.floor(min_4), np.ceil(max_4), 10)
            d5 = np.linspace(np.floor(min_5), np.ceil(max_5), 10)
            x0, y0, z0, w0, v0 = np.meshgrid(d1, d2, d3, d4, d5, indexing='ij')
            X = training_set[:, :4]
            Y = training_set[:, 5]
            XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel(), w0.ravel(), v0.ravel()))
        # Create linearly spaced vectors for each coordinate
        else:
            x0, y0, z0 = np.meshgrid(d1, d2, d3, indexing='ij')
            X = training_set[:, :2]
            Y = training_set[:, 3]
            XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel()))

        YI = scipy.interpolate.griddata(X, Y, XI, method='linear')
        YI.reshape(x0.shape)
        intergal_dist = 2  # TODO: ask michael!
        k = np.ones((intergal_dist, intergal_dist)) / (intergal_dist * intergal_dist - 1)
        k[intergal_dist // 2, intergal_dist // 2] = 0
        averageIntensities = scipy.ndimage.convolve(YI, k, mode='constant', cval=0.0)
        YI = averageIntensities
        Ix = pd.cut(test_set[:, 0], bins=d1, labels=False, include_lowest=True)
        Iy = pd.cut(test_set[:, 1], bins=d2, labels=False, include_lowest=True)
        Iz = pd.cut(test_set[:, 2], bins=d3, labels=False, include_lowest=True)
        Ix = SLM_tools.replace_nan_with_rounded_mean(Ix.to_numpy(dtype=float))
        Iy = SLM_tools.replace_nan_with_rounded_mean(Iy.to_numpy(dtype=float))
        Iz = SLM_tools.replace_nan_with_rounded_mean(Iz.to_numpy(dtype=float))
        if n_components == 5:
            Iw = pd.cut(test_set[:, 3], bins=d4, labels=False, include_lowest=True)
            Iv = pd.cut(test_set[:, 4], bins=d5, labels=False, include_lowest=True)
            Iw = SLM_tools.replace_nan_with_rounded_mean(Iw.to_numpy(dtype=float))
            Iv = SLM_tools.replace_nan_with_rounded_mean(Iv.to_numpy(dtype=float))
        tfas_real = np.zeros((test_set.shape[0], 1))
        tfas_predict = np.zeros((test_set.shape[0], 1))
        for j in range(test_set.shape[0]):
            tfas_real[j] = test_set[j, n_components + 1]
            if n_components == 3:
                tfas_predict[j] = YI[Ix[j], Iy[j], Iz[j]]
            else:
                tfas_predict[j] = YI[Ix[j], Iy[j], Iz[j], Iw[j], Iv[j]]
            if np.isnan(tfas_predict[j]):
                tfas_predict[j] = np.mean(Y)

        tfas_predict_mat_2 = tfas_predict
        tfas_actually_mat_2 = tfas_real
        mean_error_mat_2 = np.abs(tfas_real - tfas_predict)
        return tfas_predict_mat_2, tfas_actually_mat_2, mean_error_mat_2

    @staticmethod
    def after_training_2(tfas_predict_mat_2, tfas_actually_mat_2, y_ticks, x_ticks, hist_space, mean_vec, x_hist_space,
                         median_fit_vec, save_path):
        smooth_win = 0
        tfas_predict_mat_2_sorted = np.sort(tfas_predict_mat_2)
        tfas_predict_mat_2_sorted_indices = np.argsort(tfas_predict_mat_2_sorted)
        x = tfas_predict_mat_2_sorted
        y_new = tfas_actually_mat_2[tfas_predict_mat_2_sorted_indices]
        fig, ax = plt.subplots(4, 1, figsize=(10, 8))
        nn3 = ax[2]
        nn3.scatter(x, edgecolor=[.7, .7, .7], facecolor=[.7, .7, .7])
        nn3.set_xticklabels([])  # Remove x-axis tick labels
        nn3.set_yticks(y_ticks)
        nn3.set_ylabel(r'${Y_{test}}$', fontsize=14)
        nn3.set_xlim([x_ticks[0], x_ticks[-1]])
        nn3.set_ylim([x_ticks[0], x_ticks[-1]])
        nn3.tick_params(axis='both', which='major', labelsize=24)
        nn3.spines['top'].set_visible(False)
        nn3.spines['right'].set_visible(False)
        nn3.spines['bottom'].set_visible(False)
        nn3.spines['left'].set_visible(False)
        nn3.xaxis.set_tick_params(length=0)
        plt.show()  # TODO: replace with plt.save()?
        # plt.savefig(save_path, bbox_inches='tight')
        std_hista_origin = np.full(len(hist_space) - 1, np.nan)
        mean_hista_origin = np.full(len(hist_space) - 1, np.nan)
        mean_hista_last_fig = np.full(len(hist_space) - 1, np.nan)
        std_hista = np.full(len(hist_space) - 1, np.nan)
        mean_hista = np.full(len(hist_space) - 1, np.nan)
        mean_vec_pre = np.append(mean_vec[1:], np.nan)
        cutofflength = len(mean_vec)
        occurrence_probability = np.zeros((1, cutofflength))
        CV_offset = mean_vec - x_hist_space
        CV_offset = np.nan_to_num(CV_offset)
        for i in range(cutofflength):
            Ind = np.where((hist_space[i] - smooth_win < x) and (x < hist_space[i + 1] + smooth_win))[0]
            occurrence_probability[0, i] = len(Ind)
            if len(Ind) < max(1, int(0.01 * len(x))):
                std_hista[i] = np.nan
                mean_hista[i] = np.nan
            else:
                yo = np.sort(y_new[Ind])
                yo_logged = yo  # TODO: ask michael if should be np.log(yo) Line 704 LogTfas
                aop = np.ceil(0.16 * len(yo)).astype(int)
                aop2 = np.ceil(0.84 * len(yo)).astype(int)
                z0 = median_fit_vec - yo
                z1 = x_hist_space[i] - yo_logged + CV_offset[i]
                mean_hista[i] = np.median(z1)
                std_hista[i] = np.sqrt(np.sum((z1 - np.median(z1)) ** 2) / len(yo_logged))
                mean_hista_origin[i] = np.median(z0)
                mean_hista_last_fig[i] = np.median(yo)
                plt.subplot(4, 1, 3)
                plt.scatter(x[Ind] + CV_offset[i], y_new[Ind], marker='s', edgecolor='k', facecolor='k')
                plt.hold(True)
        plt.show()  # TODO: replace with plt.save()?
        # plt.savefig(save_path, bbox_inches='tight')
        # Line 732 Matlab
