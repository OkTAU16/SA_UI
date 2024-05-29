import Rbeast as rb
import numpy as np
from scipy import interpolate
from scipy import signal
import scipy.io as sio
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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
            temp1 = np.where(sa_vec == 1)[
                0]  # if temp1 is empty none of the data is saved, ask michael
            if i not in temp1:
                try:
                    temp2 = np.where(temp1 > i)[0][0]
                    omega = np.cumsum(times_vec[i:temp1[temp2]])
                    cumulated_time_vec[i] = omega[-1]
                except Exception:
                    cumulated_time_vec[i] = np.nan
            else:
                cumulated_time_vec[i] = 0
        # if not np.any(sa_vec == 1):  # current testing file doesn't have assembly return after tests, should be above the second loop?
        #     return []
        # else:
        return np.column_stack(
            (mu, std, skew, np.cumsum(times_vec), trend_vec, sa_vec, cumulated_time_vec))  # TODO: look with michael, order

        # TODO: Idan, look at the return order of give_vecs.m compered to it's call in Gather_Tfas_Single_Drive.m
        #  line 186, and cc decleration in line 188, I suspect the order is wrong.

        # TODO: all returns of segment_data of all files should v_stack before post processing and model building

    @staticmethod
    def post_beast_processing(segment_data_aggregated_output: np.array):
        # not checked
        c_reduced = []
        for i in range(len(segment_data_aggregated_output)):
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
    def pca(data: np.array, n_components: int):
        # not checked
        data = data[:, :n_components]
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
                c = c_reduced[i, 6]  # TODO: what's in here?
                b_reduced = list(np.array([x, y, z, c]).T)  # TODO: check stacking
                a_reduced.append(b_reduced)
            return np.array(a_reduced)
        else:  # n_components == 5
            for i in range(score.shape[0]):
                x = score[i, 0]  # mean_vec
                y = score[i, 1]  # std_vec
                z = score[i, 4]  # trend
                w = score[i, 2]  # skew
                v = score[i, 3]  # total_trajectory_time
                c = c_reduced[i, 6]  # TODO: what's in here?
                b_reduced = list(np.array([x, y, z, w, v, c]).T)  # TODO: check stacking
                a_reduced.append(b_reduced)
            return np.array(a_reduced)

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
        d = (np.vstack((mean_vec[1:], std_vec[1:], trend[1:])) - np.vstack((mean_vec[:-1], std_vec[:-1], trend[:-1]))) / 2
        color_triplet = np.random.rand(1, 3)
        ax.quiver(mean_vec[:-1], std_vec[:-1], trend[:-1], d[0], d[1], d[2], color=color_triplet, length=0.1, normalize=True)
        ax.scatter(mean_vec[0], std_vec[0], trend[0], s=1 * sz, c=time_to_self_assembly[0], marker='^', label='Start Point')
        ax.scatter(mean_vec[-1], std_vec[-1], trend[-1], s=1 * sz, c=time_to_self_assembly[-1], marker='d', label='Finish Point')
        ax.legend(['Points', 'Arrows', 'Trajectory', 'Start Point', 'Finish Point'])
        ax.set_xlabel('Mean')
        ax.set_ylabel('Standard Deviation')
        ax.set_zlabel('Trend')
        ax.set_title('Object Position')
        ax.grid(True)
        plt.show()  # TODO: check with UI
        plt.savefig(save_path, bbox_inches='tight', )
        return fig
