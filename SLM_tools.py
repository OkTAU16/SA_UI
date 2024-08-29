import time
from datetime import datetime
import logging
import Rbeast as rb
import numpy as np
import scipy.stats
from scipy import interpolate, signal
from scipy.stats import linregress
import scipy.io as sio
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec


class SLM_tools:
    @staticmethod
    def load_data(data_path: str, data_variable_name, target_num, data_type='csv', time_vec_exists=False):
        """
        Load data from a specified file path and return the relevant vectors.

        Parameters:
        data_path (str): The path to the data file.
        data_variable_name (str): The variable name in the .mat file (only used if data_type is '.mat').
        target_num (int): The number of target columns to load.
        data_type (str): The type of the data file ('csv', 'excel', or '.mat'). Default is 'csv'.
        time_vec_exists (bool): Flag indicating if a time vector exists in the data. Default is False.

        Returns:
        tuple: Depending on the data_type and time_vec_exists, returns:
            - (values_vec, time_vec, distance) if time_vec_exists is True.
            - (values_vec, distance) if time_vec_exists is False.

        Raises:
        Exception: If the file type is not supported or any other error occurs during data loading.
        """
        try:
            if data_type == 'csv':
                # Load data from a CSV file
                if time_vec_exists:
                    # Load time vector, values vector, and distance matrix
                    time_vec = pd.read_csv(data_path, usecols=[0]).to_numpy()
                    values_vec = pd.read_csv(data_path, usecols=[1], header=None).to_numpy()
                    distance_columns = list(range(2, 2 + target_num))
                    distance = pd.read_csv(data_path, usecols=distance_columns, header=None).to_numpy()
                    return values_vec, time_vec, distance
                else:
                    # Load values vector and distance matrix
                    values_vec = pd.read_csv(data_path, usecols=[0], header=None).to_numpy()
                    distance_columns = list(range(1, 1 + target_num))
                    distance = pd.read_csv(data_path, usecols=distance_columns, header=None).to_numpy()
                    return values_vec, distance
            elif data_type == 'xlsx':
                # Load data from an Excel file
                if time_vec_exists:
                    # Load time vector, values vector, and distance matrix
                    time_vec = pd.read_excel(data_path, usecols=[0], header=None).to_numpy()
                    values_vec = pd.read_excel(data_path, usecols=[1], header=None).to_numpy()
                    distance_columns = list(range(2, 2 + target_num))
                    distance = pd.read_excel(data_path, usecols=distance_columns, header=None).to_numpy()
                    return values_vec, time_vec, distance
                else:
                    # Load values vector and distance matrix
                    values_vec = pd.read_excel(data_path, usecols=[0], header=None).to_numpy()
                    distance_columns = list(range(1, 1 + target_num))
                    distance = pd.read_excel(data_path, usecols=distance_columns, header=None).to_numpy()
                    return values_vec, distance
            elif data_type == '.mat':
                # Load data from a .mat file
                dict_data = sio.loadmat(data_path)
                if time_vec_exists:
                    # Load time vector, values vector, and distance matrix
                    time_vec = dict_data[data_variable_name][:, 0]
                    values_vec = dict_data[data_variable_name][:, 1]
                    distance = dict_data[data_variable_name][:, 2:2 + target_num]
                    return values_vec, time_vec, distance
                else:
                    # Load values vector and distance matrix
                    values_vec = dict_data[data_variable_name][:, 0]
                    distance = dict_data[data_variable_name][:, 1:1 + target_num]
                    return values_vec, distance
            else:
                # Raise an exception if the file type is not supported
                raise Exception("file type is not supported")
        except Exception as e:
            # Raise any exception that occurs during data loading
            raise e

    # @staticmethod
    # def load_data(data_path: str, data_variable_name, target_num, data_type='csv', time_vec_exists=False):
    #     try:
    #         if data_type == 'csv':
    #             if time_vec_exists:
    #                 time_vec = pd.read_csv(data_path, usecols=[0]).to_numpy()
    #                 values_vec = pd.read_csv(data_path, usecols=[1], header=None).to_numpy()
    #                 distance_columns = list(range(2, 2 + target_num))
    #                 distance = pd.read_csv(data_path, usecols=distance_columns, header=None).to_numpy()
    #                 return values_vec, time_vec, distance
    #             else:
    #                 values_vec = pd.read_csv(data_path, usecols=[0], header=None).to_numpy()
    #                 distance_columns = list(range(1, 1 + target_num))
    #                 distance = pd.read_csv(data_path, usecols=distance_columns, header=None).to_numpy()
    #                 return values_vec, distance
    #         elif data_type == 'excel':
    #             if time_vec_exists:
    #                 time_vec = pd.read_excel(data_path, usecols=[0], header=None).to_numpy()
    #                 values_vec = pd.read_excel(data_path, usecols=[1], header=None).to_numpy()
    #                 distance_columns = list(range(2, 2 + target_num))
    #                 distance = pd.read_excel(data_path, usecols=distance_columns, header=None).to_numpy()
    #                 return values_vec, time_vec, distance
    #             else:
    #                 values_vec = pd.read_excel(data_path, usecols=[0], header=None).to_numpy()
    #                 distance_columns = list(range(1, 1 + target_num))
    #                 distance = pd.read_excel(data_path, usecols=distance_columns, header=None).to_numpy()
    #                 return values_vec, distance
    #         elif data_type == '.mat':
    #             dict_data = sio.loadmat(data_path)
    #             if time_vec_exists:
    #                 time_vec = dict_data[data_variable_name][:, 0]
    #                 values_vec = dict_data[data_variable_name][:, 1]
    #                 distance = dict_data[data_variable_name][:, 2:2 + target_num]
    #                 return values_vec, time_vec, distance
    #             else:
    #                 values_vec = dict_data[data_variable_name][:, 0]
    #                 distance = dict_data[data_variable_name][:, 1:1 + target_num]
    #                 return values_vec, distance
    #         else:
    #             raise Exception("file type is not supported")
    #     except Exception as e:
    #         raise e

    @staticmethod
    def interpolate_data_over_regular_time(data: np.array, time: np.array):
        # checked
        """interpolate data over a regularly spaced time vector
            inputs:
                data (np.array): time series data
                time (np.array) : time vector
            outputs:
                time_vec (np.array): regularly spaced time vector
                data_new (np.array): interpolated data over the regularly spaced time vector"""
        try:
            t = np.linspace(0, time[-1], len(data))
            data_new = np.interp(t, time, data)
            return data_new, t
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
            downsampled_data = data[time_vec]
            downsampled_distance = distance[time_vec, :]
            downsampled_data = np.reshape(downsampled_data, (len(data), 1))
            return downsampled_data, downsampled_distance, time_vec
        except Exception as e:
            raise e

    @staticmethod
    def beast(data: np.array):
        """call the BEAST algorithm and extract the change points
        input:
            data (np.array): time series data for analysis
        outputs:
            change_points (np.array): sorted np.array of change points index
            mean_trend (np.array): mean trend from BEAST"""

        try:
            # Call the BEAST algorithm with specified parameters
            np.random.seed(42)
            o = rb.beast(data, 0, tseg_minlength=0.01 * data.shape[0], season="none",
                         print_options=False, print_progress=False, tcp_minmax=[0, 10000], sorder_minmax=[1, 5],
                         quiet=True, print_param=False, print_warning=False)

            # Extract the mean trend from the BEAST output
            mean_trend = o.trend.Y

            # Extract and sort the change points
            cp = np.sort(o.trend.cp[0:int(o.trend.ncp_median)])

            # Remove NaN values from the change points
            cp = cp[~np.isnan(cp)]

            # Insert a starting change point at index 0
            cp = np.insert(cp, 0, 0)

            # Return the BEAST output object, change points, and mean trend
            return o, cp, mean_trend
        except Exception as e:
            # Raise any exception that occurs during the BEAST algorithm execution
            raise e

    @staticmethod
    def feature_extraction(energy: np.array, distance: np.array, mean_trend: np.array, cp: np.array,
                           Number_of_targets=None):

        """
        Perform feature extraction based on the input parameters.

        Parameters:
        - energy (np.array): Array containing energy values.
        - distance (np.array): Array containing distance values.
        - mean_trend (np.array): Array representing the mean trend.
        - cp (np.array): Array containing change points.
        - Number_of_targets (int, optional): Number of targets to consider. Defaults to None.
        """

        # Set default number of targets if not provided
        Number_of_targets = 2 if Number_of_targets is None else Number_of_targets

        # Initialize variables
        len_cp = len(cp) - 1
        mu = np.zeros(len_cp)
        std = np.zeros(len_cp)
        skew = np.zeros(len_cp)
        trend_vec = np.zeros(len_cp)
        times_vec = np.zeros(len_cp)
        sa_vec = np.zeros(len_cp)
        cumulated_time_vec = np.zeros(len_cp)
        assembly_mat = np.zeros((energy.shape[0], Number_of_targets))

        # Floor the mean trend values
        mean_trend = np.floor(mean_trend)

        # Convert change points to integers
        cp_int = np.vectorize(int)(cp)

        # Create assembly matrix based on distance values
        for i in range(Number_of_targets):
            assembly = np.zeros(energy.shape[0])
            mimic = np.where(distance[:, i] == 0)[0]
            assembly[mimic] = 1
            assembly_mat[:, i] = assembly

        # Extract features for each segment defined by change points
        for i in range(len_cp):
            start = cp_int[i]
            end = cp_int[i + 1]

            # Calculate mean, standard deviation, and skewness
            segment = energy[start:end, :]
            mu[i] = np.nanmean(segment, axis=0)
            std[i] = np.nanstd(segment, axis=0)
            median = np.nanmedian(segment, axis=0)
            skew[i] = (mu[i] - median) / 3 * std[i]

            # Calculate trend using linear fit
            x = np.arange(start, end)
            abc = np.polyfit(x, mean_trend[cp_int[i]:cp_int[i + 1]], 1)
            trend_vec[i] = abc[0]

            # Calculate duration of the segment
            times_vec[i] = end - start

            # Check for self-assembly events
            for j in range(Number_of_targets):
                sa_vec[i] += np.any(assembly_mat[start:end, j])
        sa_vec[sa_vec > 1] = 1

        # Calculate cumulated time to self-assembly
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

        # If no self-assembly events, return empty list
        if not np.any(sa_vec == 1):
            return []
        else:
            # Reshape feature vectors for outpu
            mu = np.reshape(mu, (len(mu), 1))
            std = np.reshape(std, (len(std), 1))
            skew = np.reshape(skew, (len(skew), 1))
            cumsum_vec = np.cumsum(times_vec)
            cumsum_vec = np.reshape(cumsum_vec, (len(cumsum_vec), 1))
            trend_vec = np.reshape(trend_vec, (len(trend_vec), 1))
            sa_vec = np.reshape(sa_vec, (len(sa_vec), 1))
            cumulated_time_vec = np.reshape(cumulated_time_vec, (len(cumulated_time_vec), 1))

            # Return extracted features as a list of arrays
            return [mu, std, skew, cumsum_vec, trend_vec, sa_vec, cumulated_time_vec]

    @staticmethod
    def feature_selection(segment_data_aggregated_output: np.array):
        """
        Perform feature selection on the aggregated output of segment data.

        Parameters:
        - segment_data_aggregated_output (np.array): Aggregated output of segment data.

        Returns:
        np.array: Combined reduced feature set after selection.
        """
        # Initialize an empty list to store the reduced feature sets
        c_reduced = []

        # Iterate over each list in the aggregated output
        for list in segment_data_aggregated_output:
            # Skip empty lists
            if len(list) == 0:
                continue

            # Extract feature vectors from the list
            x = list[0]  # mean_vec
            y = list[1]  # std_vec
            z = list[4]  # trend
            w = list[2]  # skew
            v = list[3]  # total_trajectory_time
            c = list[6]  # time_to_self_assembly

            # Find the index where time_to_self_assembly is zero
            ending_theme = np.where(c == 0)[0]
            if len(ending_theme) > 0:
                ending_theme = ending_theme[0]

            # Truncate the feature vectors up to the ending_theme index
            x = x[:ending_theme]
            y = y[:ending_theme]
            z = z[:ending_theme]
            c = c[:ending_theme]
            w = w[:ending_theme]
            v = v[:ending_theme]

            # Combine the truncated feature vectors into a single array
            d_reduced = np.column_stack((x, y, z, w, v, c))

            # Append the reduced feature set to the list
            c_reduced.append(d_reduced)

        # Stack the list of reduced feature sets into a single array
        c_reduced = np.vstack(tuple(c_reduced))

        # Return the combined reduced feature set
        return c_reduced

    @staticmethod
    def pca(c_reduced: np.array, n_components: int = 3):
        """
        Perform Principal Component Analysis (PCA) on the input data.

        Parameters:
        - c_reduced (np.array): Array of reduced features.
        - n_components (int): Number of principal components to retain. Defaults to 3.

        Returns:
        tuple: Principal components, normalized scores, and explained variance.
        """
        # Select the first n_components columns from the input array
        data = c_reduced[:, :n_components]

        # Calculate the mean and standard deviation of the selected data
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)

        # Normalize the data by subtracting the mean and dividing by the standard deviation
        norm_data = (data - data_mean) / data_std

        # Initialize PCA with the specified number of components
        pca = PCA(n_components=n_components)

        # Fit the PCA model to the normalized data and transform the data to the principal component space
        score = pca.fit_transform(norm_data)

        # Extract the principal components (eigenvectors)
        principal_components = pca.components_

        # Extract the explained variance (eigenvalues)
        latent = pca.explained_variance_

        # Calculate the covariance matrix of the scores
        score_cov = np.cov(score, rowvar=False)

        # Extract the standard deviation of the scores
        score_std = np.diag(score_cov)

        # Normalize the scores by dividing by their standard deviation
        score = score / (score_std[:, None] * np.ones((n_components, score.shape[0]))).T

        # Return the principal components, normalized scores, and explained variance
        return principal_components, score, latent

    @staticmethod
    def data_preparation(score: np.array, c_reduced: np.array, n_components: int = 3):
        """
        Perform data preparation by selecting and organizing features based on the input scores and reduced features.

        Parameters:
        - score (np.array): Array of scores containing mean, standard deviation, trend, skew, and total trajectory time.
        - c_reduced (np.array): Array of reduced features obtained from feature selection.
        - n_components (int): Number of components to consider. Defaults to 3.

        Returns:
        np.array: Array of prepared data based on the selected features.
        """
        a_reduced = []

        # If the number of components is 3, select specific features
        if n_components == 3:
            for i in range(score.shape[0]):
                x = score[i, 0]  # mean_vec
                y = score[i, 1]  # std_vec
                z = score[i, 2]  # trend
                c = c_reduced[i, 5]  # selected feature from c_reduced
                b_reduced = np.array([x, y, z, c])  # Combine selected features into an array
                a_reduced.append(b_reduced)  # Append the array to the list
        else:  # If the number of components is 5, select a different set of features
            for i in range(score.shape[0]):
                x = score[i, 0]  # mean_vec
                y = score[i, 1]  # std_vec
                z = score[i, 4]  # trend
                w = score[i, 2]  # skew
                v = score[i, 3]  # total_trajectory_time
                c = c_reduced[i, 5]  # selected feature from c_reduced
                b_reduced = np.array([x, y, z, w, v, c])  # Combine selected features into an array
                a_reduced.append(b_reduced)  # Append the array to the list

        # Convert the list of arrays to a numpy array
        a_reduced = np.array(a_reduced)

        return a_reduced

    @staticmethod
    def model_training_with_cv(a_reduced: np.array, n_components: int = 3, cv_num: int = 10):
        """
            Perform model training with cross-validation using the given reduced data.

            Parameters:
            - a_reduced (np.array): Array of reduced features for training.
            - n_components (int): Number of components to consider. Defaults to 3.
            - cv_num (int): Number of cross-validation iterations. Defaults to 10.

            Returns:
            tuple: Predicted values, actual values, mean error, and the random data used for training.

            Raises:
            Exception: If model building fails due to insufficient input data or target reach times.
        """
        np.random.seed(42)  # Set random seed for reproducibility
        idn = np.where(a_reduced[:, n_components] != 0)[0]  # Identify non-zero entries in the target column
        mapx = a_reduced[idn, :]  # Filter the reduced data based on identified indices
        mapx[:, n_components] = np.log(mapx[:, n_components])  # Apply log transformation to the target column
        train_index = int(np.floor(0.8 * mapx.shape[0]))  # Calculate the training set index
        validation_index = int(mapx.shape[0])  # Calculate the validation set index
        size_validation_set = len(range(train_index, validation_index))  # Calculate the size of the validation set
        tfas_predict_mat = np.zeros((cv_num, size_validation_set))  # Initialize matrix for predicted values
        tfas_actually_mat = np.zeros((cv_num, size_validation_set))  # Initialize matrix for actual values
        mean_error_mat = np.zeros((cv_num, size_validation_set))  # Initialize matrix for mean errors

        for i in range(cv_num):  # Loop over cross-validation iterations
            random_x = mapx[np.random.permutation(mapx.shape[0]), :]  # Shuffle the data
            train_index = int(np.floor(0.8 * random_x.shape[0]))  # Recalculate the training set index
            validation_index = int(random_x.shape[0])  # Recalculate the validation set index
            training_set = random_x[:train_index, :]  # Extract the training set
            validation_set = random_x[train_index:validation_index, :]  # Extract the validation set

            # Determine the min and max values for each component in the training set
            min_1 = np.nanmin(training_set[:, 0])
            max_1 = np.nanmax(training_set[:, 0])
            min_2 = np.nanmin(training_set[:, 1])
            max_2 = np.nanmax(training_set[:, 1])
            min_3 = np.nanmin(training_set[:, 2])
            max_3 = np.nanmax(training_set[:, 2])

            # Create linearly spaced arrays for each component
            d1 = np.linspace(min_1, max_1, 150)
            d2 = np.linspace(min_2, max_2, 150)
            d3 = np.linspace(min_3, max_3, 150)

            if n_components == 5:  # If using 5 components, create additional arrays
                min_4 = np.nanmin(training_set[:, 3])
                max_4 = np.nanmax(training_set[:, 3])
                min_5 = np.min(training_set[:, 4])
                max_5 = np.max(training_set[:, 4])
                d4 = np.linspace(min_4, max_4, 10)
                d5 = np.linspace(min_5, max_5, 10)
                x0, y0, z0, w0, v0 = np.meshgrid(d1, d2, d3, d4, d5, indexing='ij')
                XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel(), w0.ravel(), v0.ravel()))
            else:  # If using 3 components, create meshgrid for 3D space
                x0, y0, z0 = np.meshgrid(d1, d2, d3, indexing='ij')
                XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel()))

            X = training_set[:, :n_components]  # Extract the feature columns from the training set
            Y = training_set[:, n_components]  # Extract the target column from the training set

            # Interpolate the target values over the feature space
            YI = scipy.interpolate.griddata(X, Y, XI, method='linear')
            YI = YI.astype(float)  # Ensure the interpolated values are floats

            if n_components == 3:  # Create a convolution kernel for 3 components
                k = np.ones((n_components, n_components, n_components))
                k[1, 1, 1] = 0
                k /= k.sum()
            elif n_components == 5:  # Create a convolution kernel for 5 components
                k = np.ones((n_components, n_components, n_components, n_components, n_components))
                k[2, 2, 2, 2, 2] = 0
                k /= k.sum()

            # Apply convolution to smooth the interpolated values
            YI = np.reshape(YI, x0.shape)
            YI = scipy.signal.convolve(YI, k, mode='same', method='direct')
            YI = scipy.signal.convolve(YI, k, mode='same', method='direct')
            if np.isnan(YI).all():  # Check if all interpolated values are NaN
                raise Exception("Model building failed! reason: not enough "
                                "input data or not enough times in the dataset in which the target was reached")

            # Digitize the validation set features to find their corresponding indices in the interpolated space
            Ix = np.digitize(validation_set[:, 0], d1) - 1
            Iy = np.digitize(validation_set[:, 1], d2) - 1
            Iz = np.digitize(validation_set[:, 2], d3) - 1
            Ix[np.isnan(Ix)] = np.round(np.nanmean(Ix))
            Iy[np.isnan(Iy)] = np.round(np.nanmean(Iy))
            Iz[np.isnan(Iz)] = np.round(np.nanmean(Iz))

            if n_components == 5:  # If using 5 components, digitize the additional features
                Iw = np.digitize(validation_set[:, 3], d4) - 1
                Iv = np.digitize(validation_set[:, 4], d5) - 1
                Iw[np.isnan(Iw)] = np.round(np.nanmean(Iw))
                Iv[np.isnan(Iv)] = np.round(np.nanmean(Iv))

            tfas_real = np.zeros((validation_set.shape[0]))  # Initialize array for actual values
            tfas_predict = np.zeros((validation_set.shape[0]))  # Initialize array for predicted values

            for j in range(validation_set.shape[0]):  # Loop over the validation set
                tfas_real[j] = validation_set[j, n_components]  # Store the actual value
                if n_components == 3:  # Predict the value based on the interpolated space
                    tfas_predict[j] = YI[int(Ix[j]), int(Iy[j]), int(Iz[j])]
                elif n_components == 5:
                    tfas_predict[j] = YI[int(Ix[j]), int(Iy[j]), int(Iz[j]), int(Iw[j]), int(Iv[j])]
                if np.isnan(tfas_predict[j]):  # Handle NaN predictions
                    tfas_predict[j] = np.nanmean(Y)

            tfas_predict_mat[i, :] = tfas_predict  # Store the predicted values for this iteration
            tfas_actually_mat[i, :] = tfas_real  # Store the actual values for this iteration
            mean_error_mat[i, :] = np.abs(tfas_real - tfas_predict)  # Calculate and store the mean error

        return YI, tfas_predict_mat, tfas_actually_mat, mean_error_mat, random_x  # Return the results

    @staticmethod
    def draw_stochastic_landscape_2d(random_x: np.array, save_path, n_components):
        """
        Draws a 2D stochastic landscape based on the provided data.

        Parameters:
        - random_x (np.array): Array of random data points.
        - save_path: Path to save the generated plot.
        - n_components (int): Number of components to consider.

        Returns:
        - None
        """
        # Determine the training set size (80% of the data)
        train_index = int(np.floor(0.8 * random_x.shape[0]))
        training_set = random_x[:train_index, :]

        # Find the minimum and maximum values for the first two components
        min_1 = np.nanmin(training_set[:, 0])
        max_1 = np.nanmax(training_set[:, 0])
        min_2 = np.nanmin(training_set[:, 1])
        max_2 = np.nanmax(training_set[:, 1])

        # Create a grid of points for interpolation
        d1 = np.linspace(min_1, max_1, 150)
        d2 = np.linspace(min_2, max_2, 150)
        x0, y0 = np.meshgrid(d1, d2, indexing='ij')

        # Extract the first two components and the target component from the training set
        X = training_set[:, :2]
        Y = training_set[:, n_components]

        # Interpolate the target values over the grid
        XI = np.column_stack((x0.ravel(), y0.ravel()))
        YI = scipy.interpolate.griddata(X, Y, XI, method='linear')
        YI = YI.astype(float)

        # Create a convolution kernel
        k = np.ones((2, 2))
        k[1, 1] = 0
        k /= k.sum()

        # Reshape the interpolated values to match the grid shape
        YI = np.reshape(YI, x0.shape)

        # Apply convolution to smooth the interpolated values
        YI = scipy.signal.convolve(YI, k, mode='same', method='direct')
        YI = scipy.signal.convolve(YI, k, mode='same', method='direct')

        # Replace NaN values with a large constant
        YI[np.isnan(YI)] = np.log(5 * 10 ** 3)

        # Create a plot of the stochastic landscape
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(YI, cmap='jet')
        fig.colorbar(im, ax=ax)
        ax.set_title('Predictor Space in 2D')
        ax.set_xticks([])
        ax.set_yticks([])

        # Save the plot to the specified path
        plt.savefig(os.path.join(save_path, 'Stochastic Landscape 2D.png'))
        plt.close(fig)

    @staticmethod
    def model_eval(tfas_predict_mat, tfas_actually_mat, cv_num, save_path):
        """
        Perform model evaluation by comparing predicted and actual values using cross-validation.

        Parameters:
        - tfas_predict_mat (np.array): Matrix of predicted values.
        - tfas_actually_mat (np.array): Matrix of actual values.
        - cv_num (int): Number of cross-validation iterations.
        - save_path (str): Path to save the evaluation results.

        Returns:
        tuple: Mean vector, standard deviation vector, histogram space, x histogram space, x ticks, y ticks.
        """
        bin_width = 0.5

        # Determine the minimum and maximum predicted values across all cross-validation iterations
        min_of_all = np.nanmin(np.nanmin(tfas_predict_mat, axis=1))
        max_of_all = np.nanmax(np.nanmax(tfas_predict_mat, axis=1))

        # Calculate the number of bins for the histogram
        n_bins = int(np.ceil(max_of_all / bin_width) - np.floor(min_of_all / bin_width) + 1)

        # Create histogram space based on the bin width
        hist_space = np.linspace(bin_width * np.floor(min_of_all / bin_width),
                                 bin_width * np.ceil(max_of_all / bin_width),
                                 n_bins)

        # Initialize arrays to store mean and standard deviation for each bin across cross-validation iterations
        mean = np.zeros((cv_num, len(hist_space) - 1))
        std = np.zeros((cv_num, len(hist_space) - 1))

        # Create a color map for plotting
        color_map = plt.cm.jet(np.linspace(0, 1, cv_num))
        color_map = color_map[:, :-1]

        # Create a figure for plotting
        fig_1 = plt.figure(1, figsize=(10, 7))

        # Create subplots for scatter plot and histogram
        a = fig_1.add_subplot(2, 1, 1)
        b = fig_1.add_subplot(2, 1, 2)

        # Define x-axis ticks based on the bin width
        x_ticks = np.arange(min_of_all, max_of_all + bin_width, bin_width)

        # Determine the maximum value of the actual values
        max_labels = np.nanmax(np.nanmax(tfas_actually_mat, axis=0))

        # Define y-axis ticks based on the maximum value
        if max_of_all > max_labels:
            y_ticks = np.arange(min_of_all, max_of_all + 3, 3)
        else:
            y_ticks = np.arange(min_of_all, max_labels + 3, 3)

        # Set x and y ticks for the scatter plot
        a.set_xticks(x_ticks)
        a.set_yticks(y_ticks)
        a.set_xlim(x_ticks[0], x_ticks[-1])
        a.set_ylim(y_ticks[0], y_ticks[-1])
        a.set_ylabel("True Value")
        a.set_xlabel("Predicted Value")

        # Add vertical lines to the scatter plot for each x tick
        for n in range(1, len(x_ticks)):
            a.axvline(x_ticks[n - 1], color='k', linestyle='--', linewidth=0.5)

        # Loop through each cross-validation iteration
        for i in range(cv_num):
            x = np.squeeze(tfas_predict_mat[i, :])
            y = np.squeeze(tfas_actually_mat[i, :])

            # Sort the predicted values and corresponding actual values
            sorted_indices = np.argsort(x)
            x = x[sorted_indices]
            y = y[sorted_indices]

            # Scatter plot of predicted vs actual values
            a.scatter(x, y, color=color_map[i, :], marker='o')

            # Initialize arrays to store mean and standard deviation for each bin
            std_hista = np.zeros(len(hist_space) - 1)
            mean_hista = np.zeros(len(hist_space) - 1)

            # Loop through each bin in the histogram space
            for j in range(1, len(hist_space)):
                Ind = np.where((hist_space[j - 1] < x) & (x < (hist_space[j])))[0]
                if len(Ind) < np.floor(0.01 * len(x)):
                    std_hista[j - 1] = np.nan
                    mean_hista[j - 1] = np.nan
                else:
                    yo = np.sort(y[Ind])
                    mean_hista[j - 1] = np.nanmedian(y[Ind])
                    if len(yo) > 1:
                        aop = int(np.ceil(0.16 * len(yo)))
                        std_hista[j - 1] = mean_hista[j - 1] - yo[aop]
                    else:
                        std_hista[j - 1] = mean_hista[j - 1] - yo[0]

            # Calculate the center of each bin
            x_hist_space = (hist_space[1:] + hist_space[:-1]) / 2
            i_4 = len(mean_hista) - 1

            # Scatter plot of mean actual values for each bin
            b.scatter(x_hist_space[:i_4], mean_hista[:i_4], marker='s', color=color_map[i, :], zorder=1)

            # Store mean and standard deviation for each bin
            mean[i, :len(mean_hista)] = mean_hista
            std[i, :len(std_hista)] = std_hista

            # Scatter plot of predicted vs actual values up to the last bin
            i_5 = np.argmin(np.abs(x - x_hist_space[i_4]))
            a.scatter(x[:i_5], y[:i_5], color=color_map[i, :], marker='o')

        # Initialize arrays to store overall mean and standard deviation for each bin
        mean_vec = np.zeros(len(hist_space) - 1)
        std_vec = np.zeros(len(hist_space) - 1)

        # Replace zeros with NaN in mean and standard deviation arrays
        mean[mean == 0] = np.nan
        std[std == 0] = np.nan

        # Calculate overall mean and standard deviation for each bin
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

        # Plot the perfect predictor line
        b.plot(x_hist_space[:i_4], x_hist_space[:i_4], linestyle='--', color='k', linewidth=0.5, zorder=2,
               label="Perfect Predictor")

        # Plot the mean predictor with error bars
        b.errorbar(x_hist_space, mean_vec, yerr=std_vec, color='k', ecolor='k', zorder=3,
                   label="Mean Predictor across iterations")

        # Add vertical lines and bin labels to the histogram plot
        for k in range(1, len(x_ticks)):
            b.axvline(x_ticks[k - 1], color='k', linestyle='--', linewidth=0.5)
            x_pos = (x_ticks[k - 1] + x_ticks[k]) / 2
            y_pos = y_ticks[-1] - 2
            b.text(x_pos, y_pos, f"Bin {k}", ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7),
                   zorder=4)

        # Set x and y ticks for the histogram plot
        b.set_xticks(x_ticks)
        b.set_yticks(y_ticks)
        b.set_xlim(x_ticks[0], x_ticks[-1])
        b.set_ylim(y_ticks[0], y_ticks[-1])
        b.set_ylabel("Scatter Plot Average for each bin")
        b.set_xlabel("Center of Predictor bin")
        b.legend(loc='upper right')

        # Save the figure
        fig_1.savefig(os.path.join(save_path, 'Predictions Scatter and Histogram.png'))
        plt.close(fig_1)

        # Return the calculated mean vector, standard deviation vector, histogram space, x histogram space, x ticks, and y ticks
        return mean_vec, std_vec, hist_space, x_hist_space, x_ticks, y_ticks

    @staticmethod
    def train_again_on_validation(a_reduced, n_components=3):
        """
                Train the model again using the validation and test sets.

                Parameters:
                - a_reduced (np.array): Array of reduced features.
                - n_components (int): Number of components to consider. Defaults to 3.

                Returns:
                tuple: Predicted values, actual values, and mean error for the validation set.

                Raises:
                Exception: If model building fails due to insufficient input data or target reach times.
        """
        np.random.seed(42)  # Set random seed for reproducibility

        # Identify non-zero entries in the target column
        idn = np.where(a_reduced[:, n_components] != 0)[0]
        mapx = a_reduced[idn, :]

        # Apply log transformation to the target column
        mapx[:, n_components] = np.log(mapx[:, n_components])

        # Shuffle the data
        random_x = mapx[np.random.permutation(mapx.shape[0]), :]

        # Split the data into training and validation sets
        validation_index = int(np.ceil(0.8 * mapx.shape[0]))
        training_set = random_x[:validation_index, :]
        validation_set = random_x[validation_index:, :]

        # Determine the min and max values for each component in the training set
        min_1 = np.nanmin(training_set[:, 0])
        max_1 = np.nanmax(training_set[:, 0])
        min_2 = np.nanmin(training_set[:, 1])
        max_2 = np.nanmax(training_set[:, 1])
        min_3 = np.nanmin(training_set[:, 2])
        max_3 = np.nanmax(training_set[:, 2])

        # Create linearly spaced arrays for each component
        d1 = np.linspace(min_1, max_1, 150)
        d2 = np.linspace(min_2, max_2, 150)
        d3 = np.linspace(min_3, max_3, 150)

        if n_components == 5:
            # If using 5 components, create additional arrays
            min_4 = np.nanmin(training_set[:, 3])
            max_4 = np.nanmax(training_set[:, 3])
            min_5 = np.min(training_set[:, 4])
            max_5 = np.max(training_set[:, 4])
            d4 = np.linspace(min_4, max_4, 10)
            d5 = np.linspace(min_5, max_5, 10)
            x0, y0, z0, w0, v0 = np.meshgrid(d1, d2, d3, d4, d5, indexing='ij')
            XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel(), w0.ravel(), v0.ravel()))
        else:
            # If using 3 components, create meshgrid for 3D space
            x0, y0, z0 = np.meshgrid(d1, d2, d3, indexing='ij')
            XI = np.column_stack((x0.ravel(), y0.ravel(), z0.ravel()))

        # Extract the feature columns and target column from the training set
        X = training_set[:, :n_components]
        Y = training_set[:, n_components]

        # Interpolate the target values over the feature space
        YI = scipy.interpolate.griddata(X, Y, XI, method='linear')
        YI = YI.astype(float)  # Ensure the interpolated values are floats

        if n_components == 3:
            # Create a convolution kernel for 3 components
            k = np.ones((n_components, n_components, n_components))
            k[1, 1, 1] = 0
            k /= k.sum()
        elif n_components == 5:
            # Create a convolution kernel for 5 components
            k = np.ones((n_components, n_components, n_components, n_components, n_components))
            k[2, 2, 2, 2, 2] = 0
            k /= k.sum()

        # Reshape the interpolated values to match the grid shape
        YI = np.reshape(YI, x0.shape)

        # Apply convolution to smooth the interpolated values
        YI = scipy.signal.convolve(YI, k, mode='same', method='direct')
        YI = scipy.signal.convolve(YI, k, mode='same', method='direct')

        if np.isnan(YI).all():
            # Check if all interpolated values are NaN
            raise Exception("Model building failed! reason: not enough "
                            "input data or not enough times in the dataset in which the target was reached")

        # Digitize the validation set features to find their corresponding indices in the interpolated space
        Ix = np.digitize(validation_set[:, 0], d1) - 1
        Iy = np.digitize(validation_set[:, 1], d2) - 1
        Iz = np.digitize(validation_set[:, 2], d3) - 1
        Ix[np.isnan(Ix)] = np.round(np.nanmean(Ix))
        Iy[np.isnan(Iy)] = np.round(np.nanmean(Iy))
        Iz[np.isnan(Iz)] = np.round(np.nanmean(Iz))

        if n_components == 5:
            # If using 5 components, digitize the additional features
            Iw = np.digitize(validation_set[:, 3], d4) - 1
            Iv = np.digitize(validation_set[:, 4], d5) - 1
            Iw[np.isnan(Iw)] = np.round(np.nanmean(Iw))
            Iv[np.isnan(Iv)] = np.round(np.nanmean(Iv))

        # Initialize arrays for actual and predicted values
        tfas_actually_mat_2 = np.zeros((validation_set.shape[0]))
        tfas_predict_mat_2 = np.zeros((validation_set.shape[0]))

        for j in range(validation_set.shape[0]):
            # Store the actual value
            tfas_actually_mat_2[j] = validation_set[j, n_components]
            if n_components == 3:
                # Predict the value based on the interpolated space
                tfas_predict_mat_2[j] = YI[int(Ix[j]), int(Iy[j]), int(Iz[j])]
            elif n_components == 5:
                tfas_predict_mat_2[j] = YI[int(Ix[j]), int(Iy[j]), int(Iz[j]), int(Iw[j]), int(Iv[j])]
            if np.isnan(tfas_predict_mat_2[j]):
                # Handle NaN predictions
                tfas_predict_mat_2[j] = np.nanmean(Y)

        # Calculate the mean error
        mean_error_mat_2 = np.abs(tfas_actually_mat_2 - tfas_predict_mat_2)

        # Reshape the arrays for output
        tfas_predict_mat_2 = np.reshape(tfas_predict_mat_2, (len(tfas_predict_mat_2), 1))
        tfas_actually_mat_2 = np.reshape(tfas_actually_mat_2, (len(tfas_actually_mat_2), 1))
        mean_error_mat_2 = np.reshape(mean_error_mat_2, (len(mean_error_mat_2), 1))

        return tfas_predict_mat_2, tfas_actually_mat_2, mean_error_mat_2, YI

    @staticmethod
    def multiple_boxplot(input_data, x_labels, legend_label, ax):
        """
        Perform multiple box plots with customized styling and labels.

        Parameters:
        - input_data (list of lists): Data for box plots.
        - x_labels (list): Labels for the x-axis.
        - legend_label (str): Label for the legend.
        - ax (matplotlib axis): Axis to plot the box plots.

        Returns:
        - None
        """
        L = len(input_data)  # Number of box plots to create
        positions = np.arange(1, L + 1)  # Positions for the box plots on the x-axis

        # Create box plot
        bp1 = ax.boxplot(input_data, positions=positions, patch_artist=True, widths=0.6)

        # Set x-axis ticks and labels
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{x:.2f}" for x in x_labels])  # Format x-axis labels to 2 decimal places
        ax.set_xlim(0.5, L + 0.5)  # Set x-axis limits

        # Customize box plots
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp1[element], color='black')  # Set color of whiskers, fliers, means, medians, and caps to black

        for box in bp1['boxes']:
            box.set(facecolor='white', edgecolor='red',
                    linewidth=2)  # Set box face color to white and edge color to red

        for flier in bp1['fliers']:
            flier.set(marker='o', color='red', alpha=0.7)  # Set flier marker to red circles with alpha transparency

        # Set labels and title
        ax.set_xlabel("Predicted Value")  # Set x-axis label
        ax.set_ylabel('True Value')  # Set y-axis label

        # Add legend entries
        bp1["boxes"][0].set_label(legend_label)  # Set legend label for the first box plot
        ax.plot(positions, x_labels, linestyle='--', color='k',
                label='Perfect Predictor')  # Plot perfect predictor line

        # Add legend
        ax.legend(loc='upper right')  # Add legend to the upper right corner
        plt.gcf().subplots_adjust(bottom=0.3)  # Adjust subplot parameters to give some padding at the bottom

        # Adjust y-axis limits
        all_values = [val for sublist in input_data for val in sublist]  # Flatten the input data
        y_min, y_max = min(all_values), max(all_values)  # Find min and max values in the data
        y_range = y_max - y_min  # Calculate the range of y values
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)  # Set y-axis limits with some padding

    # def plot_helper(ax, data, box_positions, color, legend_label, n_boxes_last_group):
    #     # Create boxplot with specified data and positions
    #     bp = ax.boxplot(data, positions=np.round(box_positions, 2), patch_artist=True, widths=0.09)
    #
    #     # Apply colors to the boxes
    #     if n_boxes_last_group > 0:
    #         # Apply color to the last group of boxes
    #         for i, box in enumerate(bp['boxes'][:n_boxes_last_group], start=n_boxes_last_group):
    #             box.set(facecolor='white', edgecolor=color, linewidth=2)
    #     else:
    #         # Apply color to all boxes
    #         for i, box in enumerate(bp['boxes']):
    #             box.set(facecolor='white', edgecolor=color, linewidth=2)
    #
    #     # Set color for medians, whiskers, and caps
    #     for i, element in enumerate(bp['medians'] + bp['whiskers'] + bp['caps']):
    #         element.set_color('black')
    #
    #     # Set color for fliers
    #     for i, element in enumerate(bp['fliers']):
    #         element.set_markeredgecolor('black')
    #
    #     # Set label for the last group of boxes
    #     bp["boxes"][n_boxes_last_group - 1].set_label(legend_label)
    @staticmethod
    def plot_helper(ax, data, box_positions, colors, legend_labels, box_width=0.09):
        # Create boxplot with specified data and positions
        box_positions1 = box_positions[::2]
        box_positions2 = box_positions[1::2]

        # Extract data for the two groups
        data1 = data[0]
        data2 = data[1]

        # Create boxplots for both groups
        bp1 = ax.boxplot(data1, positions=np.round(box_positions1, 2), patch_artist=True, widths=box_width)
        bp2 = ax.boxplot(data2, positions=np.round(box_positions2, 2), patch_artist=True, widths=box_width)

        # bp = ax.boxplot(data, positions=np.round(box_positions, 2), patch_artist=True, widths=box_width)

        for box in bp1['boxes']:
            box.set(facecolor='white', edgecolor='blue', linewidth=2)
        for box in bp2['boxes']:
            box.set(facecolor='white', edgecolor='red', linewidth=2)

            # Set color for medians, whiskers, and caps
        for element in bp1['medians'] + bp1['whiskers'] + bp1['caps'] + bp2['medians'] + bp2['whiskers'] + bp2['caps']:
            element.set_color('black')

            # Set color for fliers
        for element in bp1['fliers'] + bp2['fliers']:
            element.set_markeredgecolor('black')

            # Set legend labels
        bp1['boxes'][0].set_label(legend_labels[0])
        bp2['boxes'][0].set_label(legend_labels[1])

    @staticmethod
    def multiple_boxplot_2(data, x_labels, legend_labels, colormap, ax, x_hist_space):
        """
                Create multiple box plots with customized styling and labels.

                Parameters:
                - data (list of lists): Data for box plots, where each sublist represents a group of data points.
                - x_labels (list): Labels for the x-axis.
                - legend_labels (list): Labels for the legend, corresponding to each group of data.
                - colormap (array): Array of colors for the scatter plot.
                - ax (matplotlib axis): Axis to plot the box plots.
                - x_hist_space (array): Array of x positions for the scatter plot.

                Returns:
                - None
        """
        # Get sizes
        M, L = len(data), len(data[0])  # M is the number of groups, L is the number of boxes in each group
        w = 0.25  # width of boxes
        if (np.array([x_labels[i + 1] - x_labels[i] for i in range(len(x_labels) - 1)]) ==
            np.array([x_labels[i + 1] - x_labels[i] for i in range(len(x_labels) - 1)])).all():
            if M > 1:
                YY = np.sort(np.concatenate((x_labels - w / 2, x_labels + w / 2)))
            else:
                positions = x_labels[:, 0]
                YY = positions
        else:
            if M > 1:
                positions = np.arange(1, M * L * w + 1 + w * L, w)
                positions = positions[positions <= M * L]
                YY = x_labels[0] - 1 + positions - w / 2
            else:
                positions = np.arange(1, M * L * w + 1 + w * L, w)
                positions = positions[positions % (M + 1) != 0]
                YY = x_labels[0] - 1 + positions - w
        # Extract data and label it in the group correctly
        x = []
        group = []
        for i in range(M):
            for j in range(L):
                aux = data[i][j]
                x.extend(aux)  # Flatten the data
                group.extend([j + i * M] * len(aux))  # Create group labels

        # Define x-axis limits
        x_lims = [np.min(YY) - 0.5, np.max(YY) + 0.5]
        # Define positions for the two groups of boxes
        box_positions1 = YY[::2]
        box_positions2 = YY[1::2]

        positions = np.arange(1, M * L * (0.09 + 0.1) + 1, 0.09 + 0.1)
        positions = positions[:M * L]
        YY = x_labels[0] - 1 + positions - (0.09 + 0.1) / 2

        # Extract data for the two groups
        # data1 = data[0]
        # data2 = data[1]
        # flat_data = [item for sublist in data for item in sublist]
        colors = ['red' if i % 2 == 0 else 'blue' for i in range(M * L)]
        # Extract legend labels for the two groups
        # legend_label1 = legend_labels[0]
        # legend_label2 = legend_labels[1]

        # Plot the first group of boxes
        # SLM_tools.plot_helper(ax, data1, box_positions1, 'red', legend_label1, 0)
        SLM_tools.plot_helper(ax, data, YY, colors, legend_labels * L, 0.09)

        # Plot the second group of boxes
        # SLM_tools.plot_helper(ax, data2, box_positions2, 'blue', legend_label2, len(data1))
        # SLM_tools.plot_helper(ax, data2, box_positions2, colors, legend_label2, len(data1))

        # Flatten all values for setting y-axis limits
        all_values = [val for sublist in data for val in sublist][0]

        # Set x and y labels
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Predictor Error")

        # Draw a horizontal line at y=0
        ax.axhline(y=0, linestyle='--', color='black')

        # Scatter plot for relative weight of bins
        ax.scatter(x_hist_space, np.ones(x_hist_space.shape) * (np.max(all_values) + 1), color=colormap, s=100,
                   edgecolors='black', label='Relative Weight of Bin')

        # Add legend
        ax.legend(loc='lower left', fontsize=6)

        # Set x-axis limits
        ax.set_xlim(x_lims)

    @staticmethod
    def normalize_array(arr):
        """
        Normalize an array by subtracting twice the median if the median is less than 0.

        Parameters:
        arr (np.array): Input array to be normalized.

        Returns:
        np.array: Normalized array after subtraction.
        """
        median = np.median(arr)
        return arr - 2 * median if median < 0 else arr

    @staticmethod
    def cv_bias_correction(tfas_predict_mat_2, tfas_actually_mat_2, hist_space, mean_vec, x_hist_space,
                           x_ticks, y_ticks, save_path):
        """
            Perform cross-validation bias correction and plot the results.

            Parameters:
            - tfas_predict_mat_2 (np.array): Matrix of predicted values.
            - tfas_actually_mat_2 (np.array): Matrix of actual values.
            - hist_space (np.array): Histogram space for binning.
            - mean_vec (np.array): Mean vector for bias correction.
            - x_hist_space (np.array): X-axis histogram space.
            - x_ticks (np.array): X-axis ticks for plotting.
            - y_ticks (np.array): Y-axis ticks for plotting.
            - save_path (str): Path to save the generated plots.

            Returns:
            - No
        """
        bin_width = 0.5
        median_fit_vec = np.nanmedian(np.nanmedian(tfas_predict_mat_2))

        # Flatten the matrices and sort by predicted values
        x = np.squeeze(tfas_predict_mat_2)
        y = np.squeeze(tfas_actually_mat_2)
        tfas_predict_mat_2_sorted_indices = np.argsort(x)
        x = x[tfas_predict_mat_2_sorted_indices]
        y = y[tfas_predict_mat_2_sorted_indices]

        cutofflength = len(mean_vec)
        occurrence_probability = np.zeros((cutofflength, 1))

        # Calculate the offset for bias correction
        cv_offset = mean_vec - x_hist_space
        cv_offset[np.isnan(cv_offset)] = 0

        cv_corrected_x = np.zeros_like(x)
        new_y = np.zeros(y.shape)

        fig_2 = plt.figure(figsize=(12, 24))
        # Use GridSpec for more control over subplot layout
        gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
        # Create subplots with shared x-axis
        a = fig_2.add_subplot(gs[0])
        b = fig_2.add_subplot(gs[1])
        c = fig_2.add_subplot(gs[2])

        # Scatter plot of original predicted vs actual values
        scat_1 = a.scatter(x, y, color=(0.7, 0.7, 0.7), marker='o', label="Original Predictor")
        a.set_ylabel("True Value")
        a.set_xlabel("Predicted Value")
        a.set_xticks(x_ticks)
        a.set_yticks(y_ticks)
        a.set_xlim(x_ticks[0], x_ticks[-1])
        a.set_ylim(y_ticks[0], y_ticks[-1])

        # Apply bias correction
        for i in range(1, cutofflength - 1):
            Ind = np.where((hist_space[i - 1] < x) & (x < (hist_space[i])))[0]
            if len(Ind) < np.floor(0.01 * len(x)):
                occurrence_probability[i - 1] = len(Ind)
                cv_corrected_x[Ind] = x[Ind] + cv_offset[i - 1]
                new_y[Ind] = y[Ind]
                scat_2 = a.scatter(cv_corrected_x[Ind], y[Ind], color='k', marker='s')
            else:
                occurrence_probability[i - 1] = len(Ind)
                cv_corrected_x[Ind] = x[Ind] + cv_offset[i - 1]
                scat_2 = a.scatter(cv_corrected_x[Ind], y[Ind], color='k', marker='s')
                new_y[Ind] = y[Ind]

        # Sort corrected values for linear regression
        sorted_indices = np.argsort(cv_corrected_x)
        new_y_sorted = new_y[sorted_indices]
        cv_corrected_x_sorted = cv_corrected_x[sorted_indices]

        # Perform linear regression on corrected values
        slope, intercept, r_value, p_value, std_err = linregress(cv_corrected_x_sorted, new_y_sorted)
        R_x = np.array([cv_corrected_x_sorted[0], new_y_sorted[-1]])
        R_y = intercept + slope * R_x
        a.plot(R_x, R_y, color='r', linestyle='--', label='Linear Regression')

        # Add legend to the plot
        scat_2.set_label("CV Bias Corrected Predictor")
        a.legend()

        new_x = np.copy(cv_corrected_x_sorted)

        # Define new histogram space for corrected values
        min_of_all2 = np.min(new_x)
        max_of_all2 = np.max(new_x)
        hist_space_2 = np.linspace(bin_width * np.floor(min_of_all2 / bin_width),
                                   bin_width * np.ceil(max_of_all2 / bin_width),
                                   int(np.ceil(max_of_all2 / bin_width) - np.floor(min_of_all2 / bin_width) + 1))
        x_hist_space_2 = (hist_space_2[1:] + hist_space_2[:-1]) / 2
        cutofflength2 = len(hist_space_2) - 1

        mean_hista_last_fig2 = np.full(cutofflength2, np.nan)
        occurrence_probability_r = np.full(cutofflength2, np.nan)
        yo_lst = []
        z1_lst = []
        z0_lst = []
        mega_kde = []

        # Calculate mean and occurrence probability for corrected values
        for i in range(1, cutofflength2 + 1):
            Ind = np.where((hist_space_2[i - 1] < new_x) & (new_x < (hist_space_2[i])))[0]
            if len(Ind) < 5:
                occurrence_probability_r[i - 1] = len(Ind)
            else:
                yo = np.sort(new_y_sorted[Ind])
                mean_hista_last_fig2[i - 1] = np.median(yo)
                z0 = -(median_fit_vec - yo)  # trivial predictor error
                z1 = -((x_hist_space_2[i - 1]) - yo)  # post information predictor error
                occurrence_probability_r[i - 1] = len(Ind)
                mega_kde.append([z1, z0, yo, i - 1])
                z1_lst.append(z1)
                z0_lst.append(z0)
                yo_lst.append(yo)

        occurrence_probability_r = occurrence_probability_r / np.sum(occurrence_probability_r)
        I2 = (~np.isnan(x_hist_space_2)) & (~np.isnan(mean_hista_last_fig2))

        # Plot boxplot for corrected values
        SLM_tools.multiple_boxplot(input_data=yo_lst, x_labels=x_hist_space_2[I2],
                                   legend_label="CV Bias Corrected Predictor", ax=b)

        # Normalize errors for plotting
        z0_lst = [SLM_tools.normalize_array(z0) for z0 in z0_lst]
        z1_lst = [SLM_tools.normalize_array(z1) for z1 in z1_lst]
        data = [z0_lst, z1_lst]

        # Calculate color map based on occurrence probability
        ind = np.round(occurrence_probability_r * 1000) + 1
        ind[ind < 1] = 1
        ind = np.ceil(1000 * np.log(ind) / np.log(1000))
        ind[ind < 1] = 1
        ind = np.squeeze(ind)
        colormap = plt.cm.get_cmap('cool')
        colormap = colormap(np.linspace(0, 1, 1000))
        colormap = colormap[ind.astype(int), :]

        # Plot multiple boxplots for errors
        SLM_tools.multiple_boxplot_2(data=data,
                                     x_labels=x_hist_space_2[I2],
                                     legend_labels=["CV Bias Corrected Predictor Mean Error",
                                                    "Naive (median) Predictor Mean Error"],
                                     colormap=colormap,
                                     ax=c,
                                     x_hist_space=x_hist_space_2)

        # Add color bar and save the figure
        fig_2.colorbar(plt.cm.ScalarMappable(cmap='cool'), ax=c)
        fig_2.savefig(os.path.join(save_path, 'Predictor Evaluation.png'), bbox_inches='tight')
        # fig_2.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05)
        # plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.05, hspace=0.3)
        # fig_2.align_ylabels([a, b, c])
        # a_pos = a.get_position()
        # b_pos = b.get_position()
        # b.set_position([a_pos.x0, b_pos.y0, a_pos.width, b_pos.height])
        b.yaxis.set_label_coords(-10.5, 0.5)
        plt.close(fig_2)

    @staticmethod
    def create_and_evaluate_stochastic_landscape(dir_path, n_components, downsampling_factor, target_num,
                                                 time_vec_exists, cv_num, save_path, data_variable_name):
        """
                Create and evaluate a stochastic landscape based on the provided data.

                This method performs the following steps:
                1. Load and preprocess data from the specified directory.
                2. Perform segmentation using the BEAST algorithm.
                3. Extract features from the segmented data.
                4. Select and reduce features using PCA.
                5. Train a model using cross-validation.
                6. Evaluate the model and generate plots.

                Parameters:
                - dir_path (str): Directory path containing the data files.
                - n_components (int): Number of principal components to retain.
                - downsampling_factor (int): Factor by which to downsample the data.
                - target_num (int): Number of target columns to load.
                - time_vec_exists (bool): Flag indicating if a time vector exists in the data.
                - cv_num (int): Number of cross-validation iterations.
                - save_path (str): Path to save the evaluation results and plots.
                - data_variable_name (str): Variable name in the .mat file (only used if data_type is '.mat').

                Returns:
                - None

                Raises:
                - Exception: If any error occurs during the process.
        """
        # Get the current date and time for logging purposes
        current_datetime = datetime.now().strftime("%d_%m_%H_%M")

        # Get the total number of data files in the directory
        total_data_files = len(os.listdir(dir_path))

        # Initialize counters and logs
        counters = {"load": [], "extraction": [], "segmentation": [],
                    "feature_selection": 0, "N_extracted_features": [],
                    "total_features_extracted": 0, "features_selected": 0,
                    "total_runtime": 0, "files_with_target": 0, "log": [],
                    }

        # Initialize a list to store extracted features
        C = []

        # Iterate over each file in the directory
        for i, file in enumerate(os.listdir(dir_path)):
            file_path = os.path.abspath(os.path.join(dir_path, file))
            if os.path.isfile(file_path):
                file_name, file_extension = os.path.splitext(file)

                # Start timing the loading process
                start1 = time.time()

                # Set default data variable name if not provided
                if not data_variable_name:
                    data_variable_name = 'foo'

                # Load data based on whether a time vector exists
                if not time_vec_exists:
                    values_vec, distance_data = SLM_tools.load_data(file_path, data_variable_name, int(target_num),
                                                                    file_extension, time_vec_exists)
                else:
                    values_vec, time_vec, distance_data = SLM_tools.load_data(file_path, data_variable_name,
                                                                              int(target_num), file_extension,
                                                                              time_vec_exists)
                    values_vec, time_vec = SLM_tools.interpolate_data_over_regular_time(values_vec, time_vec)

                # Downsample the data if a downsampling factor is provided
                if downsampling_factor:
                    values_vec, distance_data, time_vec = SLM_tools.downsample(values_vec, distance_data,
                                                                               downsampling_factor)

                # Reshape the loaded data
                values_vec = np.reshape(values_vec, (len(values_vec), 1))
                distance_data = np.reshape(distance_data, (len(values_vec), int(target_num)))

                # Calculate and log the loading time
                loading_time = round(time.time() - start1, 3)
                log1 = f"\nFile {file_name}, {i + 1}/{total_data_files}  loaded, Loading time: {loading_time} seconds"
                counters["log"].append(log1)
                logging.info(log1)

                # Start timing the segmentation process
                start2 = time.time()

                # Perform segmentation using the BEAST algorithm
                o, cp, mean_trend = SLM_tools.beast(values_vec)

                # Calculate and log the segmentation time
                segmentation_time = round(time.time() - start2, 3)
                log2 = f"\nFinished Segmentation, runtime: {segmentation_time} seconds"
                counters["log"].append(log2)
                logging.info(log2)

                # Start timing the feature extraction process
                start3 = time.time()

                # Extract features from the segmented data
                A = SLM_tools.feature_extraction(values_vec, distance_data, mean_trend, cp)

                # Calculate and log the feature extraction time
                extraction_time = round(time.time() - start3, 3)
                log = f"\nFinished feature extraction {file_name}, {i + 1}/{total_data_files}, runtime: {extraction_time} seconds"
                counters["log"].append(log)
                logging.info(log)

                # Log whether the target was reached
                log = f"\nTarget Reached? : {len(A) > 0}"
                counters["log"].append(log)
                logging.info(log)

                # Update counters if features were extracted
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

                # Append extracted features to the list
                C.append(A)

                # Update the total runtime counter
                counters["total_runtime"] += (loading_time + extraction_time + segmentation_time)

        # Log the total preprocessing time
        log6 = f"\nFinished pre-processing,total runtime {counters['total_runtime']} for {total_data_files} files"
        counters["log"].append(log6)
        logging.info(log6)

        # Start timing the feature selection process
        start_feature_selection = time.time()

        # Perform feature selection and PCA
        c_reduced = SLM_tools.feature_selection(C)
        principal_components, score, latent = SLM_tools.pca(c_reduced, n_components)
        a_reduced = SLM_tools.data_preparation(score, c_reduced, n_components)

        # Update counters for selected features
        counters["selected_features"] = a_reduced.shape[0]
        runtime_feature_selection = round(time.time() - start_feature_selection, 3)
        counters["feature_selection"] = runtime_feature_selection
        counters["total_runtime"] += runtime_feature_selection

        # Log the feature selection time
        log9 = f"\nFinished Feature Selection runtime {runtime_feature_selection} seconds"
        counters["log"].append(log9)
        logging.info(log9)

        # Save the selected features to a .mat file
        selected_features_path = os.path.join(save_path, f"selected_features_{current_datetime}.mat")
        sio.savemat(selected_features_path, {'selected_features': a_reduced})
        log10 = f"\nSaved all selected_features"
        logging.info(log10)
        counters["log"].append(log10)

        # Log a summary of the preprocessing steps
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
        logging.info(log11)

        # Save the preprocessing log to a text file
        log_path = os.path.join(save_path, f"preprocessing_log_{current_datetime}.txt")
        with open(log_path, 'w') as f:
            for line in counters["log"]:
                f.write(line)

        # Log the start of model training
        logging.info("Training start")

        # Train the model using cross-validation
        YI, tfas_predict_mat, tfas_actually_mat, mean_error_mat, random_x = SLM_tools.model_training_with_cv(a_reduced,
                                                                                                             n_components,
                                                                                                             int(cv_num))

        # Draw the stochastic landscape in 2D
        SLM_tools.draw_stochastic_landscape_2d(random_x, save_path, n_components)

        # Log the start of the first evaluation
        logging.info("Eval 1 start")

        # Evaluate the model and generate plots
        mean_vec, std_vec, hist_space, x_hist_space, x_ticks, y_ticks = SLM_tools.model_eval(tfas_predict_mat,
                                                                                             tfas_actually_mat,
                                                                                             int(cv_num),
                                                                                             save_path)

        # Train the model again using the validation set
        predictions, labels, mean_error, model = SLM_tools.train_again_on_validation(a_reduced,
                                                                                     n_components)

        model_path = os.path.join(save_path, f"SLM_Model_{current_datetime}.mat")
        sio.savemat(model_path, {'SLM': model})
        predictions_path = os.path.join(save_path, f"Predictions_{current_datetime}.mat")
        sio.savemat(predictions_path, {'predictions': predictions})
        mean_error_path = os.path.join(save_path, f"Mean_Error_{current_datetime}.mat")
        sio.savemat(mean_error_path, {'error': mean_error})
        # Log the start of the second evaluation
        logging.info("Eval 2 start")
        # Perform cross-validation bias correction and generate plots
        SLM_tools.cv_bias_correction(predictions, labels, hist_space, mean_vec, x_hist_space,
                                     x_ticks, y_ticks, save_path)
