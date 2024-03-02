import Rbeast as rb
import numpy as np
from scipy import interpolate


class SLM_tools:
    @staticmethod
    def creat_time_vec(data: np.array, sample_rate: int = 1):

        """interpolate data over a regularly spaced time vector
            inputs:
                data (np.array): time series data
                sample_rate (int): optional
            outputs:
                time_vec (np.array): regularly spaced time vector
                data_new (np.array): interpolated data over the regularly spaced time vector"""
        try:
            t = np.arange(0, np.shape(data)[1], 1/sample_rate)
            # t = np.reshape(t, np.shape(data))
            data_new = interpolate.interp1d(t, data, kind="linear")
            return data_new.y, t
        except Exception as e:
            print(e)


if __name__ == "__main__":
    print('hello world')
