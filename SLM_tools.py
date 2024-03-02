import Rbeast as rb
import numpy as np
from scipy import interpolate


class SLM_tools:
    @staticmethod
    def creat_time_vec(data):
        t = np.arange(0, len(data), 1)
        data_new = interpolate.interp1d(t, data, kind="linear")
        return data_new


if __name__ == "__main__":
    print('hello world')
