import numpy as np
from scipy import interpolate
import scipy.io as sio
import Rbeast as rb
from SLM_tools import *
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    energy = sio.loadmat(
        r"C:\Users\User\OneDrive - mail.tau.ac.il\Documents\Final Project\Project2-Omri and Idan\Results\06_30_2022_10_12_02_mu_0__0_2__2_8_Js__4_0_runs_2\2\3\energy_vec_mu_1.6_energy_4.0_run_num_1_total_num_target_2.mat")
    energy = energy["foo"]
    data, time = SLM_tools.interpolate_data_over_regular_time(energy)
    data, time = SLM_tools.downsample(data, time, 10)
    data, time = SLM_tools.downsample(data, time, 10)
    o = rb.beast(data, 0, tseg_minlength=0.1 * data.shape[1], season="none", torder_minmax=[1, 1.01])
    rb.print(o)
    rb.plot(o)
