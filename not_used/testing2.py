import os
import re


def get_file_paths(folder, file_dict):
    pattern = re.compile(
        r"(?P<type>distance|energy)_vec_mu_(?P<mu>[-+]?\d*\.\d+|\d+)_energy_(?P<energy>[-+]?\d*\.\d+|\d+)_run_num_(?P<run_num>\d+)_total_num_target_\d+\.mat")
    distance_dict = {}
    energy_dict = {}
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and filename.endswith('.mat'):
            match = pattern.match(filename)
            if match:
                file_info = match.groupdict()
                key = (float(file_info['mu']), int(file_info['run_num']))
                if file_info['type'] == 'distance':
                    distance_dict[key] = filename
                elif file_info['type'] == 'energy':
                    energy_dict[key] = filename
    distance_dict = {k: v for k, v in distance_dict.items() if k[0] == 0.6}
    energy_dict = {k: v for k, v in energy_dict.items() if k[0] == 0.6}
    for i, (energy_path, distance_path) in enumerate(zip(energy_dict.values(), distance_dict.values()), start=len(file_dict)):
      file_dict[i] = (os.path.join(folder, energy_path), os.path.join(folder, distance_path))
    return file_dict


if __name__ == '__main__':
    file_dict = {}
    path = r"C:\Users\User\OneDrive - mail.tau.ac.il\Documents\Final Project\Project2-Omri and Idan\Results\06_30_2022_10_12_02_mu_0__0_2__2_8_Js__4_0_runs_2\2\3"
    file_dict = get_file_paths(path, file_dict)