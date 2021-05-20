from functions import preprocessing, simulate, utils
from mne.io import read_raw_edf
import pandas as pd
import numpy as np
import os


def data_parse(nfft, norm_mode, generate_data, dataset, window_size):
    # use simulsted data
    if generate_data:
        if dataset == 'MEAN':
            timeseries, windows_TD, parameters = simulate.generate_jumpingmean(window_size)
            windows_FD = utils.calc_fft(windows_TD, nfft, norm_mode)
            return windows_TD, windows_FD, parameters
        elif dataset == 'VAR':
            timeseries, windows_TD, parameters = simulate.generate_scalingvariance(window_size)
            windows_FD = utils.calc_fft(windows_TD, nfft, norm_mode)
            return windows_TD, windows_FD, parameters
        elif dataset == 'GAUSS':
            timeseries, windows_TD, parameters = simulate.generate_gaussian(window_size)
            windows_FD = utils.calc_fft(windows_TD, nfft, norm_mode)
            return windows_TD, windows_FD, parameters
        elif dataset == 'AR1':
            timeseries, windows_TD, parameters = simulate.generate_changingcoefficients(window_size, p=1)
            windows_FD = utils.calc_fft(windows_TD, nfft, norm_mode)
            return windows_TD, windows_FD, parameters
        elif dataset == 'AR2':
            timeseries, windows_TD, parameters = simulate.generate_changingcoefficients(window_size, p=2)
            windows_FD = utils.calc_fft(windows_TD, nfft, norm_mode)
            return windows_TD, windows_FD, parameters
        elif dataset == 'AR5':
            timeseries, windows_TD, parameters = simulate.generate_changingcoefficients_5(window_size)
            windows_FD = utils.calc_fft(windows_TD, nfft, norm_mode)
            return windows_TD, windows_FD, parameters
    # load dataset
    # note: loaded data can be preprocessed using utils.ts_to_windows and utils.combine_ts
    else:
        path = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
        if dataset == 'eeg_eye_state':
            data = np.genfromtxt(path + "/Data/EEG Eye State.csv", delimiter=",")
            parameters = data[:, -1]
            for idx in range(np.shape(data)[1] - 1):
            #for idx in [0]:
                windows_td = preprocessing.ts_to_windows(data[:, idx], 0, window_size, 1, normalization="timeseries")
                windows_fd = utils.calc_fft(windows_td, norm_mode=norm_mode)
                windows_td = utils.norm_windows(windows_td)
                if idx == 0:
                    windows_TD_bu = [windows_td]
                    windows_FD_bu = [windows_fd]
                else:
                    windows_TD_bu.append(windows_td)
                    windows_FD_bu.append(windows_fd)
            windows_TD = preprocessing.combine_ts(windows_TD_bu)
            windows_FD = preprocessing.combine_ts(windows_FD_bu)
            return windows_TD, windows_FD, parameters

        elif 'HASC' in dataset:
            data = np.genfromtxt(path + "/Data/HASC/" + dataset + ".csv", delimiter=",")[:, 1:]
            parameters = np.genfromtxt(path + "/Data/HASC/" + dataset + "_label.txt", delimiter=" ")
            for idx in range(np.shape(data)[1]):
                windows_td = preprocessing.ts_to_windows(data[:, idx], 0, window_size, 1, normalization="timeseries")
                windows_fd = utils.calc_fft(windows_td, norm_mode=norm_mode)
                windows_td = utils.norm_windows(windows_td)
                if idx == 0:
                    windows_TD_bu = [windows_td]
                    windows_FD_bu = [windows_fd]
                else:
                    windows_TD_bu.append(windows_td)
                    windows_FD_bu.append(windows_fd)
            windows_TD = preprocessing.combine_ts(windows_TD_bu)
            windows_FD = preprocessing.combine_ts(windows_FD_bu)
            return windows_TD, windows_FD, parameters

        elif "P_ID" in dataset:
            file_name = dataset.split('_')[:-1]

            file_path = os.path.join(path, "Data", "SeizeIT1", '_'.join(file_name))
            # read EDF
            raw = read_raw_edf(os.path.join(file_path, dataset+'.edf'), preload=False)
            data = raw.get_data()
            parameters = np.zeros(data.shape[1])
            info = raw.info
            freq = info["sfreq"]

            # read tsv
            tsv_file = os.path.join(file_path, dataset+'_a1.tsv')
            anno = pd.read_csv(tsv_file, sep="\t", skiprows=9, header=None)
            anno = anno.iloc[0, 0:2].to_numpy(dtype='int')

            parameters[int(anno[0] * freq):int(anno[1] * freq)] = 1

            if anno[0] > 25:
                start_point = int((anno[0] - 25) * freq)
            else:
                start_point = 0

            if anno[1] < int(raw.times[-1]) - 25:
                end_point = int((anno[1] + 25) * freq)
            else:
                end_point = int(np.max(raw["times"]) * freq)

            data = np.transpose(data[:, start_point:end_point])
            parameters = parameters[start_point:end_point]
            for idx in range(np.shape(data)[1]):
                windows_td = preprocessing.ts_to_windows(data[:, idx], 0, window_size, 1, normalization="timeseries")
                windows_fd = utils.calc_fft(windows_td, norm_mode=norm_mode)
                windows_td = utils.norm_windows(windows_td)
                if idx == 0:
                    windows_TD_bu = [windows_td]
                    windows_FD_bu = [windows_fd]
                else:
                    windows_TD_bu.append(windows_td)
                    windows_FD_bu.append(windows_fd)
            windows_TD = preprocessing.combine_ts(windows_TD_bu)
            windows_FD = preprocessing.combine_ts(windows_FD_bu)
            return windows_TD, windows_FD, parameters

        elif dataset == 'Wind_turbine':
            data = np.genfromtxt(path + "/Data/R80711.csv", delimiter=",")[2000:5000, 3:53]
            parameters = np.genfromtxt(path + "/Data/R80711.csv", delimiter=",")[2000:5000, -1]
            for idx in range(np.shape(data)[1]):
            #for idx in [0]:
                windows_td = preprocessing.ts_to_windows(data[:, idx], 0, window_size, 1, normalization="timeseries")
                windows_fd = utils.calc_fft(windows_td, norm_mode=norm_mode)
                windows_td = utils.norm_windows(windows_td)
                if idx == 0:
                    windows_TD_bu = [windows_td]
                    windows_FD_bu = [windows_fd]
                else:
                    windows_TD_bu.append(windows_td)
                    windows_FD_bu.append(windows_fd)
            windows_TD = preprocessing.combine_ts(windows_TD_bu)
            windows_FD = preprocessing.combine_ts(windows_FD_bu)
            return windows_TD, windows_FD, parameters

        elif 'bee_dance' in dataset:
            data_name = "bee" + dataset[-1] + "_data.txt"
            label_name = "bee" + dataset[-1] + "_labels.txt"
            data = np.genfromtxt(path + "/Data/used_data/bee/" + data_name, delimiter=" ")
            parameters = np.genfromtxt(path + "/Data/used_data/bee/" + label_name, delimiter=" ")
            for idx in range(np.shape(data)[1]):
                windows_td = preprocessing.ts_to_windows(data[:, idx], 0, window_size, 1, normalization="timeseries")
                windows_fd = utils.calc_fft(windows_td, norm_mode=norm_mode)
                windows_td = utils.norm_windows(windows_td)
                if idx == 0:
                    windows_TD_bu = [windows_td]
                    windows_FD_bu = [windows_fd]
                else:
                    windows_TD_bu.append(windows_td)
                    windows_FD_bu.append(windows_fd)
            windows_TD = preprocessing.combine_ts(windows_TD_bu)
            windows_FD = preprocessing.combine_ts(windows_FD_bu)
            return windows_TD, windows_FD, parameters

        elif dataset == "hasc":
            data = np.genfromtxt(path + "/Data/used_data/" + dataset + "/" + dataset + "0_data.txt", delimiter="\t")
            parameters = np.genfromtxt(path + "/Data/used_data/" + dataset + "/" + dataset + "0_labels.txt",
                                       delimiter=" ")
            parameters = parameters - np.ones_like(parameters)
            for idx in range(np.shape(data)[1]):
                windows_td = preprocessing.ts_to_windows(data[:, idx], 0, window_size, 1, normalization="timeseries")
                windows_fd = utils.calc_fft(windows_td, norm_mode=norm_mode)
                windows_td = utils.norm_windows(windows_td)
                if idx == 0:
                    windows_TD_bu = [windows_td]
                    windows_FD_bu = [windows_fd]
                else:
                    windows_TD_bu.append(windows_td)
                    windows_FD_bu.append(windows_fd)
            windows_TD = preprocessing.combine_ts(windows_TD_bu)
            windows_FD = preprocessing.combine_ts(windows_FD_bu)
            return windows_TD, windows_FD, parameters

        elif dataset == "well":
            data = np.genfromtxt(path + "/Data/used_data/" + dataset + "/" + dataset + "0_data.txt", delimiter=" ")
            parameters = np.genfromtxt(path + "/Data/used_data/" + dataset + "/" + dataset + "0_labels.txt",
                                       delimiter=" ")
            windows_TD = preprocessing.ts_to_windows(data, 0, window_size, 1, normalization="timeseries")
            windows_FD = utils.calc_fft(windows_TD, norm_mode=norm_mode)
            windows_TD = utils.norm_windows(windows_TD)
            return windows_TD, windows_FD, parameters

        elif ("rank" in dataset):
            data = np.genfromtxt(path + "/Data/used_data/rank/rank20_data.txt", delimiter=" ")
            parameters = np.genfromtxt(path + "/Data/used_data/rank/rank20_labels.txt", delimiter=" ")
            nr_channels = int(dataset.split('_')[-1])
            data = data[:, :nr_channels]
            for idx in range(np.shape(data)[1]):
                windows_td = preprocessing.ts_to_windows(data[:, idx], 0, window_size, 1,
                                                         normalization="timeseries")
                windows_fd = utils.calc_fft(windows_td, norm_mode=norm_mode)
                windows_td = utils.norm_windows(windows_td)
                if idx == 0:
                    windows_TD_bu = [windows_td]
                    windows_FD_bu = [windows_fd]
                else:
                    windows_TD_bu.append(windows_td)
                    windows_FD_bu.append(windows_fd)
            windows_TD = preprocessing.combine_ts(windows_TD_bu)
            windows_FD = preprocessing.combine_ts(windows_FD_bu)
            return windows_TD, windows_FD, parameters

        elif ("MC" in dataset):
            file_name = dataset.split('_')[-1]
            data_name = file_name + "_data.txt"
            label_name = file_name + "_labels.txt"
            data = np.genfromtxt(path + "/Data/used_data/" + dataset[:-1] + "/" + data_name, delimiter=" ")
            parameters = np.genfromtxt(path + "/Data/used_data/" + dataset[:-1] + "/" + label_name, delimiter=" ")
            # data = data[:1000,:]
            # parameters = parameters[:1000]
            for idx in range(np.shape(data)[1]):
                windows_td = preprocessing.ts_to_windows(data[:, idx], 0, window_size, 1,
                                                         normalization="timeseries")
                windows_fd = utils.calc_fft(windows_td, norm_mode=norm_mode)
                windows_td = utils.norm_windows(windows_td)
                if idx == 0:
                    windows_TD_bu = [windows_td]
                    windows_FD_bu = [windows_fd]
                else:
                    windows_TD_bu.append(windows_td)
                    windows_FD_bu.append(windows_fd)
            windows_TD = preprocessing.combine_ts(windows_TD_bu)
            windows_FD = preprocessing.combine_ts(windows_FD_bu)
            return windows_TD, windows_FD, parameters

        elif ("mean" in dataset) or ("var" in dataset) or ("gauss" in dataset) or ("ar" in dataset):
            data_name = dataset + "_data.txt"
            label_name = dataset + "_labels.txt"
            data = np.genfromtxt(path + "/Data/used_data/" + dataset[:-1] + "/" + data_name, delimiter=" ")
            parameters = np.genfromtxt(path + "/Data/used_data/" + dataset[:-1] + "/" + label_name, delimiter=" ")
            windows_TD = preprocessing.ts_to_windows(data, 0, window_size, 1, normalization="timeseries")
            windows_FD = utils.calc_fft(windows_TD, norm_mode=norm_mode)
            windows_TD = utils.norm_windows(windows_TD)
            return windows_TD, windows_FD, parameters

        elif "BabyECG" in dataset:
            data = np.genfromtxt(path + "/Data/BabyECG/BabyECG.csv", skip_header=1, delimiter=",")[:, -1]
            parameters = np.genfromtxt(path + "/Data/BabyECG/BabySS.csv", skip_header=1, delimiter=",")[:, -1]
            windows_TD = preprocessing.ts_to_windows(data, 0, window_size, 1, normalization="timeseries")
            windows_FD = utils.calc_fft(windows_TD, norm_mode=norm_mode)
            windows_TD = utils.norm_windows(windows_TD)
            return windows_TD, windows_FD, parameters

        elif "UCI" in dataset:
            data = np.genfromtxt(path + "/Data/UCI HAR Dataset/" + dataset + ".csv", skip_header=1, delimiter=",")[:,
                   1:51]
            parameters = np.genfromtxt(path + "/Data/UCI HAR Dataset/" + dataset + ".csv", skip_header=1,
                                       delimiter=",")[:,-1]
            for idx in range(np.shape(data)[1]):
                # for idx in [0]:
                windows_td = preprocessing.ts_to_windows(data[1:, idx], 0, window_size, 1, normalization="timeseries")
                windows_fd = utils.calc_fft(windows_td, nfft, norm_mode=norm_mode)
                windows_td = utils.norm_windows(windows_td)
                if idx == 0:
                    windows_TD_bu = [windows_td]
                    windows_FD_bu = [windows_fd]
                else:
                    windows_TD_bu.append(windows_td)
                    windows_FD_bu.append(windows_fd)
            windows_TD = preprocessing.combine_ts(windows_TD_bu)
            windows_FD = preprocessing.combine_ts(windows_FD_bu)
            return windows_TD, windows_FD, parameters

        else:  # TODO: extend to further dataset
            return 0, 0, 0