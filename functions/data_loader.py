from functions import preprocessing, simulate, utils
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
    else:
        path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
        if 'bee_dance' in dataset:
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
            windows_TD = np.transpose(np.array(windows_TD_bu), [1, 2, 0])
            windows_FD = np.transpose(np.array(windows_FD_bu), [1, 2, 0])
            return data, windows_TD, windows_FD, parameters

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
            windows_TD = np.transpose(np.array(windows_TD_bu), [1, 2, 0])
            windows_FD = np.transpose(np.array(windows_FD_bu), [1, 2, 0])
            return data, windows_TD, windows_FD, parameters

        elif ("change" in dataset):
            folder = dataset[:-1]
            data = np.genfromtxt(path + "/Data/used_data/" + folder + "/" + dataset + "_data.txt", delimiter=" ")
            parameters = np.genfromtxt(path + "/Data/used_data/" + folder + "/" + dataset + "_labels.txt", delimiter=" ")
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
            # windows_TD_bu = utils.norm_windows(windows_TD_bu)
            windows_TD = np.transpose(np.array(windows_TD_bu), [1, 2, 0])
            windows_FD = np.transpose(np.array(windows_FD_bu), [1, 2, 0])
            return data, windows_TD, windows_FD, parameters

        elif ("MC" in dataset):
            file_name = dataset.split('_')[-1]
            data_name = file_name + "_data.txt"
            label_name = file_name + "_labels.txt"
            data = np.genfromtxt(path + "/Data/used_data/" + dataset[:-1] + "/" + data_name, delimiter=" ")
            parameters = np.genfromtxt(path + "/Data/used_data/" + dataset[:-1] + "/" + label_name, delimiter=" ")
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
            # windows_TD_bu = utils.norm_windows(windows_TD_bu)
            windows_TD = np.transpose(np.array(windows_TD_bu), [1, 2, 0])
            windows_FD = np.transpose(np.array(windows_FD_bu), [1, 2, 0])
            return data, windows_TD, windows_FD, parameters

        elif "UCI" in dataset:
            data = np.genfromtxt(path + "/Data/used_data/" + dataset + ".csv", skip_header=1, delimiter=",")[:, 1:51]
            parameters = np.genfromtxt(path + "/Data/used_data/" + dataset + ".csv", skip_header=1,
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
            windows_TD = np.transpose(np.array(windows_TD_bu), [1, 2, 0])
            windows_FD = np.transpose(np.array(windows_FD_bu), [1, 2, 0])
            return data, windows_TD, windows_FD, parameters

        else:  # TODO: extend to further dataset
            return 0, 0, 0