from functions import utils, data_loader
from functions import evaluation
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# --------------------------- #
# SET PARAMETERS
architecture = 'MC-TIRE'
generate_data = False
datasets = ["change-A2", "change-A3", "change-A4", "change-A5", "change-A6", "change-A7", "change-A8", "change-A9",
            "change-B2", "change-B3", "change-B4", "change-B5", "change-B6", "change-B7", "change-B8", "change-B9",
            "MC-CC2", "MC-CC3", "MC-CC4", "MC-CC5", "MC-CC6", "MC-CC7", "MC-CC8", "MC-CC9",
            "MC-JM2", "MC-JM3", "MC-JM4", "MC-JM5", "MC-JM6", "MC-JM7", "MC-JM8", "MC-JM9",
            "MC-SV2", "MC-SV3", "MC-SV4", "MC-SV5", "MC-SV6", "MC-SV7", "MC-SV8", "MC-SV9",
            "bee_dance0", "bee_dance1", "bee_dance2", "bee_dance3", "bee_dance4", "bee_dance5",
            "UCI_test", "hasc"]
enable_feature_plot = False
enable_eval_plot = False
enable_1st_phase = False
enable_model_summary = False
save_txt = True
training_verbose = 0
used_feature = 'all'   # A, S, B AND all
domains = ["TD", "FD", "both"]
rank = 1

# parameters TD
TD_weight_shared_AS = 1E-2    # lambda^FD in paper
TD_weight_shared_B = 1E-2
TD_weight_uncor = 0.1

# parameters FD
FD_weight_shared_AS = 1E-2   # lambda^FD in paper
FD_weight_shared_B = 1E-2
FD_weight_uncor = 0.1
nfft = 30   # number of points for DFT
norm_mode = "timeseries"   # for calculation of DFT, should the timeseries have mean zero or each window?
n_filter = 2

# --------------------------- #
# load model
path = os.path.join(os.path.abspath(os.getcwd()), "results_all")
if not os.path.exists(path):
    os.makedirs(path)

for dataset in datasets:
    # --------------------------- #
    # load data
    window_size, threshold = utils.set_windowsize_and_threshold(dataset)
    time_series, windows_TD, windows_FD, parameters = data_loader.data_parse(nfft, norm_mode, generate_data, dataset, window_size)
    nr_channels = windows_TD.shape[-1]
    for i in range(10):
        for domain in domains:
            network = utils.load_architecture(architecture)
            print(dataset+"_repeat"+str(i)+domain)
            if save_txt:
                file = open(os.path.join(path, dataset + "_" + used_feature + domain + ".txt"), "a")
                if enable_1st_phase:
                    file_P1 = open(os.path.join(path, dataset + "_" + used_feature + domain + "_P1.txt"), "a")
            else:
                file = None
                file_P1 = None

            # --------------------------- #
            # TRAIN the Model
            if domain == "TD":
                shared_AS_TD, shared_B_TD, loss_coherent, loss_incoherent = \
                    network.train_model(windows_TD, TD_weight_shared_AS, TD_weight_shared_B, TD_weight_uncor, n_filter,
                                        training_verbose, enable_model_summary, rank)
                shared_AS_FD = None
                shared_B_FD = None
            elif domain == "FD":
                shared_AS_TD = None
                shared_B_TD = None
                shared_AS_FD, shared_B_FD, loss_coherent, loss_incoherent = \
                    network.train_model(windows_FD, FD_weight_shared_AS, FD_weight_shared_B, FD_weight_uncor, n_filter,
                                        training_verbose, enable_model_summary, rank)
            else:
                shared_AS_TD, shared_B_TD, loss_coherent_TD, loss_incoherent_TD = \
                    network.train_model(windows_TD, TD_weight_shared_AS, TD_weight_shared_B, TD_weight_uncor, n_filter,
                                        training_verbose, enable_model_summary, rank)
                shared_AS_FD, shared_B_FD, loss_coherent_FD, loss_incoherent_FD = \
                    network.train_model(windows_FD, FD_weight_shared_AS, FD_weight_shared_B, FD_weight_uncor, n_filter,
                                        training_verbose, enable_model_summary, rank)
                loss_coherent = (loss_coherent_TD + loss_coherent_FD) / 2
                loss_incoherent = (loss_incoherent_TD + loss_incoherent_FD) / 2

            # --------------------------- #
            # POSTPROCESSING AND PEAK DETECTION
            dissimilarities = evaluation.smoothened_dissimilarity_measures(shared_AS_TD, shared_AS_FD, shared_B_TD,
                                                                           shared_B_FD, window_size)
            evaluation.show_result(generate_data, window_size, dissimilarities, parameters, threshold, loss_coherent,
                                   loss_incoherent, enable_eval_plot, file, time_series)
