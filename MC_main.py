from functions import utils, data_loader
from functions import evaluation
import os
import warnings
warnings.filterwarnings("ignore")

# --------------------------- #
# SET PARAMETERS
architecture = 'MC-TIRE'   # TIRE, AE_torch, AE_predict, TIRE_predict
generate_data = False
dataset = "change-A8"                  # mean0-9, var0-9, gauss0-9, ar0-9, hasc, bee_dance0-5, eeg_eye_state, well..., if generate_data = False
enable_feature_plot = False
enable_eval_plot = True
enable_1st_phase = False
enable_model_summary = False
save_txt = False
training_verbose = 0
used_feature = 'all'   # A, S, B AND all
domain = "FD"
rank = 1

# parameters TD
TD_weight_shared_AS = 1e-1    # lambda^FD in paper
TD_weight_shared_B = 1E-3
TD_weight_uncor = 0.1

# parameters FD
FD_weight_shared_AS = 1e-1   # lambda^FD in paper
FD_weight_shared_B = 1E-3
FD_weight_uncor = 0.1
nfft = 30   # number of points for DFT
norm_mode = "timeseries"   # for calculation of DFT, should the timeseries have mean zero or each window?
n_filter = 2

# --------------------------- #
# BEGINNING OF CODE
# load data and model
window_size, threshold = utils.set_windowsize_and_threshold(dataset)
time_series, windows_TD, windows_FD, parameters = data_loader.data_parse(nfft, norm_mode, generate_data, dataset, window_size)
network = utils.load_architecture(architecture)
path = os.path.join(os.path.abspath(os.getcwd()), "results")
nr_channels = windows_TD.shape[-1]

# TRAIN THE Model
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

# Investigating the feature space
# if enable_feature_plot == True:
#     utils.plot_feature_space(shared_features_TD, shared_features_FD, parameters)

# --------------------------- #
# POSTPROCESSING AND PEAK DETECTION
dissimilarities = evaluation.smoothened_dissimilarity_measures(shared_AS_TD, shared_AS_FD, shared_B_TD,
                                                               shared_B_FD, window_size)
evaluation.show_result(generate_data, window_size, dissimilarities, parameters, threshold, loss_coherent,
                       loss_incoherent, enable_eval_plot, None, time_series)