from functions import evaluation, utils, data_loader
import numpy as np


# --------------------------- #
# SET PARAMETERS
domain = "FD"   # chtoose from: TD (time domain), FD (frequency domain) or both
architecture = 'tire_cluster'   # TIRE, AE_torch, AE_predict, TIRE_predict
generate_data = False
dataset = "P_ID16_r10"       # MEAN, VARIANCE, GAUSSIAN, AR1, AR2..., if generate_data = True
                        # mean0-9, var0-9, gauss0-9, ar0-9, hasc, bee_dance0-5, eeg_eye_state, well..., if generate_data = False
shared_weight = 1
predict_weight = 1
enable_feature_plot = False
enable_eval_plot = True

# parameters TD
intermediate_dim_TD = 140
latent_dim_TD = 14   # h^TD in paper
nr_spatial_TD = 9
nr_temporal_TD = 5
K_TD = 1    # as in paper
nr_ae_TD = K_TD+1    # number of parallel AEs = K+1
TD_weight_shared = 1    # lambda^FD in paper
TD_weight_CE = 1e-3
TD_weight_center = 0.001

# parameters FD
intermediate_dim_FD = 400   # (latent_dim*2)
latent_dim_FD = 40 # h^FD in paper (rank^2+rank)
nr_spatial_FD = 26
nr_temporal_FD = 13
K_FD = 1    # as in paper
nr_ae_FD = K_FD + 1   # number of parallel AEs = K+1
FD_weight_shared = 1   # lambda^FD in paper
FD_weight_CE = 1e-3
FD_weight_center = 0.001
nfft = 30   # number of po ints for DFT
norm_mode = "timeseries"   # for calculation of DFT, should the timeseries have mean zero or each window?


# --------------------------- #
# load data and model
window_size, threshold = utils.set_windowsize_and_threshold(dataset)
windows_TD, windows_FD, parameters = data_loader.data_parse(nfft, norm_mode, generate_data, dataset, window_size)
network = utils.load_architecture(architecture)
labels = np.arange(windows_TD.shape[0])

# --------------------------- #
# TRAIN THE AUTOENCODERS
if domain == "TD":
    shared_features_TD, features_TD = network.train_AE(windows_TD, labels, intermediate_dim_TD,
                                                       latent_dim_TD, 0, 0, nr_ae_TD, TD_weight_shared,
                                                       TD_weight_CE, nr_epochs=80)
    shared_features_FD = features_FD = None
elif domain == "FD":
    shared_features_TD = features_TD = None
    shared_features_FD, features_FD = network.train_AE(windows_FD, labels, intermediate_dim_FD,
                                                       latent_dim_FD, 0, 0, nr_ae_FD, FD_weight_shared,
                                                       FD_weight_CE, nr_epochs=80)
else:
    shared_features_TD, features_TD = network.train_AE(windows_TD, labels, intermediate_dim_TD,
                                                       latent_dim_TD, 0, 0, nr_ae_TD, TD_weight_shared,
                                                       TD_weight_CE, nr_epochs=80)
    shared_features_FD, features_FD = network.train_AE(windows_FD, labels, intermediate_dim_FD,
                                                       latent_dim_FD, 0, 0, nr_ae_FD, FD_weight_shared,
                                                       FD_weight_CE, nr_epochs=80)

# --------------------------- #
# POSTPROCESSING AND PEAK DETECTION
dissimilarities = evaluation.smoothened_dissimilarity_measures(features_TD, features_FD, window_size)
change_point_scores = evaluation.change_point_score(dissimilarities, window_size)
evaluation.show_result(generate_data, window_size, dissimilarities, parameters, threshold, enable_eval_plot)

# --------------------------- #
# Clustering
# labels = utils.clustering2(0.5, nr_classes, shared_features_TD, shared_features_FD, dis_matrix, num_array,
#                            window_size)
labels = utils.clustering3(features_TD, features_FD, window_size)
# --------------------------- #
# TRAIN THE AUTOENCODERS
if domain == "TD":
    shared_features_TD, features_TD = network.train_AE(windows_TD, labels, intermediate_dim_TD, latent_dim_TD, nr_spatial_TD,
                                             nr_temporal_TD, nr_ae_TD, TD_weight_shared, TD_weight_CE)
    shared_features_FD = None
elif domain == "FD":
    shared_features_TD = None
    shared_features_FD, features_FD = network.train_AE(windows_FD, labels, intermediate_dim_FD, latent_dim_FD, nr_spatial_FD,
                                             nr_temporal_FD, nr_ae_FD, FD_weight_shared, FD_weight_CE)
else:
    shared_features_TD, features_TD = network.train_AE(windows_TD, labels, intermediate_dim_TD, latent_dim_TD, nr_spatial_TD,
                                             nr_temporal_TD, nr_ae_TD, TD_weight_shared, TD_weight_CE)
    shared_features_FD, features_FD = network.train_AE(windows_FD, labels, intermediate_dim_FD, latent_dim_FD,nr_spatial_FD,
                                             nr_temporal_FD, nr_ae_FD, FD_weight_shared, FD_weight_CE)
# Investigating the feature space
if enable_feature_plot == True:
    utils.plot_feature_space(shared_features_TD, shared_features_FD, parameters)
np.savetxt("features_FD.txt",features_FD)
# --------------------------- #
# POSTPROCESSING AND PEAK DETECTION
dissimilarities = evaluation.smoothened_dissimilarity_measures(shared_features_TD, shared_features_FD,
                                                               window_size)
change_point_scores = evaluation.change_point_score(dissimilarities, window_size)
evaluation.show_result(generate_data, window_size, dissimilarities, parameters, threshold, enable_eval_plot)
