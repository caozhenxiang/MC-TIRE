from functions import evaluation, utils, data_loader
import numpy as np
import os


# settings
use_origin_TIRE = False
generate_data = False
datasets = ["hasc"]
# "bee_dance0", "bee_dance1", "bee_dance2", "bee_dance3", "bee_dance4","bee_dance5", "hasc"
# "UCI_TEST"
# "multi_var2", "multi_var4", "multi_var6", "multi_var8",
#             "multi_ar2", "multi_ar4", "multi_ar6", "multi_ar8",
#             "multi_mean2", "multi_mean4", "multi_mean6", "multi_mean8",
#             "multi_gauss2", "multi_gauss4", "multi_gauss6", "multi_gauss8"

nfft = 30
norm_mode = "timeseries"
architecture = "tire_cluster"
domains = ['TD', 'both']
enable_feature_plot = False
enable_eval_plot = False

# parameters TD
K_TD = 1    # as in paper
nr_ae_TD = K_TD+1    # number of parallel AEs = K+1
TD_weight_shared = 1    # lambda^FD in paper
TD_weight_CE = 1e-3
TD_weight_center = 0.001

# parameters FD
K_FD = 1    # as in paper
nr_ae_FD = K_FD+1   # number of parallel AEs = K+1
FD_weight_shared = 1   # lambda^FD in paper
FD_weight_CE = 1e-3
FD_weight_center = 0.001
nfft = 30    # number of points for DFT
norm_mode = "timeseries"    # for calculation of DFT, should the timeseries have mean zero or each window?


for dataset in datasets:
    if use_origin_TIRE:
        nr_spatial_FD = 0
        nr_temporal_FD = 1
        latent_dim_FD = 1
        intermediate_dim_FD = 10
        nr_spatial_TD = 0
        nr_temporal_TD = 1
        latent_dim_TD = 1
        intermediate_dim_TD = 10
    elif dataset == 'BabyECG':
        nr_spatial_FD = 0
        nr_temporal_FD = 1
        latent_dim_FD = 2
        intermediate_dim_FD = 20
        nr_spatial_TD = 0
        nr_temporal_TD = 1
        latent_dim_TD = 2
        intermediate_dim_TD = 20
    else:
        # rank = int(dataset[-1])
        rank = 3
        nr_spatial_FD = rank
        nr_temporal_FD = rank//2
        latent_dim_FD = nr_spatial_FD + nr_temporal_FD + 1
        intermediate_dim_FD = 10 * latent_dim_FD
        nr_spatial_TD = rank
        nr_temporal_TD = rank // 2
        latent_dim_TD = nr_spatial_TD + nr_temporal_TD + 1
        intermediate_dim_TD = 10 * latent_dim_TD

    # load data
    window_size, threshold = utils.set_windowsize_and_threshold(dataset)
    windows_TD, windows_FD, parameters = data_loader.data_parse(nfft, norm_mode, generate_data, dataset, window_size)
    # load model and training
    network = utils.load_architecture(architecture)
    for domain in domains:
        for i in range(5):
            labels = np.arange(windows_TD.shape[0])
            # labels = parameters
            # dis_matrix = None
            # num_array = np.ones_like(labels)
            # idx_array = np.zeros_like(labels)
            # nr_classes = labels.size
            nr_shared_TD = nr_temporal_TD + nr_spatial_TD
            nr_shared_FD = nr_temporal_FD + nr_spatial_FD
            path = os.path.join(os.path.abspath(os.getcwd()), "result")
            if use_origin_TIRE:
                f1 = open(os.path.join(path, dataset + "_" + domain + "_r1.txt"), "a")
                f2 = open(os.path.join(path, dataset + "_" + domain + "_r2.txt"), "a")
            elif architecture == "TIRE":
                f1 = open(os.path.join(path, dataset + "_" + domain + "_1.txt"), "a")
                f2 = open(os.path.join(path, dataset + "_" + domain + "_2.txt"), "a")
            else:
                f1 = open(os.path.join(path, dataset + "_" + domain + "_c1.txt"), "a")
                f2 = open(os.path.join(path, dataset + "_" + domain + "_c2.txt"), "a")
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
            evaluation.show_result(generate_data, window_size, dissimilarities, parameters, threshold,
                                   enable_eval_plot, f1)

            # --------------------------- #
            # Clustering
            # labels = utils.clustering2(0.5, nr_classes, shared_features_TD, shared_features_FD, dis_matrix, num_array,
            #                            window_size)
            labels = utils.clustering3(features_TD, features_FD, window_size)
            # --------------------------- #
            # TRAIN THE AUTOENCODERS
            if domain == "TD":
                shared_features_TD, _ = network.train_AE(windows_TD, labels, intermediate_dim_TD, latent_dim_TD,
                                                         nr_spatial_TD,
                                                         nr_temporal_TD, nr_ae_TD, TD_weight_shared, TD_weight_CE)
                shared_features_FD = None
            elif domain == "FD":
                shared_features_TD = None
                shared_features_FD, _ = network.train_AE(windows_FD, labels, intermediate_dim_FD, latent_dim_FD,
                                                      nr_spatial_FD,
                                                      nr_temporal_FD, nr_ae_FD, FD_weight_shared, FD_weight_CE)
            else:
                shared_features_TD, _ = network.train_AE(windows_TD, labels, intermediate_dim_TD, latent_dim_TD,
                                                      nr_spatial_TD,
                                                      nr_temporal_TD, nr_ae_TD, TD_weight_shared, TD_weight_CE)
                shared_features_FD, _ = network.train_AE(windows_FD, labels, intermediate_dim_FD, latent_dim_FD,
                                                      nr_spatial_FD,
                                                      nr_temporal_FD, nr_ae_FD, FD_weight_shared, FD_weight_CE)
            # Investigating the feature space
            if enable_feature_plot == True:
                utils.plot_feature_space(shared_features_TD, shared_features_FD, parameters)

            # --------------------------- #
            # POSTPROCESSING AND PEAK DETECTION
            dissimilarities = evaluation.smoothened_dissimilarity_measures(shared_features_TD, shared_features_FD,
                                                                           window_size)
            change_point_scores = evaluation.change_point_score(dissimilarities, window_size)
            evaluation.show_result(generate_data, window_size, dissimilarities, parameters, threshold,
                                   enable_eval_plot, f2)