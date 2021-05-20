import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
import numpy as np
import matplotlib.pyplot as plt
from functions import postprocessing, evaluation
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.signal import find_peaks
from sklearn.cluster import SpectralClustering


def load_architecture(architecture_name):
    """Loads the network definition from the architectures folder

    :param architecture_name: String with the name of the network
    :return: module with the architecture
    """
    return getattr(__import__('architectures.%s' % architecture_name), '%s' % architecture_name)


def set_windowsize_and_threshold(dataset):
    if (("AR" in dataset) and not(("VAR" in dataset))) \
            or (("ar" in dataset) and not(("var" in dataset) or ("MC" in dataset))):
        windowsize = 35
        threshold = 30
    elif "bee_dance" in dataset:
        windowsize = 15
        threshold = 15
    elif "hasc" in dataset:
        windowsize = 250
        threshold = 300
    elif "HASC" in dataset:
        windowsize = 200
        threshold = 60
    elif "well" in dataset:
        windowsize = 100
        threshold = 60
    elif "eye_state" in dataset:
        windowsize = 170
        threshold = 200
    elif "UCI" in dataset:
        windowsize = 8
        threshold = 3
    elif "Santa" in dataset:
        windowsize = 90
        threshold = 60
    elif "rank" in dataset:
        windowsize = 35
        threshold = 30
    elif "BabyECG" in dataset:
        windowsize = 25
        threshold = 30
    elif "turbine" in dataset:
        windowsize = 15
        threshold = 15
    elif "P_ID" in dataset:
        windowsize = 1000
        threshold = 1000
    else:
        for keyword in ["mean", "MEAN", "var", "VAR", "gauss", "GAUSS", "ar", "MC"]:
            if keyword in dataset:
                windowsize = 30
                threshold = 30
    return windowsize, threshold


def calc_fft(windows, nfft=30, norm_mode="timeseries"):
    """
    Calculates the DFT for each window and transforms its length

    Args:
        windows: time series windows
        nfft: number of points used for the calculation of the DFT
        norm_mode: ensure that the timeseries / each window has zero mean

    Returns:
        frequency domain windows, each window having size nfft//2 (+1 for timeseries normalization)
    """
    mean_per_segment = np.mean(windows, axis=-1)
    mean_all = np.mean(mean_per_segment, axis=0)

    if norm_mode == "window":
        windows = windows - mean_per_segment[:, None]
        windows_fft = abs(np.fft.fft(windows, nfft))[..., 1:nfft // 2 + 1]
    elif norm_mode == "timeseries":
        windows = windows - mean_all
        windows_fft = abs(np.fft.fft(windows, nfft))[..., :nfft // 2 + 1]

    fft_max = np.amax(windows_fft)
    fft_min = np.amin(windows_fft)

    windows_fft = 2 * (windows_fft - fft_min) / (fft_max - fft_min) - 1

    return windows_fft


def norm_windows(windows):
    fft_max = np.amax(windows)
    fft_min = np.amin(windows)
    windows_normed = 2 * (windows - fft_min) / (fft_max - fft_min) - 1
    return windows_normed


def minmaxscale(data, a, b):
    """
    Scales data to the interval [a,b]
    """
    data_min = np.amin(data)
    data_max = np.amax(data)
    return (b-a)*(data-data_min)/(data_max-data_min)+a


def plot_feature_space(shared_features_TD, shared_features_FD, parameters):
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2)
    im = Isomap(n_components=2)
    se = SpectralEmbedding(n_components=2)

    if shared_features_TD is not None:
        f_pca = pca.fit_transform(shared_features_TD)
        f_tsne = tsne.fit_transform(shared_features_TD)
        f_im = im.fit_transform(shared_features_TD)
        f_se = se.fit_transform(shared_features_TD)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        ax1.scatter(f_pca[:, 0], f_pca[:, 1], c=parameters[:f_pca.shape[0]], s=8, alpha=0.5, cmap='gist_rainbow')
        ax1.legend()
        sns.despine(trim=True, ax=ax1)
        ax1.set_title('Deep-Learning \n PCA visualization')

        ax2.scatter(f_tsne[:, 0], f_tsne[:, 1], c=parameters[:f_tsne.shape[0]], s=8, alpha=0.5, cmap='gist_rainbow')
        ax2.legend()
        sns.despine(trim=True, ax=ax2)
        ax2.set_title('Deep-Learning \n t-SNE visualization')

        ax3.scatter(f_im[:, 0], f_im[:, 1], c=parameters[:f_im.shape[0]], s=8, alpha=0.5, cmap='gist_rainbow')
        ax3.legend()
        sns.despine(trim=True, ax=ax3)
        ax3.set_title('Deep-Learning \n Isomap visualization')

        ax4.scatter(f_se[:, 0], f_se[:, 1], c=parameters[:f_se.shape[0]], s=8, alpha=0.5, cmap='gist_rainbow')
        ax4.legend()
        sns.despine(trim=True, ax=ax4)
        ax4.set_title('Deep-Learning \n SpectralEmbedding visualization')

        fig.suptitle('Time Domian')
        plt.show()

    if shared_features_FD is not None:
        f_pca = pca.fit_transform(shared_features_FD)
        f_tsne = tsne.fit_transform(shared_features_FD)
        f_im = im.fit_transform(shared_features_FD)
        f_se = se.fit_transform(shared_features_FD)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        ax1.scatter(f_pca[:, 0], f_pca[:, 1], c=parameters[:f_pca.shape[0]], s=8, alpha=0.5, cmap='gist_rainbow')
        ax1.legend()
        sns.despine(trim=True, ax=ax1)
        ax1.set_title('Deep-Learning \n PCA visualization')

        ax2.scatter(f_tsne[:, 0], f_tsne[:, 1], c=parameters[:f_tsne.shape[0]], s=8, alpha=0.5, cmap='gist_rainbow')
        ax2.legend()
        sns.despine(trim=True, ax=ax2)
        ax2.set_title('Deep-Learning \n t-SNE visualization')

        ax3.scatter(f_im[:, 0], f_im[:, 1], c=parameters[:f_im.shape[0]], s=8, alpha=0.5, cmap='gist_rainbow')
        ax3.legend()
        sns.despine(trim=True, ax=ax3)
        ax3.set_title('Deep-Learning \n Isomap visualization')

        ax4.scatter(f_se[:, 0], f_se[:, 1], c=parameters[:f_se.shape[0]], s=8, alpha=0.5, cmap='gist_rainbow')
        ax4.legend()
        sns.despine(trim=True, ax=ax4)
        ax4.set_title('Deep-Learning \n SpectralEmbedding visualization')

        fig.suptitle('Frequency Domian')
        plt.show()

def clustering(encoded_windows=None, encoded_windows_fft=None, dis_matrix=None,
               num_array=None, window_size=20, k=20):

    # combine encoded data
    if encoded_windows_fft is None:
        encoded_windows_both = encoded_windows
    elif encoded_windows is None:
        encoded_windows_both = encoded_windows_fft
    else:
        beta = np.quantile(postprocessing.distance(encoded_windows, window_size), 0.95)
        alpha = np.quantile(postprocessing.distance(encoded_windows_fft, window_size), 0.95)
        encoded_windows_both = np.concatenate((encoded_windows * alpha, encoded_windows_fft * beta), axis=1)

    # initialize the dissimilarity array in 1st round
    if dis_matrix is None:
        dist_list = pdist(encoded_windows_both, metric='euclidean')
        dis_matrix = squareform(dist_list)
    dis_array = dis_matrix.diagonal(offset=1)

    # find the k classes with minimal distance and merge them
    for i in np.arange(k):
        merge_idx = np.where(dis_array == np.min(dis_array))[0][0]
        na = num_array[merge_idx]
        nb = num_array[merge_idx+1]
        # update dis_matrix
        for s in np.arange(num_array.size):
            dis_matrix[s,merge_idx] = (na/(na+nb))*dis_matrix[s,merge_idx]+(nb/(na+nb))*dis_matrix[s,merge_idx+1]
            dis_matrix[merge_idx,s] = (na/(na+nb))*dis_matrix[merge_idx,s]+(nb/(na+nb))*dis_matrix[merge_idx+1,s]
        dis_matrix = np.delete(dis_matrix, merge_idx+1, axis=0)
        dis_matrix = np.delete(dis_matrix, merge_idx+1, axis=1)
        dis_array = dis_matrix.diagonal(offset=1)
        # update num_array
        num_array[merge_idx] = na + nb
        num_array = np.delete(num_array, merge_idx+1)

    labels = np.array([])
    for idx, num in enumerate(num_array):
        seg = idx*np.ones(num)
        labels = np.append(labels, seg)

    return labels.astype(int), dis_matrix, num_array

def clustering2(lamb, nr_classes, encoded_windows=None, encoded_windows_fft=None, dis_matrix=None,
                num_array=None, window_size=20):
    # get number of clusters
    dissimilarities = evaluation.smoothened_dissimilarity_measures(encoded_windows, encoded_windows_fft, window_size)
    dissimilarities = np.concatenate((np.zeros((window_size,)), dissimilarities, np.zeros((window_size,))))
    peaks = find_peaks(dissimilarities)[0]
    k = peaks.size
    quantile = nr_classes // 10
    # combine encoded data
    if encoded_windows_fft is None:
        encoded_windows_both = encoded_windows
    elif encoded_windows is None:
        encoded_windows_both = encoded_windows_fft
    else:
        beta = np.quantile(postprocessing.distance(encoded_windows, window_size), 0.95)
        alpha = np.quantile(postprocessing.distance(encoded_windows_fft, window_size), 0.95)
        encoded_windows_both = np.concatenate((encoded_windows * alpha, encoded_windows_fft * beta), axis=1)

    # initialize the dissimilarity array in 1st round
    if dis_matrix is None:
        if encoded_windows_both.shape[-1] == 1:
            dist_list = pdist(encoded_windows_both, metric='euclidean')
        else:
            dist_list = pdist(encoded_windows_both, metric='correlation')
        dis_matrix = squareform(dist_list)
        dis_array = dis_matrix.diagonal(offset=1)
        dispersion_array = np.zeros_like(dis_matrix.diagonal(offset=0))
    last_part = None
    while nr_classes > k:
        current_part = nr_classes // quantile
        if current_part != last_part:
            if last_part is not None:
                print('finish!')
            print('begin to process part {}/10'.format(10-current_part))
        # find the k classes with minimal distance and merge them
        merge_idx = np.where(dis_array == np.min(dis_array))[0][0]
        na = num_array[merge_idx]
        nb = num_array[merge_idx+1]
        # update dis_matrix
        for s in np.arange(num_array.size):
            dis_matrix[s,merge_idx] = (na/(na+nb))*dis_matrix[s,merge_idx]+(nb/(na+nb))*dis_matrix[s,merge_idx+1]
            dis_matrix[merge_idx,s] = (na/(na+nb))*dis_matrix[merge_idx,s]+(nb/(na+nb))*dis_matrix[merge_idx+1,s]
        dis_matrix = np.delete(dis_matrix, merge_idx+1, axis=0)
        dis_matrix = np.delete(dis_matrix, merge_idx+1, axis=1)
        # update the dispersion_array
        new_dispersion = ((na * dispersion_array[merge_idx]) + (nb * dispersion_array[merge_idx+1]) + \
                          (na * nb * dis_array[merge_idx])) / (na + nb + na * nb)
        dispersion_array[merge_idx] = new_dispersion
        dispersion_array = np.delete(dispersion_array, merge_idx+1)
        dispersions = dispersion_array[0:-1] + dispersion_array[1:]
        dis_array = dis_matrix.diagonal(offset=1) + lamb * dispersions
        # update num_array
        num_array[merge_idx] = na + nb
        num_array = np.delete(num_array, merge_idx+1)
        nr_classes = num_array.size
        last_part = current_part
    labels = np.array([])
    # labels = np.zeros([window_size])
    for idx, num in enumerate(num_array):
        seg = idx*np.ones(num)
        labels = np.append(labels, seg)
    # labels = labels[:-window_size]
    return labels.astype(int)

def clustering3(encoded_windows=None, encoded_windows_fft=None, window_size=20):
    dissimilarities = evaluation.smoothened_dissimilarity_measures(encoded_windows, encoded_windows_fft, window_size)
    dissimilarities = np.concatenate((np.zeros((window_size,)), dissimilarities, np.zeros((window_size,))))
    peaks = find_peaks(dissimilarities)[0]
    k = peaks.size

    # combine encoded data
    if encoded_windows_fft is None:
        encoded_windows_both = encoded_windows
    elif encoded_windows is None:
        encoded_windows_both = encoded_windows_fft
    else:
        beta = np.quantile(postprocessing.distance(encoded_windows, window_size), 0.95)
        alpha = np.quantile(postprocessing.distance(encoded_windows_fft, window_size), 0.95)
        encoded_windows_both = np.concatenate((encoded_windows * alpha, encoded_windows_fft * beta), axis=1)

    order = np.expand_dims(np.arange(encoded_windows_both.shape[0]), 1)
    encoded_windows_both = np.concatenate((order, encoded_windows_both), axis=1)
    clustering = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels="discretize", n_jobs=-1).fit(encoded_windows_both)
    labels = clustering.labels_
    return labels