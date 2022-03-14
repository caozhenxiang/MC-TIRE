import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
import numpy as np
import matplotlib.pyplot as plt
from functions import postprocessing, evaluation
from scipy.spatial.distance import cdist
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
    if (("AR" in dataset) and not("VAR" in dataset)) \
            or (("ar" in dataset) and not("var" in dataset)) or ("MC" in dataset):
        windowsize = 40
        threshold = 40
    elif "bee_dance" in dataset:
        windowsize = 16
        threshold = 15
    elif "hasc" in dataset:
        windowsize = 280
        threshold = 300
    elif "HASC" in dataset:
        windowsize = 200
        threshold = 60
    elif "well" in dataset:
        windowsize = 100
        threshold = 60
    elif "eye_state" in dataset:
        windowsize = 200
        threshold = 200
    elif "UCI" in dataset:
        windowsize = 8
        threshold = 4
    elif "Santa" in dataset:
        windowsize = 80
        threshold = 60
    elif "change" in dataset:
        windowsize = 40
        threshold = 40
    elif "BabyECG" in dataset:
        windowsize = 24
        threshold = 30
    elif "turbine" in dataset:
        windowsize = 8
        threshold = 15
    elif "P_ID" in dataset:
        windowsize = 1000
        threshold = 1000
    else:
        for keyword in ["mean", "MEAN", "var", "VAR", "gauss", "GAUSS", "ar"]:
            if keyword in dataset:
                windowsize = 32
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


def clustering_phase1(encoded_windows, encoded_windows_fft, window_size, nr_f):
    dissimilarities = evaluation.smoothened_dissimilarity_measures(encoded_windows, encoded_windows_fft, window_size,
                                                                   nr_f, None)
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


def clustering_phase2(encoded_windows_both, selected_peaks_AS, selected_peaks_B, peaks_both, window_size, dis_AS, dis_B):
    # get the real location
    selected_peaks_AS = np.array(selected_peaks_AS) + window_size
    selected_peaks_B = np.array(selected_peaks_B) + window_size
    peaks_both = np.array(peaks_both) + window_size

    nr_AS = selected_peaks_AS.shape[0]
    nr_B = selected_peaks_B.shape[0]
    nr_anchor = np.minimum(nr_AS, nr_B)
    if nr_AS == 0:
        for peak in peaks_both:
            selected_peaks_B = np.append(selected_peaks_B, peak)
    elif nr_B == 0:
        for peak in peaks_both:
            selected_peaks_AS = np.append(selected_peaks_AS, peak)
    else:
        anchor_AS = np.argsort(dis_AS)[-nr_anchor:]
        anchor_B = np.argsort(dis_B)[-nr_anchor:]
        features_AS = encoded_windows_both[selected_peaks_AS[anchor_AS], :]
        features_B = encoded_windows_both[selected_peaks_B[anchor_B], :]

        # get features of peaks
        features_both = encoded_windows_both[np.array(peaks_both), :]

        # compute distances & assign labels
        dis_AS = cdist(features_both, features_AS, 'euclidean')
        dis_B = cdist(features_both, features_B, 'euclidean')
        dis = np.concatenate((dis_AS, dis_B), axis=-1)
        labels = []
        for element in dis:
            indices = element.argsort()[:2 * nr_anchor - 1]
            indices = np.where(indices < nr_anchor, 0, 1)
            mean = np.mean(indices)
            if mean < 0.5:
                labels.append("A")
            else:
                labels.append("B")

        # re-assign types of changes
        for idx, label in enumerate(labels):
            if label == "A":
                selected_peaks_AS = np.append(selected_peaks_AS, peaks_both[idx])
            else:
                selected_peaks_B = np.append(selected_peaks_B, peaks_both[idx])
    return np.sort(selected_peaks_AS), np.sort(selected_peaks_B)


def overlap_test(list1, list2, threshold):
    for element in list1:
        if list2.size == 0:
            break
        nearst_element = list2[np.abs(list2 - element).argmin()]
        if np.abs(nearst_element - element) <= threshold:
            list1 = np.delete(list1, np.where(list1 == element))
    return list1