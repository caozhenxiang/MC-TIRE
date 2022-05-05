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
        threshold = 16
    elif "hasc" in dataset:
        windowsize = 280
        threshold = 280
    elif "UCI" in dataset:
        windowsize = 8
        threshold = 5
    elif "change" in dataset:
        windowsize = 40
        threshold = 40
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


def overlap_test(list1, list2, threshold):
    for element in list1:
        if list2.size == 0:
            break
        nearst_element = list2[np.abs(list2 - element).argmin()]
        if np.abs(nearst_element - element) <= threshold:
            list1 = np.delete(list1, np.where(list1 == element))
    return list1