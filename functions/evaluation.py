import numpy as np
from functions import postprocessing, metrics


def show_result(generate_data, window_size, dissimilarities, parameters, threshold, enable_plot=True, f=None):
    # use simulsted data
    if generate_data:
        precision, recall, f1 = metrics.print_f1(dissimilarities, [threshold], parameters, window_size, generate_data)
        auc = metrics.get_auc(dissimilarities, [threshold], parameters, window_size, generate_data, enable_plot)[0]
        print("ratio:", f1 / auc)
        if enable_plot:
            metrics.plot_cp(dissimilarities, parameters, window_size, 0, 5000, threshold, plot_prominences=True,
                            simulate_data=generate_data)
    # load dataset
    else:
        change_points = []
        for idx in range(np.shape(parameters)[0] - 1):
            pre = parameters[idx]
            suc = parameters[idx + 1]
            if pre != suc:
                change_points.append(1)
            else:
                change_points.append(0)
        precision, recall, f1 = metrics.print_f1(dissimilarities, [threshold], change_points, window_size, generate_data)
        auc = metrics.get_auc(dissimilarities, [threshold], change_points, window_size, generate_data, enable_plot)[0]
        # print("ratio:", f1/auc)
        print(f1 / auc)
        if enable_plot:
            metrics.plot_cp(dissimilarities, change_points, window_size, 0, np.shape(parameters)[0], threshold,
                            plot_prominences=True, simulate_data=generate_data)
    if f is not None:
        print(precision, file=f)
        print(recall, file=f)
        print(f1, file=f)
        print(auc, file=f)
        print(f1 / auc, file=f)
        print("\n", file=f)


def smoothened_dissimilarity_measures(encoded_windows=None, encoded_windows_fft=None, window_size=20):
    """
    Calculation of smoothened dissimilarity measures

    Args:
        encoded_windows: TD latent representation of windows
        encoded_windows_fft:  FD latent representation of windows
        domain: TD/FD/both
        parameters: array with used parameters
        window_size: window size used
        par_smooth

    Returns:
        smoothened dissimilarity measures
    """
    if encoded_windows_fft is None:
        encoded_windows_both = encoded_windows
    elif encoded_windows is None:
        encoded_windows_both = encoded_windows_fft
    else:
        beta = np.quantile(postprocessing.distance(encoded_windows, window_size), 0.95)
        alpha = np.quantile(postprocessing.distance(encoded_windows_fft, window_size), 0.95)
        encoded_windows_both = np.concatenate((encoded_windows * alpha, encoded_windows_fft * beta), axis=1)

    encoded_windows_both = postprocessing.matched_filter(encoded_windows_both, window_size)  # smoothing for shared features (9)
    distances = postprocessing.distance(encoded_windows_both, window_size)
    distances = postprocessing.matched_filter(distances, window_size)  # smoothing for dissimilarity (12)

    return distances


def change_point_score(distances, window_size):
    """
    Gives the change point score for each time stamp. A change point score > 0 indicates that a new segment starts at that time stamp.

    Args:
    distances: postprocessed dissimilarity measure for all time stamps
    window_size: window size used in TD for CPD

    Returns:
    change point scores for every time stamp (i.e. zero-padded such that length is same as length time series)
    """
    prominences = np.array(postprocessing.new_peak_prominences(distances)[0])
    prominences = prominences / np.amax(prominences)
    return np.concatenate((np.zeros((window_size,)), prominences, np.zeros((window_size - 1,))))