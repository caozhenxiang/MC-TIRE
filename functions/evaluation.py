import numpy as np
import pandas as pd
from functions import postprocessing, utils
from functions import metrics
from scipy.signal import find_peaks


def show_result(generate_data, window_size, dissimilarities, parameters, threshold, loss_coherent, loss_incoherent,
                enable_plot=True, f=None, data=None):
    # use simulsted data
    if generate_data:
        precision, recall, f1 = metrics.print_f1(dissimilarities, [threshold], parameters, window_size, generate_data)
        auc = metrics.get_auc(dissimilarities, [threshold], parameters, window_size, generate_data, enable_plot)[0]
        print("ratio:", f1 / auc)
        if enable_plot:
            metrics.plot_cp(dissimilarities, parameters, window_size, 0, 5000, threshold, plot_prominences=True,
                            simulate_data=generate_data)
    # load generated datasets
    else:
        change_points = []
        for idx in range(np.shape(parameters)[0] - 1):
            pre = parameters[idx]
            suc = parameters[idx + 1]
            if pre != suc:
                change_points.append(1)
            else:
                change_points.append(0)
        if isinstance(dissimilarities, dict):
            related_channels = np.arange(0, loss_coherent.size)
            ratio = loss_incoherent/(loss_coherent + loss_incoherent)
            index_array = np.argsort(ratio)
            max_ratio = ratio[index_array[-1]]
            large_ratio = ratio[ratio >= 0.65]
            small_ratio = ratio[ratio < 0.65]
            if small_ratio.size == 0:
                small_ratio = [0]

            # check ratio to assign change-types
            peaks_AS = find_peaks(dissimilarities['A&S'])[0]
            peaks_B = find_peaks(dissimilarities['B'])[0]
            if np.mean(small_ratio) < 0.25 and large_ratio.size != 0:
                peaks_both = []
                selected_peaks_AS = []
                selected_peaks_B = peaks_B

            elif np.mean(ratio) < 0.25:
                peaks_both = []
                selected_peaks_AS = peaks_AS
                selected_peaks_B = []

            # merge detections from two branches
            else:
                peaks_both = []
                new_peaks_AS = []
                peaks_B_bu = peaks_B
                stop_point = peaks_AS[-1]
                nr_AS = int(0.1 * peaks_AS.size)
                nr_B = int(0.1 * peaks_B.size)
                # find overlaps of two branches
                for peak_AS in peaks_AS:
                    if peaks_B.size == 0:
                        stop_point = peak_AS
                        break
                    nearst_peak = peaks_B[np.abs(peaks_B - peak_AS).argmin()]
                    if np.abs(nearst_peak - peak_AS) <= threshold:
                        peaks_B = np.delete(peaks_B, np.where(peaks_B == nearst_peak))
                        alpha = dissimilarities['A&S'][peak_AS] / np.mean(np.sort(dissimilarities['A&S'][peaks_AS])[-nr_AS:])
                        beta = dissimilarities['B'][nearst_peak] / np.mean(np.sort(dissimilarities['B'][peaks_B_bu])[-nr_B:])
                        merged_loc = np.around((alpha * peak_AS + beta * nearst_peak) / (alpha + beta)).astype(int)
                        peaks_both.append(merged_loc)
                    else:
                        new_peaks_AS.append(peak_AS)
                new_peaks_AS = np.asarray(np.concatenate([new_peaks_AS, peaks_AS[np.argwhere(peaks_AS == stop_point)
                                                                                 [0][0]:]], axis=0), dtype=np.int64)
                new_peaks_B = np.asarray(peaks_B, dtype=np.int64)
                peaks_B = find_peaks(dissimilarities['B'])[0]

                # only take significant changes in each branch

                selected_peaks_AS = np.array(new_peaks_AS)[list(np.where(dissimilarities['A&S'][new_peaks_AS] >= 0.25 *
                                                           np.mean(np.sort(dissimilarities['A&S'][peaks_AS])[-nr_AS:])))[0]]
                selected_peaks_B = np.array(new_peaks_B)[list(np.where(dissimilarities['B'][new_peaks_B] >= 0.25 *
                                                         np.mean(np.sort(dissimilarities['B'][peaks_B])[-nr_B:])))[0]]
                selected_peaks_AS = utils.overlap_test(selected_peaks_AS, np.array(peaks_both), window_size)
                selected_peaks_B = utils.overlap_test(selected_peaks_B, np.array(peaks_both), window_size)
            detected_peaks = np.sort(np.concatenate((selected_peaks_AS, selected_peaks_B, peaks_both)))
            precision, recall, f1 = metrics.print_f1(None, [threshold], change_points, window_size, generate_data,
                                                     peaks=detected_peaks)
            if enable_plot:
                metrics.plot_cp_channel(data, selected_peaks_AS, selected_peaks_B, peaks_both, change_points, threshold,
                                        window_size, related_channels)
        else:
            precision, recall, f1 = metrics.print_f1(dissimilarities, [threshold], change_points, window_size, generate_data)
            if enable_plot:
                metrics.plot_cp(dissimilarities, change_points, window_size, 0, np.shape(parameters)[0], threshold,
                                plot_prominences=True, simulate_data=generate_data)

    if f is not None:
        print(precision, file=f)
        print(recall, file=f)
        print(f1, file=f)
        print("\n", file=f)
        print("\n", file=f)



def smoothened_dissimilarity_measures(encoded_AS, encoded_AS_fft, encoded_B, encoded_B_fft, window_size):
    # window_size = 40
    if encoded_AS_fft is None:
        encoded_AS_both = encoded_AS
        encoded_B_both = encoded_B
    elif encoded_AS is None:
        encoded_AS_both = encoded_AS_fft
        encoded_B_both = encoded_B_fft
    else:
        beta_AS = np.quantile(postprocessing.distance(encoded_AS, window_size), 0.95)
        alpha_AS = np.quantile(postprocessing.distance(encoded_AS_fft, window_size), 0.95)
        encoded_AS_both = np.concatenate((encoded_AS * alpha_AS, encoded_AS_fft * beta_AS), axis=1)
        beta_B = np.quantile(postprocessing.distance(encoded_B, window_size), 0.95)
        alpha_B = np.quantile(postprocessing.distance(encoded_B_fft, window_size), 0.95)
        encoded_B_both = np.concatenate((encoded_B * alpha_B, encoded_B_fft * beta_B), axis=1)
    distances = {"A&S":[], "B":[]}

    encoded_windows_both = {"A&S": postprocessing.matched_filter(encoded_AS_both, window_size),
                            "B": postprocessing.matched_filter(encoded_B_both, window_size)}
    for idx, value in encoded_windows_both.items():
        distances[idx] = postprocessing.matched_filter(postprocessing.distance(value, window_size), window_size)
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