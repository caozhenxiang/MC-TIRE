import numpy as np
import matplotlib.pyplot as plt
import warnings
from functions import postprocessing
from scipy.signal import find_peaks, peak_prominences
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


def tpr_fpr(bps, distances, method="prominence", tol_dist=0):
    """Calculation of TPR and FPR

    Args:
        bps: list of breakpoints (change points)
        distances: list of dissimilarity scores
        method: prominence- or height-based change point score
        tol_dist: toleration distance

    Returns:
        list of TPRs and FPRs for different values of the detection threshold
    """
    peaks = find_peaks(distances)[0]
    peaks_prom = peak_prominences(distances, peaks)[0]
    peaks_prom_all = np.array(postprocessing.new_peak_prominences(distances)[0])
    distances = np.array(distances)

    bps = np.array(bps)

    if method == "prominence":
        nr_bps = len(bps)
        # determine for each bp the allowed range s.t. alarm is closest to bp
        ranges = [0] * nr_bps
        for i in range(nr_bps):
            if i == 0:
                left = 0
            else:
                left = right
            if i == (nr_bps - 1):
                right = len(distances)
            else:
                right = int((bps[i][-1 ] + bps[ i +1][0] ) /2 ) +1
            ranges[i] = [left ,right]

        quantiles = np.quantile(peaks_prom, np.array(range(101))/100)
        quantiles_set = set(quantiles)
        quantiles_set.add(0.)
        quantiles = list(quantiles_set)
        quantiles.sort()

        nr_quant = len(quantiles)
        ncr = [0. ] *nr_quant
        nal = [0. ] *nr_quant

        for i in range(nr_quant):
            for j in range(nr_bps):
                bp_nbhd = peaks_prom_all[max(bps[j][0]-tol_dist, ranges[j][0]):min(bps[j][-1]+tol_dist+1, ranges[j][1])]
                if len(bp_nbhd) > 0:
                    if max(bp_nbhd) >= quantiles[i]:
                        ncr[i] += 1
            heights_alarms = distances[peaks_prom_all >= quantiles[i]]
            nal[i] = len(heights_alarms)

        ncr = np.array(ncr)
        nal = np.array(nal)

        ngt = nr_bps

        tpr = ncr / ngt
        fpr = (nal - ncr) / nal
        tpr = list(tpr[nal != 0])
        fpr = list(fpr[nal != 0])
    return tpr, fpr


def plot_cp(distances, parameters, window_size, time_start, time_stop, threshold, plot_prominences=False,
            simulate_data=True, weights=None):
    """
    Plots dissimilarity measure with ground-truth changepoints

    Args:
        distances: dissimilarity measures
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD
        time_start: first time stamp of plot
        time_stop: last time stamp of plot
        plot_prominences: True/False

    Returns:
        plot of dissimilarity measure with ground-truth changepoints

    """
    if simulate_data:
        breakpoints = postprocessing.parameters_to_cps(parameters, window_size)
    else:
        distances = np.concatenate((np.zeros((window_size,)), distances, np.zeros((window_size,))))
        if weights is not None:
            weights = np.concatenate((np.zeros((2, window_size)), weights, np.zeros((2, window_size))), axis=1)
        breakpoints = parameters.copy()
        breakpoints.append(0)
        np.append(breakpoints, 0)

    t = range(len(distances))

    x = t
    z = distances
    y = breakpoints  # [:,0]
    cps = [idx for idx in range(len(y)) if y[idx] > 0]
    peaks = find_peaks(distances)[0]
    peaks_prom_all = np.array(postprocessing.new_peak_prominences(distances)[0])

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(x, z, color="black")
    hit_list = []
    for cp in cps:
        if peaks.size == 0:
            break
        nearst_peak = peaks[np.abs(peaks - cp).argmin()]
        if np.abs(nearst_peak - cp) <= threshold:
            peaks = np.delete(peaks, np.where(peaks == nearst_peak))
            hit_list.append(nearst_peak)

    if plot_prominences:
        ax.plot(x, peaks_prom_all, color="blue")

    ax.set_xlim(time_start, time_stop)
    ax.set_ylim(0, 1.5 * max(z))
    plt.xlabel("time")
    plt.ylabel("dissimilarity")
    ax.plot(peaks, distances[peaks], 'ko', label='false positives')
    reset_flags = {"A&S": True, "B": True}
    if weights is None:
        ax.plot(hit_list, distances[hit_list], 'go', label='accurately detected')
        ax.legend()
    else:
        for idx, item in enumerate(hit_list):
            # weight = tuple(weights[:, item])
            # ax.plot(item, distances[item], marker='o', color=weight)
            cls = np.argmax(weights, axis=0)[item]
            if cls == 0:
                if reset_flags["A&S"]:
                    ax.plot(item, distances[item], 'go', label='changes detected as A or S type')
                    reset_flags["A&S"] = False
                else:
                    ax.plot(item, distances[item], 'go')
            elif cls == 1:
                if reset_flags["B"]:
                    ax.plot(item, distances[item], 'ro', label='changes detected as B type')
                    reset_flags["B"] = False
                else:
                    ax.plot(item, distances[item], 'ro')
        ax.legend()
    height_line = 1

    ax.fill_between(x[0:len(y)], 0, height_line, where=y > 0.0001 * np.ones_like(y),
                    color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x[0:len(y)], 0, height_line, where=y > 0.25 * np.ones_like(y),
                    color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x[0:len(y)], 0, height_line, where=y > 0.5 * np.ones_like(y),
                    color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x[0:len(y)], 0, height_line, where=y > 0.75 * np.ones_like(y),
                    color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x[0:len(y)], 0, height_line, where=y > 0.9 * np.ones_like(y),
                    color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    plt.show()


def get_auc(distances, tol_distances, parameters, window_size, simulate_data=True, enable_plot=True):
    """
    Calculation of AUC for toleration distances in range(TD_start, TD_stop, TD_step) + plot of corresponding ROC curves

    Args:
        distances: dissimilarity measures
        tol_distances: list of different toleration distances
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD

    Returns:
        list of AUCs for every toleration distance
    """
    if simulate_data:
        breakpoints = postprocessing.parameters_to_cps(parameters, window_size)
    else:
        distances = np.concatenate((np.zeros((window_size,)), distances, np.zeros((window_size,))))
        breakpoints = parameters.copy()
    legend = []
    list_of_lists = postprocessing.cp_to_timestamps(breakpoints, 0, np.size(breakpoints))
    auc = []

    for i in tol_distances:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tpr, fpr = tpr_fpr(list_of_lists, distances, "prominence", i)
        plt.plot(fpr, tpr)
        legend.append("tol. dist. = " + str(i))
        auc.append(np.abs(np.trapz(tpr, x=fpr)))

    #print("partial_AUC:", auc)
    print(auc[0])
    if enable_plot:
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC curve")
        plt.plot([0, 1], [0, 1], '--')
        legend.append("TPR=FPR")
        plt.legend(legend)
        plt.show()

        plt.plot(tol_distances, auc)
        plt.xlabel("toleration distance")
        plt.ylabel("AUC")
        plt.title("AUC")
        plt.show()
    return auc


def cal_f1(bps, distances, tol_dist, peaks=None):
    if peaks is not None:
        peaks = peaks
    else:
        peaks = find_peaks(distances)[0]
    hit_list = []  # TP
    miss_list = []  # FN
    for bp in bps:
        if peaks.size == 0:
            break
        nearst_peak = peaks[np.abs(peaks - bp).argmin()]
        if np.abs(nearst_peak - bp) <= tol_dist:
            peaks = np.delete(peaks, np.where(peaks == nearst_peak))
            hit_list.append(nearst_peak)
        else:
            miss_list.append(bp)
    n_tp = len(hit_list)
    n_fn = len(miss_list)
    n_fp = peaks.shape[0]
    precision = n_tp / (n_tp + n_fp + 1e-8)
    recall = n_tp / (n_tp + n_fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision, recall, f1


def print_f1(distances, tol_distance, parameters, window_size, simulate_data=True, peaks=None):
    """
    Calculation of AUC for toleration distances in range(TD_start, TD_stop, TD_step) + plot of corresponding ROC curves

    Args:
        distances: dissimilarity measures
        tol_distances: list of different toleration distances
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD

    Returns:
        list of AUCs for every toleration distance
    """
    if peaks is not None:
        if simulate_data:
            breakpoints = postprocessing.parameters_to_cps(parameters, window_size)
        else:
            breakpoints = parameters.copy()
        list_of_lists = postprocessing.cp_to_timestamps(breakpoints, 0, np.size(breakpoints))
        precision, recall, f1 = cal_f1(list_of_lists, distances, tol_distance, peaks+window_size)
    else:
        if simulate_data:
            breakpoints = postprocessing.parameters_to_cps(parameters, window_size)
        else:
            distances = np.concatenate((np.zeros((window_size,)), distances, np.zeros((window_size,))))
            breakpoints = parameters.copy()
        list_of_lists = postprocessing.cp_to_timestamps(breakpoints, 0, np.size(breakpoints))
        precision, recall, f1 = cal_f1(list_of_lists, distances, tol_distance)
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f1-score:" + str(f1))

    return precision, recall, f1


def plot_cp_channel(data, peaks_AS, peaks_B, peaks_both, change_points, threshold, window_size, related_cahnnels):
    with sns.plotting_context(context="paper"):
        plt.rcParams['font.size'] = '20'
        nr_channels = data.shape[1]
        cols = []
        for i in range(nr_channels):
            cols.append("ch" + str(i))
        df = pd.DataFrame(data, columns=cols)
        axes = df.plot(subplots=True, figsize=(10, 1.5*nr_channels), color="silver")
        cps = np.argwhere(np.array(change_points) > 0)

        peaks_AS = np.array(peaks_AS) + window_size
        peaks_B = np.array(peaks_B) + window_size
        peaks_both = np.array(peaks_both) + window_size
        # peaks_both = np.array([])

        hit_list_AS = []
        hit_list_B = []
        hit_list_both = []

        for cp in cps:
            if peaks_both.size == 0:
                break
            nearst_peak = peaks_both[np.abs(peaks_both - cp).argmin()]
            if np.abs(nearst_peak - cp) <= threshold:
                peaks_both = np.delete(peaks_both, np.where(peaks_both == nearst_peak))
                hit_list_both.append(nearst_peak)
                cps = np.delete(cps, np.where(cps == cp)[0])
        for cp in cps:
            if peaks_AS.size == 0:
                break
            nearst_peak = peaks_AS[np.abs(peaks_AS - cp).argmin()]
            if np.abs(nearst_peak - cp) <= threshold:
                peaks_AS = np.delete(peaks_AS, np.where(peaks_AS == nearst_peak))
                hit_list_AS.append(nearst_peak)
                cps = np.delete(cps, np.where(cps == cp)[0])
        for cp in cps:
            if peaks_B.size == 0:
                break
            nearst_peak = peaks_B[np.abs(peaks_B - cp).argmin()]
            if np.abs(nearst_peak - cp) <= threshold:
                peaks_B = np.delete(peaks_B, np.where(peaks_B == nearst_peak))
                hit_list_B.append(nearst_peak)
                cps = np.delete(cps, np.where(cps == cp)[0])

        false_potivies = np.concatenate((peaks_AS, peaks_B))

        hit_list_both = np.array(hit_list_both)
        hit_list_AS = np.array(hit_list_AS)
        hit_list_B = np.array(hit_list_B)
        legend_elements = []
        flags = {"both": False,
                 "co": False,
                 "inco": False,
                 "fp": False}
        for idx, ax in enumerate(axes):
            ax.legend([], [], frameon=False)
            ax.set_ylabel("amplitude")
            if idx in related_cahnnels:
                ax.fill_between(range(len(change_points)), 0, 1,
                                  where=change_points > np.zeros_like(change_points),
                                  color='red', alpha=1, transform=ax.get_xaxis_transform())
                if hit_list_both.size != 0:
                    flags["both"] = True
                    ax.plot(hit_list_both, 0.5 * np.ones_like(hit_list_both), 'g*', markersize=10,
                              label='detected by both branches', transform=ax.get_xaxis_transform())
                    ax.vlines(hit_list_both, 0, 0.5, color='g', transform=ax.get_xaxis_transform())
                if hit_list_AS.size != 0:
                    flags["co"] = True
                    ax.plot(hit_list_AS, 0.5 * np.ones_like(hit_list_AS), 'bs', markersize=6,
                              label='detected as coherent changes', transform=ax.get_xaxis_transform())
                    ax.vlines(hit_list_AS, 0, 0.5, color='b', transform=ax.get_xaxis_transform(), linewidth=2)
                if hit_list_B.size != 0:
                    flags["inco"] = True
                    ax.plot(hit_list_B, 0.5 * np.ones_like(hit_list_B), '^', markerfacecolor='darkorange',
                            markeredgecolor="darkorange", markersize=6,
                            label='detected as incoherent changes',
                            transform=ax.get_xaxis_transform())
                    ax.vlines(hit_list_B, 0, 0.5, color='orange', transform=ax.get_xaxis_transform(), linewidth=2)
                if false_potivies.size != 0:
                    flags["fp"] = True
                    ax.plot(false_potivies, 0.5 * np.ones_like(false_potivies), 'ko', label='false positives',
                            markersize=6, transform=ax.get_xaxis_transform())
                    ax.vlines(false_potivies, 0, 0.5, color='black', transform=ax.get_xaxis_transform(), linewidth=2)
        legend_elements.append(Line2D([0], [0], color='silver', label="Time series' visualization in each channel"))
        legend_elements.append(Line2D([0], [0], color='red', label="Ground-truth CPs"))
        if flags["both"]:
            legend_elements.append(Line2D([0], [0], color='w', marker='*', markerfacecolor="green", markersize=12,
                                          label="Alarms detected by both branches"))
        if flags["co"]:
            legend_elements.append(Line2D([0], [0], color='w', marker='s', markerfacecolor="blue", markersize=8,
                                          label="Alarms detected by coherence branch"))
        if flags["inco"]:
            legend_elements.append(Line2D([0], [0], color='w', marker='^', markerfacecolor="darkorange", markersize=8,
                                          label="Alarms detected by residual branch"))
        if flags["fp"]:
            legend_elements.append(Line2D([0], [0], color='w', marker='o', markerfacecolor="k", markersize=8,
                                          label="false-positive detections"))
        plt.xlabel("time")

        plt.figlegend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.16), fontsize="10.5")
        # plt.savefig("test_UCI.png", bbox_inches="tight",dpi=300)
        plt.show()