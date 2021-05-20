import numpy as np

def ts_to_windows(ts, onset, window_size, stride, normalization="timeseries"):
    """Transforms time series into list of windows"""
    windows = []
    len_ts = len(ts)
    onsets = range(onset, len_ts -window_size +1, stride)

    if normalization == "timeseries":
        for timestamp in onsets:
            windows.append(ts[timestamp:timestamp +window_size])
    elif normalization == "window":
        for timestamp in onsets:
            windows.append \
                (np.array(ts[timestamp:timestamp +window_size] ) -np.mean(ts[timestamp:timestamp +window_size]))

    return np.array(windows)


def combine_ts(list_of_windows):
    """
    Combines a list of windows from multiple views to one list of windows

    Args:
        list_of_windows: list of windows from multiple views

    Returns:
        one array with the concatenated windows
    """
    nr_ts, nr_windows, window_size = np.shape(list_of_windows)
    tss = np.array(list_of_windows)
    new_ts = []
    for i in range(nr_windows):
        new_ts.append(tss[:, i, :].flatten())
    return np.array(new_ts)