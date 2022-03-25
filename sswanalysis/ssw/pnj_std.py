# Standard 60N parallel method

import numpy as np


def standard(arr, nc_lat_data):
    idx = find_nearest(nc_lat_data, 60)

    u = np.average(arr[idx, :])

    return u


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
