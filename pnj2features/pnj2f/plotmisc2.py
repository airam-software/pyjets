# Plot some results

import h5py
import matplotlib.pyplot as plt
import numpy as np

from pnj2f.read_nc import find_file


def plot_months(classify_model, classify_period, classify_level, pathh5, pathclass, pathplot):
    ''' Data to classify '''
    file = find_file('ua', pathh5, classify_level, classify_model, classify_period)
    h5file = h5py.File(pathh5 + file, 'r')
    nc_lat_data = h5file['nc_lat_data'][:]
    nc_lon_data = h5file['nc_lon_data'][:]
    ua2classify = h5file['x_data_sel'][:]
    grid_type = h5file['grid_type'][()]
    h5file.close()

    '''zg data'''
    file = find_file('zg', pathh5, classify_level, classify_model, classify_period)
    h5file = h5py.File(pathh5 + file, 'r')
    zg_data = h5file['x_data_sel'][:]
    h5file.close()

    nmonths = ua2classify.shape[0]
    month2classify = np.arange(0, nmonths)

    seedsx = []
    seedsy = []

    for idx, month in enumerate(month2classify):
        img = ua2classify[idx, :, :] + abs(np.min(ua2classify[idx, :, :]))
        seed_x, seed_y = np.where(img == np.max(img))
        seedsx.append(seed_x[0].item())
        seedsy.append(seed_y[0].item())

    ''' Read hdf5 file'''
    f = h5py.File(pathclass + classify_period + '/' + classify_model + '/PNJ.h5', 'r')

    grp0 = f[classify_model]

    if 'uamap_normalPNJ' in grp0.keys():
        uamap_normalPNJ = grp0['uamap_normalPNJ'][:]
        zgmap_normalPNJ = grp0['zgmap_normalPNJ'][:]
        classmap_normalPNJ = grp0['classmap_normalPNJ'][:]
        Month_normalPNJ = grp0['Month_normalPNJ'][:]
        ua_normalPNJ = grp0['ua_normalPNJ'][:]
        uastd_normalPNJ = grp0['uastd_normalPNJ'][:]
        zg_normalPNJ = grp0['zg_normalPNJ'][:]
        lat_normalPNJ = grp0['lat_normalPNJ'][:]
        lon_normalPNJ = grp0['lon_normalPNJ'][:]

    if 'uamap_type1event' in grp0.keys():
        uamap_type1event = grp0['uamap_type1event'][:]
        zgmap_type1event = grp0['zgmap_type1event'][:]
        classmap_type1event = grp0['classmap_type1event'][:]
        Month_type1event = grp0['Month_type1event'][:]
        ua_type1event = grp0['ua_type1event'][:]
        uastd_type1event = grp0['uastd_type1event'][:]
        zg_type1event = grp0['zg_type1event'][:]
        lat_type1event = grp0['lat_type1event'][:]
        lon_type1event = grp0['lon_type1event'][:]

    if 'uamap_type2event' in grp0.keys():
        uamap_type2event = grp0['uamap_type2event'][:]
        zgmap_type2event = grp0['zgmap_type2event'][:]
        classmap_type2event = grp0['classmap_type2event'][:]
        Month_type2event = grp0['Month_type2event'][:]
        ua_type2event = grp0['ua_type2event'][:]
        uastd_type2event = grp0['uastd_type2event'][:]
        zg_type2event = grp0['zg_type2event'][:]
        lat_type2event = grp0['lat_type2event'][:]
        lon_type2event = grp0['lon_type2event'][:]

    ''' Plot uastd versus classified latitude'''
    plt.figure()
    if 'uamap_normalPNJ' in grp0.keys():
        plt.plot(lat_normalPNJ, uastd_normalPNJ, 'bo', label='60N method PNJ')
    if 'uamap_type1event' in grp0.keys():
        plt.plot(lat_type1event, uastd_type1event, 'ro', label='60N method type 1')
    if 'uamap_type2event' in grp0.keys():
        plt.plot(lat_type2event, uastd_type2event, 'go', label='60N method type 2')

    if 'uamap_normalPNJ' in grp0.keys():
        plt.plot(lat_normalPNJ, ua_normalPNJ, 'bx', label='classifier PNJ')
    if 'uamap_type1event' in grp0.keys():
        plt.plot(lat_type1event, ua_type1event, 'rx', label='classifier type 1')
    if 'uamap_type2event' in grp0.keys():
        plt.plot(lat_type2event, ua_type2event, 'gx', label='classifier type 2')

    plt.xlabel('latitude [deg]')
    plt.ylabel('$u$ [m/s]')
    plt.legend()
    plt.grid()
    plt.savefig(pathplot + classify_period + '/' + classify_model + '/comparison1.png',
                format='png', bbox_inches='tight')
    plt.close()
