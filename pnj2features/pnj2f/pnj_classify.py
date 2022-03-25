# Classify climate databases

import pickle

import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pnj2f.averages import get_avg_mag, get_avg_lat, get_avg_lon
from pnj2f.classifiers import classifyfull2, classify_composite2
from pnj2f.pnj_std import standard
from pnj2f.read_nc import find_file


def pnj_classify(classify_model, classify_period, classify_level, pathh5, pathplot, pathclassifier, pathclass):
    ''' Find data file and read '''
    file = find_file('ua', pathh5, classify_level, classify_model, classify_period)
    h5file = h5py.File(pathh5 + file, 'r')
    print(pathh5 + file)
    nc_lat_data = h5file['nc_lat_data'][:]
    nc_lon_data = h5file['nc_lon_data'][:]
    ua_data = h5file['x_data_sel'][:]
    grid_type = h5file['grid_type'][()]
    if not isinstance(grid_type, str):
        grid_type = h5file['grid_type'][()].decode("utf-8")
    h5file.close()

    '''zg data'''
    file = find_file('zg', pathh5, classify_level, classify_model, classify_period)
    h5file = h5py.File(pathh5 + file, 'r')
    zg_data = h5file['x_data_sel'][:]
    h5file.close()

    nmonths = ua_data.shape[0]
    if classify_model == 'jra':
        zg_data = zg_data[:174, :, :]
    if classify_period == 'piControl':
        nmonths = 1497
    month2classify = np.arange(0, nmonths)
    # month2classify = np.arange(0, 6)  # for tests only

    seedsx = []
    seedsy = []

    for idx, month in enumerate(month2classify):
        img = ua_data[idx, :, :] + abs(np.min(ua_data[idx, :, :]))
        seed_x, seed_y = np.where(img == np.max(img))
        seedsx.append(seed_x[0].item())
        seedsy.append(seed_y[0].item())

    '''Feature 1 for classification'''
    x1 = np.zeros_like(ua_data)
    scaler = MinMaxScaler()
    for month in month2classify:
        x1[month, :, :] = scaler.fit_transform(ua_data[month, :, :])

    '''Feature 2 for classification'''
    lat_matrix = np.vstack([nc_lat_data] * len(nc_lon_data)).T
    x2_0 = np.array([lat_matrix] * nmonths)
    x2 = np.zeros_like(x2_0)
    for month in month2classify:
        x2[month, :, :] = np.absolute(x2_0[month, :, :] - nc_lat_data[seedsx[month]])

    '''Extract trained classifier'''
    clf_file = open(pathclassifier, 'rb')
    clf = pickle.load(clf_file)

    '''Classify all months'''
    y_final, classtime = classifyfull2(x1, x2, clf, nmonths)
    y_final = np.reshape(y_final, [nmonths, len(nc_lat_data), len(nc_lon_data)])

    '''Characterization magnitudes'''
    Month = []
    ua_PNJ = []
    lat_PNJ = []
    uastd_PNJ = []
    lon_PNJ = []
    idx_type1 = []
    idx_type2 = []
    idx_normal = []
    month_type1 = []
    month_type2 = []
    zg_PNJ = []

    weight_factor = np.zeros(y_final[0, :, :].shape)  # Initialize numpy array
    for i in np.arange(0, len(nc_lat_data)):
        weight_factor[i, :] = np.cos(np.deg2rad(nc_lat_data[i]))  # Factor for average weighing

    for idx, month in enumerate(month2classify):
        # print('Characterizing month ' + str(month))

        Month.append(month + 1)

        auxlat = (get_avg_lat(y_final[idx, :, :], nc_lat_data,
                              ua_data[month, :, :], grid_type))
        if auxlat > 90:  # Singularity (cos): recompute
            auxlat = get_avg_lat(y_final[idx, :, :], nc_lat_data, np.ones_like(ua_data[month, :, :]),
                                 grid_type)
        lat_PNJ.append(auxlat)

        ua_PNJ.append(get_avg_mag(ua_data[month, :, :], y_final[idx, :, :], weight_factor, grid_type))
        uastd_PNJ.append(standard(ua_data[month, :, :], nc_lat_data))

        cross_opt = 'yes'
        lon_PNJ.append(get_avg_lon(y_final[idx, :, :], nc_lon_data, ua_data[month, :, :], cross_opt, grid_type))

        zg_PNJ.append(get_avg_mag(zg_data[month, :, :], y_final[idx, :, :], weight_factor, grid_type))

    for idx, month in enumerate(month2classify):

        if lat_PNJ[idx] < 35:  # not polar
            month_type1.append(month)
            idx_type1.append(idx)
        elif ua_PNJ[idx] < (np.average(ua_PNJ) - 2 * np.std(ua_PNJ)) and \
                np.count_nonzero(y_final[idx, :, :]) < 0.10 * len(nc_lat_data) * len(
            nc_lon_data):  # 192x288 es nc_lat_dataxnc_lon_data length
            month_type2.append(month)
            idx_type2.append(idx)
        else:
            idx_normal.append(idx)

    ''' Save the classified data to hdf5 file'''
    f = h5py.File(pathclass + classify_period + '/' + classify_model + '/'
                  + '/PNJ' + classify_level + '.h5', 'w')

    grp0 = f.create_group(classify_model)

    grp0['Month_PNJ'] = month2classify
    grp0['ua_PNJ'] = ua_PNJ
    grp0['uastd_PNJ'] = uastd_PNJ
    grp0['zg_PNJ'] = zg_PNJ
    grp0['lat_PNJ'] = lat_PNJ
    grp0['lon_PNJ'] = lon_PNJ

    if len(idx_normal) > 0:
        Month_normalPNJ = np.delete(Month, idx_type1 + idx_type2)
        lat_normalPNJ = np.delete(lat_PNJ, idx_type1 + idx_type2)
        lon_normalPNJ = np.delete(lon_PNJ, idx_type1 + idx_type2)
        ua_normalPNJ = np.delete(ua_PNJ, idx_type1 + idx_type2)
        uastd_normalPNJ = np.delete(uastd_PNJ, idx_type1 + idx_type2)
        zg_normalPNJ = np.delete(zg_PNJ, idx_type1 + idx_type2)

        uamap_normalPNJ = np.average(ua_data[idx_normal, :, :], axis=0)
        zgmap_normalPNJ = np.average(zg_data[idx_normal, :, :], axis=0)
        x1, x2 = features_composite(uamap_normalPNJ, zgmap_normalPNJ, nc_lat_data, nc_lon_data)
        classmap_normalPNJ, classtime = classify_composite2(x1, x2, clf)

        grp0['uamap_normalPNJ'] = uamap_normalPNJ
        grp0['zgmap_normalPNJ'] = np.average(zg_data[idx_normal, :, :], axis=0)
        grp0['classmap_normalPNJ'] = classmap_normalPNJ

        grp0['Month_normalPNJ'] = Month_normalPNJ
        grp0['ua_normalPNJ'] = ua_normalPNJ
        grp0['uastd_normalPNJ'] = uastd_normalPNJ
        grp0['zg_normalPNJ'] = zg_normalPNJ
        grp0['lat_normalPNJ'] = lat_normalPNJ
        grp0['lon_normalPNJ'] = lon_normalPNJ

        print('Month PNJ', np.array(Month_normalPNJ) - 1)

    if len(idx_type1) > 0:
        Month_type1event = [Month[i] for i in idx_type1]  # cannot access slides of list
        lat_type1event = [lat_PNJ[i] for i in idx_type1]
        lon_type1event = [lon_PNJ[i] for i in idx_type1]
        ua_type1event = [ua_PNJ[i] for i in idx_type1]
        uastd_type1event = [uastd_PNJ[i] for i in idx_type1]
        zg_type1event = [zg_PNJ[i] for i in idx_type1]

        uamap_type1event = np.average(ua_data[idx_type1, :, :], axis=0)
        zgmap_type1event = np.average(zg_data[idx_type1, :, :], axis=0)
        x1, x2 = features_composite(uamap_type1event, zgmap_type1event, nc_lat_data, nc_lon_data)
        classmap_type1event, classtime = classify_composite2(x1, x2, clf)

        grp0['uamap_type1event'] = uamap_type1event
        grp0['zgmap_type1event'] = np.average(zg_data[idx_type1, :, :], axis=0)
        grp0['classmap_type1event'] = classmap_type1event
        grp0['Month_type1event'] = Month_type1event
        grp0['ua_type1event'] = ua_type1event
        grp0['uastd_type1event'] = uastd_type1event
        grp0['zg_type1event'] = zg_type1event
        grp0['lat_type1event'] = lat_type1event
        grp0['lon_type1event'] = lon_type1event

        print('Month type 1', np.array(Month_type1event) - 1)

    if len(idx_type2) > 0:
        Month_type2event = [Month[i] for i in idx_type2]  # cannot access slides of list
        lat_type2event = [lat_PNJ[i] for i in idx_type2]
        lon_type2event = [lon_PNJ[i] for i in idx_type2]
        ua_type2event = [ua_PNJ[i] for i in idx_type2]
        uastd_type2event = [uastd_PNJ[i] for i in idx_type2]
        zg_type2event = [zg_PNJ[i] for i in idx_type2]

        uamap_type2event = np.average(ua_data[idx_type2, :, :], axis=0)
        zgmap_type2event = np.average(zg_data[idx_type2, :, :], axis=0)
        x1, x2 = features_composite(uamap_type2event, zgmap_type2event, nc_lat_data, nc_lon_data)
        classmap_type2event, classtime = classify_composite2(x1, x2, clf)

        grp0['uamap_type2event'] = uamap_type2event
        grp0['zgmap_type2event'] = np.average(zg_data[idx_type2, :, :], axis=0)
        grp0['classmap_type2event'] = classmap_type2event

        grp0['Month_type2event'] = Month_type2event
        grp0['ua_type2event'] = ua_type2event
        grp0['uastd_type2event'] = uastd_type2event
        grp0['zg_type2event'] = zg_type2event
        grp0['lat_type2event'] = lat_type2event
        grp0['lon_type2event'] = lon_type2event

        print('Month type 2', np.array(Month_type2event) - 1)

    f.close()


def features_composite(ua_data, zg_data, nc_lat_data, nc_lon_data):
    seed_x, seed_y = np.where(ua_data == np.max(ua_data))

    '''Feature 1 for classification'''
    scaler = MinMaxScaler()
    x1 = np.zeros_like(ua_data)
    x1[:, :] = scaler.fit_transform(ua_data)

    '''Feature 2 for classification'''
    lat_matrix = np.vstack([nc_lat_data] * len(nc_lon_data)).T
    x2_0 = lat_matrix.copy()
    x2 = np.absolute(x2_0 - nc_lat_data[seed_x])

    return x1, x2
