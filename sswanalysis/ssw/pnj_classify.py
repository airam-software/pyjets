# Classify climate databases

import pickle

import h5py
import numpy as np
import pandas as pd
import skimage.measure as measure
import sklearn.preprocessing
from scipy import ndimage

from ssw.averages import get_avg_mag, get_avg_lat, get_avg_lon
from ssw.classifiers import classifyfull3, classify_composite3
from ssw.plotmisc2 import plot_onewinter
from ssw.pnj_std import standard
from ssw.read_nc import find_file

dates_ssw = ['1958-01-30',
             '1958-01-31',
             '1958-02-01',
             '1963-01-27',
             '1963-01-28',
             '1963-01-29',
             '1966-02-21',
             '1966-02-23',
             '1966-02-24',
             '1968-01-06',
             '1968-01-07',
             '1968-01-08',
             '1971-01-17',
             '1971-01-18',
             '1971-01-19',
             '1973-01-30',
             '1973-01-31',
             '1973-02-01',
             '1977-01-08',
             '1977-01-09',
             '1977-01-10',
             '1979-02-21',
             '1979-02-22',
             '1979-02-23',
             '1984-12-31',
             '1985-01-01',
             '1985-01-02',
             '1987-12-06',
             '1987-12-07',
             '1987-12-08',
             '1988-03-13',
             '1988-03-14',
             '1988-03-15',
             '1989-02-20',
             '1989-02-21',
             '1989-02-22',
             '1999-02-25',
             '1999-02-26',
             '1999-02-27'
             ]


def pnj_classify(classify_model, classify_period, classify_level, pathh5, pathplot, pathclassifier, pathclass):
    winterstart = [0,
                   59,
                   149,
                   240,
                   330,
                   420,
                   510,
                   601,
                   691,
                   781,
                   871,
                   962,
                   1052,
                   1142,
                   1232,
                   1323,
                   1413,
                   1503,
                   1593,
                   1684,
                   1774,
                   1864,
                   1954,
                   2045,
                   2135,
                   2225,
                   2315,
                   2406,
                   2496,
                   2586,
                   2676,
                   2767,
                   2857,
                   2947,
                   3037,
                   3128,
                   3218,
                   3308,
                   3398,
                   3489,
                   3579,
                   3669,
                   3759,
                   3850,
                   3940,
                   4030,
                   4120,
                   4211,
                   4301,
                   4391,
                   4481,
                   4572,
                   4662,
                   4752,
                   4842,
                   4933,
                   5023,
                   5113]

    winterend = [58,
                 148,
                 239,
                 329,
                 419,
                 509,
                 600,
                 690,
                 780,
                 870,
                 961,
                 1051,
                 1141,
                 1231,
                 1322,
                 1412,
                 1502,
                 1592,
                 1683,
                 1773,
                 1863,
                 1953,
                 2044,
                 2134,
                 2224,
                 2314,
                 2405,
                 2495,
                 2585,
                 2675,
                 2766,
                 2856,
                 2946,
                 3036,
                 3127,
                 3217,
                 3307,
                 3397,
                 3488,
                 3578,
                 3668,
                 3758,
                 3849,
                 3939,
                 4029,
                 4119,
                 4210,
                 4300,
                 4390,
                 4480,
                 4571,
                 4661,
                 4751,
                 4841,
                 4932,
                 5022,
                 5112,
                 5202
                 ]

    ''' Find data file and read '''

    file = find_file('ua', pathh5, classify_level, classify_model, classify_period)
    h5file = h5py.File(pathh5 + file, 'r')
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

    df = pd.read_csv('./results/dates.csv', sep=',', header=0)
    datesformat = df['Date'].values
    datesformat2 = pd.to_datetime(datesformat, format='%Y-%m-%d')

    # nyears = 1  # TESTS
    # winterstart = winterstart[0:nyears] # TESTS
    # winterend = winterend[0:nyears] # TESTS

    day2classify = np.arange(winterstart[0], winterend[-1] + 1)

    '''Extract trained classifier'''
    clf_file = open(pathclassifier, 'rb')
    clf = pickle.load(clf_file)

    y_final = np.array([])

    # Scale features
    scaler = sklearn.preprocessing.RobustScaler()
    x1_data = np.zeros_like(ua_data)
    lat_matrix = np.vstack([nc_lat_data] * len(nc_lon_data)).T
    x2_data = np.array([lat_matrix] * len(day2classify))
    x3_data = np.zeros_like(zg_data)
    for i in day2classify:
        x1_data[i, :, :] = scaler.fit_transform(ua_data[i, :, :])
        x2_data[i, :, :] = scaler.fit_transform(x2_data[i, :, :])
        x3_data[i, :, :] = scaler.fit_transform(zg_data[i, :, :])

    f1 = h5py.File('./scaled.h5', 'w')
    f1.create_dataset('x1', data=x1_data)
    f1.create_dataset('x2', data=x2_data)
    f1.create_dataset('x3', data=x3_data)
    f1.close()

    f1 = h5py.File('./scaled.h5', 'r')
    x1_data = f1['x1'][:]
    x2_data = f1['x2'][:]
    # x3_data = f1['x3'][:]
    x3_data = zg_data.copy()
    f1.close()

    for i in np.arange(0, len(winterstart)):
        idx1 = winterstart[i]
        idx2 = winterend[i]

        print('Classifying winter ' + str(datesformat2[idx1]) + '-' + str(datesformat2[idx2]))

        x1 = x1_data[idx1:idx2 + 1, :, :].copy()
        x2 = x2_data[idx1:idx2 + 1, :, :].copy()
        x3 = x3_data[idx1:idx2 + 1, :, :].copy()

        '''Classify all days'''
        y_final0 = classifyfull3(x1, x2, x3, clf, len(np.arange(idx1, idx2 + 1)))
        y_final0 = np.asarray(y_final0)
        y_final = np.append(y_final, y_final0)

    y_final = np.reshape(y_final, [len(day2classify), len(nc_lat_data), len(nc_lon_data)])

    '''Characterization magnitudes'''
    Day = []
    ua_PNJ = []
    lat_PNJ = []
    uastd_PNJ = []
    lon_PNJ = []
    idx_type1 = []
    idx_type2 = []
    idx_normal = []
    zg_PNJ = []
    type_PNJ = []

    weight_factor = np.zeros(ua_data[0, :, :].shape)
    for i in np.arange(0, len(nc_lat_data)):
        weight_factor[i, :] = np.cos(np.deg2rad(nc_lat_data[i]))  # Factor for average weighing

    for idx, day in enumerate(day2classify):
        Day.append(day + 1)

        if np.sum(y_final[idx, :, :]) == 0:
            print('ERROR IN MONTH ' + str(idx))
            lat_PNJ.append(float('nan'))
            ua_PNJ.append(float('nan'))
            uastd_PNJ.append(float('nan'))
            lon_PNJ.append(float('nan'))
            zg_PNJ.append(float('nan'))
        else:
            ua_i, uastd_i, zg_i, lat_i, lon_i = \
                characterization(nc_lat_data, nc_lon_data, grid_type, ua_data[day, :, :],
                                 zg_data[day, :, :], y_final[idx, :, :], weight_factor, day)

            ua_PNJ.append(ua_i)
            uastd_PNJ.append(uastd_i)
            zg_PNJ.append(zg_i)
            lat_PNJ.append(lat_i)
            lon_PNJ.append(lon_i)

    ualim = np.nanmean(ua_PNJ) - 0.5 * np.nanstd(ua_PNJ)  # for type 2 events

    for idx, item in enumerate(day2classify):
        type_pnj_i = detectpnjtype(y_final[idx, :, :], lat_PNJ[idx], ua_PNJ[idx], ualim)
        if type_pnj_i == 2:
            idx_type2.append(idx)
            type_PNJ.append(2)
        elif lat_PNJ[idx] < 40:
            idx_type1.append(idx)
            type_PNJ.append(1)
        else:
            idx_normal.append(idx)
            type_PNJ.append(0)

    # Plot one winter
    for i in np.arange(0, len(winterstart)):
        idx1 = winterstart[i]
        idx2 = winterend[i]

        idx3 = winterstart[i] - winterstart[0]
        idx4 = winterend[i] - winterstart[0]

        print('Plotting winter ' + str(datesformat2[idx1]) + '-' + str(datesformat2[idx2]))

        plot_onewinter(y_final[idx3:idx4 + 1, :, :], ua_data[idx1:idx2 + 1, :, :], zg_data[idx1:idx2 + 1, :, :],
                       type_PNJ[idx3:idx4 + 1], nc_lat_data, nc_lon_data,
                       str(datesformat2.year[idx1]) + '-' + str(datesformat2.year[idx2]))

    ''' Save the classified data to hdf5 file'''

    print('Saving data to hdf5')
    f = h5py.File(pathclass + classify_period + '/' + classify_model + '/'
                  + '/PNJ' + classify_level + '.h5', 'w')

    grp0 = f.create_group(classify_model)

    grp0['Day_PNJ'] = day2classify
    grp0['ua_PNJ'] = ua_PNJ
    grp0['uastd_PNJ'] = uastd_PNJ
    grp0['zg_PNJ'] = zg_PNJ
    grp0['lat_PNJ'] = lat_PNJ
    grp0['lon_PNJ'] = lon_PNJ

    if len(idx_normal) > 0:
        Day_normalPNJ = np.delete(Day, idx_type1 + idx_type2)
        lat_normalPNJ = np.delete(lat_PNJ, idx_type1 + idx_type2)
        lon_normalPNJ = np.delete(lon_PNJ, idx_type1 + idx_type2)
        ua_normalPNJ = np.delete(ua_PNJ, idx_type1 + idx_type2)
        uastd_normalPNJ = np.delete(uastd_PNJ, idx_type1 + idx_type2)
        zg_normalPNJ = np.delete(zg_PNJ, idx_type1 + idx_type2)

        uamap_normalPNJ = np.nanmean(ua_data[idx_normal, :, :], axis=0)
        zgmap_normalPNJ = np.nanmean(zg_data[idx_normal, :, :], axis=0)
        x1, x2, x3 = features_composite(uamap_normalPNJ, nc_lat_data, nc_lon_data, zgmap_normalPNJ)

        classmap_normalPNJ, classtime = classify_composite3(x1, x2, x3, clf)

        grp0['uamap_normalPNJ'] = uamap_normalPNJ
        grp0['zgmap_normalPNJ'] = np.nanmean(zg_data[idx_normal, :, :], axis=0)
        grp0['classmap_normalPNJ'] = classmap_normalPNJ

        grp0['Day_normalPNJ'] = Day_normalPNJ
        grp0['ua_normalPNJ'] = ua_normalPNJ
        grp0['uastd_normalPNJ'] = uastd_normalPNJ
        grp0['zg_normalPNJ'] = zg_normalPNJ
        grp0['lat_normalPNJ'] = lat_normalPNJ
        grp0['lon_normalPNJ'] = lon_normalPNJ

    if len(idx_type1) > 0:
        Day_type1event = [Day[i] for i in idx_type1]  # cannot access slides of list
        lat_type1event = [lat_PNJ[i] for i in idx_type1]
        lon_type1event = [lon_PNJ[i] for i in idx_type1]
        ua_type1event = [ua_PNJ[i] for i in idx_type1]
        uastd_type1event = [uastd_PNJ[i] for i in idx_type1]
        zg_type1event = [zg_PNJ[i] for i in idx_type1]

        uamap_type1event = np.nanmean(ua_data[idx_type1, :, :], axis=0)
        zgmap_type1event = np.nanmean(zg_data[idx_type1, :, :], axis=0)
        x1, x2, x3 = features_composite(uamap_type1event, nc_lat_data, nc_lon_data, zgmap_type1event)
        classmap_type1event, classtime = classify_composite3(x1, x2, x3, clf)

        grp0['uamap_type1event'] = uamap_type1event
        grp0['zgmap_type1event'] = np.nanmean(zg_data[idx_type1, :, :], axis=0)
        grp0['classmap_type1event'] = classmap_type1event
        grp0['Day_type1event'] = Day_type1event
        grp0['ua_type1event'] = ua_type1event
        grp0['uastd_type1event'] = uastd_type1event
        grp0['zg_type1event'] = zg_type1event
        grp0['lat_type1event'] = lat_type1event
        grp0['lon_type1event'] = lon_type1event

        print('Day typepnj 1', np.array(Day_type1event) - 1)

    if len(idx_type2) > 0:
        Day_type2event = [Day[i] for i in idx_type2]  # cannot access slides of list
        lat_type2event = [lat_PNJ[i] for i in idx_type2]
        lon_type2event = [lon_PNJ[i] for i in idx_type2]
        ua_type2event = [ua_PNJ[i] for i in idx_type2]
        uastd_type2event = [uastd_PNJ[i] for i in idx_type2]
        zg_type2event = [zg_PNJ[i] for i in idx_type2]

        uamap_type2event = np.nanmean(ua_data[idx_type2, :, :], axis=0)
        zgmap_type2event = np.nanmean(zg_data[idx_type2, :, :], axis=0)

        x1, x2, x3 = features_composite(uamap_type2event, nc_lat_data, nc_lon_data, zgmap_type2event)
        classmap_type2event, classtime = classify_composite3(x1, x2, x3, clf)

        grp0['uamap_type2event'] = uamap_type2event
        grp0['zgmap_type2event'] = np.nanmean(zg_data[idx_type2, :, :], axis=0)
        grp0['classmap_type2event'] = classmap_type2event

        grp0['Day_type2event'] = Day_type2event
        grp0['ua_type2event'] = ua_type2event
        grp0['uastd_type2event'] = uastd_type2event
        grp0['zg_type2event'] = zg_type2event
        grp0['lat_type2event'] = lat_type2event
        grp0['lon_type2event'] = lon_type2event

        print('Day typepnj 2', np.array(Day_type2event) - 1)

    print('Displaced events on dates:')
    datesformat = datesformat[day2classify]
    print(datesformat[idx_type1])
    print('SSW events on dates:')
    print(datesformat[idx_type2])

    f.close()


def features_composite(ua_data, nc_lat_data, nc_lon_data, zg_data):
    scaler = sklearn.preprocessing.RobustScaler()

    x1 = scaler.fit_transform(ua_data)

    lat_matrix = np.vstack([nc_lat_data] * len(nc_lon_data)).T
    x2 = scaler.fit_transform(lat_matrix)

    x3 = zg_data.copy()

    return x1, x2, x3


def detectpnjtype(y_final, lat, ua, ualim):
    covered = np.count_nonzero(y_final)
    limmax = round(0.10 * y_final.shape[0] * y_final.shape[1])

    if covered <= limmax:
        nblobs = count_blobs(y_final)
        if nblobs == 2 and sum(y_final[:, 0]) > 1 and sum(y_final[:, -1]) > 1:
            typepnj = 0
        elif nblobs == 2 and sum(y_final[-1, :]) < 30:
            typepnj = 2
        else:
            typepnj = 0
    else:
        typepnj = 0

    return typepnj


def count_blobs(y_final):
    limmax = round(0.10 * y_final.shape[0] * y_final.shape[1])
    limmin = round(0.005 * y_final.shape[0] * y_final.shape[1])

    blobs = y_final >= 1.0

    labeled_img, nlabels = ndimage.label(blobs)

    labelled = ndimage.label(blobs)
    resh_labelled = labelled[0].reshape(
        (y_final.shape[0],
         y_final.shape[1]))
    props = measure.regionprops(resh_labelled)

    size = np.array([props[i].area for i in range(0, nlabels)])
    idx = np.nonzero((size > limmin) & (size < limmax))

    idx = np.nonzero((size > limmin))
    aux = len(size[idx])

    return aux


def characterization(nc_lat_data, nc_lon_data, grid_type, ua_data, zg_data, y_final, weight_factor, day):
    ua = get_avg_mag(ua_data, y_final, weight_factor, grid_type)
    uastd = standard(ua_data, nc_lat_data)
    zg = get_avg_mag(zg_data, y_final, weight_factor, grid_type)
    lat = get_avg_lat(y_final, nc_lat_data, ua_data, grid_type)
    lon = get_avg_lon(y_final, nc_lon_data, ua_data, 'yes', grid_type)

    return ua, uastd, zg, lat, lon
