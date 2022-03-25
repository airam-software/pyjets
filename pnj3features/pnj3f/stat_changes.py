# Identify significant changes with climate change

import h5py
import numpy as np
import scipy


def changesh5():
    model_vec = ['CESM2-WACCM', 'CanESM5', 'IPSL-CM6A-LR']

    for i in np.arange(0, len(model_vec)):
        period_vec1 = 'piControl'
        period_vec2 = 'abrupt-4xCO2'

        f1 = h5py.File('./results/class/' + period_vec1 + '/' + model_vec[i] + '/PNJ.h5', 'r')
        f2 = h5py.File('./results/class/' + period_vec2 + '/' + model_vec[i] + '/PNJ.h5', 'r')

        pnjtype = 'normalPNJ'
        ua1, lat1, lon1, uasdt1 = get_variables(f1, pnjtype, model_vec[i])
        ua2, lat2, lon2, uastd2 = get_variables(f2, pnjtype, model_vec[i])

        # perform_ttest(ua1, ua2, model_vec[i], 'ua')
        # perform_ttest(ua1, ua2, model_vec[i], 'lat')
        # perform_ttest(ua1, ua2, model_vec[i], 'lon')
        perform_ttest(ua1, ua2, model_vec[i], 'uastd')

        f1.close()
        f2.close()


def perform_ttest(sample1, sample2, model, label):
    ttest, pval = scipy.stats.ttest_ind(sample1, sample2)

    if pval > 0.05:
        print('We accept null hypothesis', label, model, 'p-value', pval)
    else:
        print('We reject null hypothesis', label, model)


def get_variables(f, pnjtype, model):
    ua = f[model]['ua_' + pnjtype][:]
    lat = f[model]['lat_' + pnjtype][:]
    lon = f[model]['lon_' + pnjtype][:]
    uastd = f[model]['uastd_' + pnjtype][:]

    return ua, lat, lon, uastd
