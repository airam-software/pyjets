# Print results to table format

import h5py
import numpy as np
import pandas as pd


def tablesh5(pathclass, pathresults, level):
    for opt in [0]:
        if opt == 0:
            period_vec = ['jra', 'piControl']
            model_vec = ['jra', 'CESM2-WACCM']
        elif opt == 1:
            period_vec = ['jra', 'piControl', 'piControl', 'piControl']
            model_vec = ['jra', 'CESM2-WACCM', 'CanESM5', 'IPSL-CM6A-LR']
        else:
            period_vec = ['abrupt-4xCO2', 'abrupt-4xCO2', 'abrupt-4xCO2']
            model_vec = ['CESM2-WACCM', 'CanESM5', 'IPSL-CM6A-LR']

        ''' Write dataframe with results '''
        pnjtypes = ['normalPNJ', 'type1event', 'type2event']

        txtf = open(pathresults + 'summary' + level + 'hpa.txt', 'w')
        for pnjtype in pnjtypes:

            df1 = pd.DataFrame(
                columns=['model', 'period',
                         'ua_mean', 'ua_std',
                         'lat_mean', 'lat_std',
                         'lon_mean', 'lon_std'])

            df2 = pd.DataFrame(
                columns=['model',
                         'period',
                         'freq', 'freq_pct',
                         'ua_mean'])

            df3 = pd.DataFrame(
                columns=['model', 'period',
                         'uastd_mean',
                         'uastd_std'])

            for i in np.arange(0, len(model_vec)):

                f = h5py.File(pathclass + period_vec[i] + '/' + model_vec[i] +
                              '/PNJ' + level + '.h5', 'r')

                ''' Total length '''
                total = 0
                grp0 = f[model_vec[i]]
                if 'Month_normalPNJ' in grp0.keys():
                    total += len(f[model_vec[i]]['Month' + '_' + 'normalPNJ'])
                if 'Month_type2event' in grp0.keys():
                    total += len(f[model_vec[i]]['Month' + '_' + 'type2event'])
                if 'Month_type1event' in grp0.keys():
                    total += len(f[model_vec[i]]['Month' + '_' + 'type1event'])

                if 'Month_' + pnjtype in grp0.keys():
                    freq, freq_pct, ua_mean, ua_std, uastd_mean, uastd_std, lat_mean, lat_std, lon_mean, lon_std, \
                    zg_mean, zg_std = get_variables(f, pnjtype, model_vec[i], total)

                    to_append1 = [model_vec[i],
                                  period_vec[i],
                                  ua_mean, ua_std,
                                  lat_mean, lat_std,
                                  lon_mean, lon_std]

                    to_append2 = [model_vec[i],
                                  period_vec[i],
                                  freq, freq_pct,
                                  ua_mean]

                    to_append3 = [model_vec[i],
                                  period_vec[i],
                                  uastd_mean,
                                  uastd_std]

                    a_series = pd.Series(to_append1, index=df1.columns)
                    df1 = df1.append(a_series, ignore_index=True).replace('\\midrule', '\\hline')

                    a_series = pd.Series(to_append2, index=df2.columns)
                    df2 = df2.append(a_series, ignore_index=True).replace('\\midrule', '\\hline')

                    a_series = pd.Series(to_append3, index=df3.columns)
                    df3 = df3.append(a_series, ignore_index=True).replace('\\midrule', '\\hline')

            print('-------')
            print(pnjtype)
            txtf.write('-------\n')
            txtf.write(pnjtype + '\n')
            if not df1.empty and pnjtype == 'normalPNJ':
                txtf.write(df1.to_latex(float_format="%.2f", index=False))
                txtf.write('\n')
                print(df1)
                print('\n')
            if not df2.empty:
                txtf.write(df2.to_latex(float_format="%.2f", index=False))
                txtf.write('\n')
                print(df2)
                print('\n')
            if not df3.empty:
                txtf.write(df3.to_latex(float_format="%.2f", index=False))
                txtf.write('\n')
                print(df3)
                print('\n')

        txtf.close()


def write_txt(f, ftxt, pnjtype, var, model, total):
    aux = f[model][var + '_' + pnjtype]

    if var == 'Month':
        ftxt.write(pnjtype + ' ' + 'Freq' + ' ' + str(len(aux[:])) + '\n')
        ftxt.write(pnjtype + ' ' + 'Freq [%]' + ' ' + str(len(aux[:]) / total) + '\n')
        ftxt.write('\n')
    else:
        ftxt.write(var + '_' + pnjtype + ' ' + 'Mean' + ' ' + str(np.nanmean(aux[:])) + '\n')
        ftxt.write(var + '_' + pnjtype + ' ' + 'Std' + ' ' + str(np.nanstd(aux[:])) + '\n')
        ftxt.write('\n')


def get_variables(f, pnjtype, model, total):
    freq = str(len(f[model]['ua_' + pnjtype][:]))
    freq_pct = '{:5.2f}'.format(len(f[model]['ua_' + pnjtype][:]) / total * 100)
    ua_mean = np.nanmean(f[model]['ua_' + pnjtype][:])
    ua_std = np.nanstd(f[model]['ua_' + pnjtype][:])
    uastd_mean = np.nanmean(f[model]['uastd_' + pnjtype][:])
    uastd_std = np.nanstd(f[model]['uastd_' + pnjtype][:])
    zg_mean = np.nanmean(f[model]['zg_' + pnjtype][:])
    zg_std = np.nanstd(f[model]['zg_' + pnjtype][:])
    lat_mean = np.nanmean(f[model]['lat_' + pnjtype][:])
    lat_std = np.nanstd(f[model]['lat_' + pnjtype][:])
    lon_mean = np.nanmean(f[model]['lon_' + pnjtype][:])
    lon_std = np.nanstd(f[model]['lon_' + pnjtype][:])

    return freq, freq_pct, ua_mean, ua_std, uastd_mean, uastd_std, lat_mean, lat_std, lon_mean, lon_std, \
           zg_mean, zg_std
