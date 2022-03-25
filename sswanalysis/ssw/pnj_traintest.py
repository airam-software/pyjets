# Train test dataset with expert input

import pickle

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing

from ssw.classifiers import fulldataset
from ssw.classifiers import holdoutstraintest
from ssw.read_nc import find_file

plot_opt = 0

dates = ['1958-01-30',
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

dates_ssw = [
    '1958-01-30',
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

dates_displacements = [
    '1960-01-14',
    '1960-01-15',
    '1960-01-16',
    '1965-12-08',
    '1965-12-09',
    '1965-12-10',
    '1965-12-11',
    '1965-12-12',
    '1965-12-13',
    '1965-12-14',
    '1965-12-15',
    '1966-02-21',
    '1966-02-23',
    '1966-02-24',
    '1965-12-16',
    '1969-12-31',
    '1969-01-01',
    '1969-01-02',
    '1973-01-30',
    '1973-01-31',
    '1977-01-08',
    '1977-01-09',
    '1977-01-10',
    '1980-02-27',
    '1980-02-28',
    '1980-02-29',
    '1981-02-03',
    '1981-02-04',
    '1981-02-05',
    '1984-02-23',
    '1984-02-24',
    '1984-02-25',
    '1987-12-08',
    '1987-01-22',
    '1987-01-23',
    '1987-01-24',
    '1988-12-14',
    '1988-12-15',
    '1988-12-16',
    '1988-03-13',
    '1999-02-25',
    '2001-12-30',
    '2001-12-31',
    '2002-01-01',
    '2002-01-02'
]


def pnj_traintest(file_path_h5, file_path_store, training_level, option,
                  classifiers, projectpath, selclass, pathplot):
    month_traintest1 = np.arange(0, 39)
    idx_nottrain1 = [3, 4, 5, 6, 7, 8, 15, 16, 18, 19, 20, 27, 28, 29, 30, 36]
    # idx_nottrain1 = [x - 1 for x in idx_nottrain1]
    month_traintest1 = np.delete(month_traintest1, idx_nottrain1)  # specifically trained for ssw

    month_traintest2 = np.arange(0, 50)  # PNJ normal
    idx_nottrain2 = [1, 30, 31, 46]
    month_traintest2 = np.delete(month_traintest2, idx_nottrain2)  # polar vortex is broken in this month

    month_traintest3 = np.arange(0, 45)
    idx_nottrain3 = [0, 1, 15, 16, 17, 23, 24, 26, 27, 37, 38, 39]
    month_traintest3 = np.delete(month_traintest3, idx_nottrain3)  # polar vortex is broken in this month

    month_traintest = np.concatenate((month_traintest1, month_traintest2,
                                      month_traintest3))  # Total traintest months

    nmonths = len(month_traintest)

    '''y initialization'''
    y_train = np.zeros((nmonths, 192, 288))  # trained

    for idx, month in enumerate(month_traintest1):  # number of months in file
        h5name = projectpath + '/results/expert/jra/jra/expertdatassw' + str(month + 1) + '.h5'
        h5file = h5py.File(h5name, 'r')
        labeled_img = h5file['jra']['day' + str(month + 1)][:]
        y_train[idx, :, :] = labeled_img

        h5file.close()

    for idx, month in enumerate(month_traintest2):  # number of months in file
        h5name = projectpath + '/results/expert/jra/jra/expertdata' + str(month + 1) + '.h5'
        h5file = h5py.File(h5name, 'r')
        labeled_img = h5file['jra']['Month' + str(month + 1)][:]
        y_train[len(month_traintest1) + idx, :, :] = labeled_img
        h5file.close()

    for idx, month in enumerate(month_traintest3):  # number of months in file
        h5name = projectpath + '/results/expert/jra/jra/expertdatadispl' + str(month + 1) + '.h5'
        h5file = h5py.File(h5name, 'r')
        labeled_img = h5file['jra']['day' + str(month + 1)][:]
        y_train[len(month_traintest1) + len(month_traintest2) + idx, :, :] = labeled_img
        h5file.close()

    # Read original data to find classification features
    file = find_file('ua', file_path_h5 + '/sswtrain/', training_level, 'jra', 'jra')
    h5file = h5py.File(file_path_h5 + '/sswtrain/' + file, 'r')
    ua_data1 = h5file['x_data_sel'][:]
    ua_data1 = ua_data1[month_traintest1, :, :]
    nc_lat_data = h5file['nc_lat_data'][:]
    nc_lon_data = h5file['nc_lon_data'][:]
    h5file.close()

    file = find_file('zg', file_path_h5 + '/sswtrain/', training_level, 'jra', 'jra')
    h5file = h5py.File(file_path_h5 + '/sswtrain/' + file, 'r')
    zg_data1 = h5file['x_data_sel'][:]
    zg_data1 = zg_data1[month_traintest1, :, :]
    h5file.close()

    file = find_file('ua', file_path_h5 + '/h5remapped/', training_level, 'jra', 'jra')
    h5file = h5py.File(file_path_h5 + '/h5remapped/' + file, 'r')
    ua_data2 = h5file['x_data_sel'][:]
    ua_data2 = ua_data2[month_traintest2, :, :]
    h5file.close()

    file = find_file('zg', file_path_h5 + '/h5remapped/', training_level, 'jra', 'jra')
    h5file = h5py.File(file_path_h5 + '/h5remapped/' + file, 'r')
    zg_data2 = h5file['x_data_sel'][:]
    zg_data2 = zg_data2[month_traintest2, :, :]
    h5file.close()

    file = find_file('ua', file_path_h5 + '/displtrain/', training_level, 'jra', 'jra')
    h5file = h5py.File(file_path_h5 + '/displtrain/' + file, 'r')
    ua_data3 = h5file['x_data_sel'][:]
    ua_data3 = ua_data3[month_traintest3, :, :]
    h5file.close()

    file = find_file('zg', file_path_h5 + '/displtrain/', training_level, 'jra', 'jra')
    h5file = h5py.File(file_path_h5 + '/displtrain/' + file, 'r')
    zg_data3 = h5file['x_data_sel'][:]
    zg_data3 = zg_data3[month_traintest3, :, :]
    h5file.close()

    ua_data = np.concatenate((ua_data1, ua_data2, ua_data3), axis=0)
    zg_data = np.concatenate((zg_data1, zg_data2, zg_data3), axis=0)

    seedsx = []
    seedsy = []

    for idx, month in enumerate(month_traintest):  # number of months in file

        img = ua_data[idx, :, :] + abs(np.min(ua_data[idx, :, :]))
        seed_x, seed_y = np.where(img == np.max(img))
        seedsx.append(seed_x[0].item())
        seedsy.append(seed_y[0].item())

    # Features
    scaler = sklearn.preprocessing.RobustScaler()
    x1 = np.zeros_like(ua_data)
    for month in np.arange(0, nmonths):
        x1[month, :, :] = scaler.fit_transform(ua_data[month, :, :])

    lat_matrix = np.vstack([nc_lat_data] * len(nc_lon_data)).T
    x2 = np.array([lat_matrix] * nmonths)
    for month in np.arange(0, nmonths):
        x2[month, :, :] = scaler.fit_transform(x2[month, :, :])

    x3 = zg_data.copy()

    X = np.vstack((x1.flatten(), x2.flatten(), x3.flatten())).T.tolist()

    y = y_train.flatten().T.tolist()

    '''
    Train-test the dataset
    '''

    if option == 'compare':

        for clf in classifiers:
            traintime, classtime, acc, _ = holdoutstraintest(X, y, clf)
            print('----')
            print(clf)
            print('Test accuracy: %.3f ' % acc)
            print('Elapsed training time: %.1f ' % traintime, traintime)
            print('Elapsed testing time: %.1f ' % classtime, classtime)

    if option == 'classify':
        traintime, classtime, y_final, clf = \
            fulldataset(X, y, selclass)  # Selected classifier after comparison
        y_final = np.reshape(y_final, [nmonths, len(nc_lat_data), len(nc_lon_data)])
        pickle.dump(clf, open(file_path_store + 'classifier', 'wb'))
        print('Elapsed training time: %.1f ' % traintime)
        print('Elapsed classification time: %.1f ' % classtime)

        if plot_opt == 1:
            '''Plots'''
            lon, lat = np.meshgrid(nc_lon_data, nc_lat_data)
            gridx, gridy = lon, lat
            for idx, month in enumerate(np.arange(0, nmonths)):  # Number of months in file
                print('Plotting ' + str(month))
                plt.figure()

                plt.subplot(121)
                plt.contourf(gridx, gridy, ua_data[idx, :, :], cmap=cm.coolwarm,
                             levels=np.linspace(0, 100, 11),
                             vmin=0, vmax=100,
                             extend='min')
                plt.ylabel('$\phi$ [' + u'\N{DEGREE SIGN}' + ']')
                plt.xlabel('$\lambda$ [' + u'\N{DEGREE SIGN}' + ']')
                plt.grid()
                cb = plt.colorbar()
                cb.set_label('u [m/s]')

                plt.subplot(122)
                plt.contourf(gridx, gridy, zg_data[idx, :, :], cmap=cm.coolwarm,
                             levels=np.linspace(28800, 31900, 11),
                             vmin=28800, vmax=31900,
                             extend='min')
                plt.ylabel('$\phi$ [' + u'\N{DEGREE SIGN}' + ']')
                plt.xlabel('$\lambda$ [' + u'\N{DEGREE SIGN}' + ']')
                plt.grid()
                cb = plt.colorbar()
                cb.set_label('zg [m]')

                fig = plt.gcf()
                size = fig.get_size_inches()
                fig.set_size_inches(2 * size[0], size[1])

                '''Save figure'''
                fign = pathplot + '/' + 'PNJ_' + str(month + 1) + '_a'
                plt.savefig(fign + '.png', format='png', bbox_inches='tight')

                plt.close()

                plt.figure()

                plt.subplot(121)

                plt.contourf(gridx, gridy, ua_data[idx, :, :], cmap=cm.coolwarm,
                             levels=np.linspace(0, 100, 11),
                             vmin=0, vmax=100,
                             extend='min')
                plt.ylabel('$\phi$ [' + u'\N{DEGREE SIGN}' + ']')
                plt.xlabel('$\lambda$ [' + u'\N{DEGREE SIGN}' + ']')
                plt.grid()
                cb = plt.colorbar()
                cb.set_label('u [m/s]')

                plt.scatter(gridx, gridy, y_final[idx, :, :], c='yellow', label='classifier')

                plt.contour(gridx, gridy, y_train[idx], [0], colors='k', label='ground truth')

                plt.legend(loc='lower left')

                plt.subplot(122)
                plt.contourf(gridx, gridy, zg_data[idx, :, :], cmap=cm.coolwarm,
                             levels=np.linspace(28800, 31900, 11),
                             vmin=28800, vmax=31900,
                             extend='min')
                plt.ylabel('$\phi$ [' + u'\N{DEGREE SIGN}' + ']')
                plt.xlabel('$\lambda$ [' + u'\N{DEGREE SIGN}' + ']')
                plt.grid()
                cb = plt.colorbar()
                cb.set_label('zg [m]')

                plt.scatter(gridx, gridy, y_final[idx, :, :], c='yellow', label='classifier')

                plt.contour(gridx, gridy, y_train[idx], [0], colors='k', label='ground truth')

                plt.legend(loc='lower left')

                fig = plt.gcf()
                size = fig.get_size_inches()
                fig.set_size_inches(2 * size[0], size[1])

                '''Save figure'''
                fign = pathplot + '/' 'PNJ_' + str(month + 1) + '_b'
                plt.savefig(fign + '.png', format='png', bbox_inches='tight')
                plt.close()
