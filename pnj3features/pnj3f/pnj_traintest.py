# Train test dataset with expert input

import pickle

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pnj3f.classifiers import holdoutstraintest, trainfull
from pnj3f.read_nc import find_file
from pnj3f.regiongrowing import segmentimage2


def pnj_traintest(file_path_h5, file_path_store, training_level, option,
                  classifiers, rg_threshold, projectpath, selclass, pathplot):
    month_traintest1 = np.arange(0, 45)
    idx_nottrain1 = [9, 11, 17, 23, 28]
    month_traintest1 = np.delete(month_traintest1, idx_nottrain1)  # polar vortex is broken in this month

    month_traintest2 = np.arange(0, 50)
    idx_nottrain2 = [3, 29, 40, 47]
    month_traintest2 = np.delete(month_traintest2, idx_nottrain2)  # polar vortex is broken in this month

    month_traintest3 = np.arange(0, 50)
    idx_nottrain3 = [0, 5, 10, 30, 34, 43]
    month_traintest3 = np.delete(month_traintest3, idx_nottrain3)  # polar vortex is broken in this month

    month_traintest4 = np.arange(0, 50)
    idx_nottrain4 = [1, 30, 31, 46]
    month_traintest4 = np.delete(month_traintest4, idx_nottrain4)  # polar vortex is broken in this month

    month_traintest = np.concatenate((month_traintest1, month_traintest2,
                                      month_traintest3, month_traintest4))  # Total traintest months
    nmonths = len(month_traintest)

    '''y initialization'''
    y_train = np.zeros((nmonths, 192, 288))  # trained
    y_rg = np.zeros((nmonths, 192, 288))  # region growing

    for idx, month in enumerate(month_traintest1):  # number of months in file
        h5name = projectpath + '/results/expert/piControl/CESM2-WACCM/expertdata' + str(month + 1) + '.h5'
        h5file = h5py.File(h5name, 'r')
        labeled_img = h5file['CESM2-WACCM']['Month' + str(month + 1)][:]  # Expert image
        y_train[idx, :, :] = labeled_img
        h5file.close()

    for idx, month in enumerate(month_traintest2):  # number of months in file
        h5name = projectpath + '/results/expert/piControl/CanESM5/expertdata' + str(month + 1) + '.h5'
        h5file = h5py.File(h5name, 'r')
        labeled_img = h5file['CanESM5']['Month' + str(month + 1)][:]
        y_train[len(month_traintest1) + idx, :, :] = labeled_img
        h5file.close()

    for idx, month in enumerate(month_traintest3):  # number of months in file
        h5name = projectpath + '/results/expert/piControl/IPSL-CM6A-LR/expertdata' + str(month + 1) + '.h5'
        h5file = h5py.File(h5name, 'r')
        labeled_img = h5file['IPSL-CM6A-LR']['Month' + str(month + 1)][:]
        y_train[len(month_traintest1) + len(month_traintest2) + idx, :, :] = labeled_img
        h5file.close()

    for idx, month in enumerate(month_traintest4):  # number of months in file
        h5name = projectpath + '/results/expert/jra/jra/expertdata' + str(month + 1) + '.h5'
        h5file = h5py.File(h5name, 'r')
        labeled_img = h5file['jra']['Month' + str(month + 1)][:]
        y_train[len(month_traintest1) + len(month_traintest2) + len(month_traintest3) + idx, :, :] = labeled_img
        h5file.close()

    # Read original data to find classification features

    file = find_file('ua', file_path_h5, training_level, 'CESM2-WACCM', 'piControl')
    h5file = h5py.File(file_path_h5 + file, 'r')
    nc_lat_data = h5file['nc_lat_data'][:]
    nc_lon_data = h5file['nc_lon_data'][:]
    ua_data1 = h5file['x_data_sel'][:]
    ua_data1 = ua_data1[month_traintest1, :, :]
    h5file.close()

    file = find_file('zg', file_path_h5, training_level, 'CESM2-WACCM', 'piControl')
    h5file = h5py.File(file_path_h5 + file, 'r')
    zg_data1 = h5file['x_data_sel'][:]
    zg_data1 = zg_data1[month_traintest1, :, :]
    h5file.close()

    file = find_file('ua', file_path_h5, training_level, 'CanESM5', 'piControl')
    h5file = h5py.File(file_path_h5 + file, 'r')
    ua_data2 = h5file['x_data_sel'][:]
    ua_data2 = ua_data2[month_traintest2, :, :]
    h5file.close()

    file = find_file('zg', file_path_h5, training_level, 'CanESM5', 'piControl')
    h5file = h5py.File(file_path_h5 + file, 'r')
    zg_data2 = h5file['x_data_sel'][:]
    zg_data2 = zg_data2[month_traintest2, :, :]
    h5file.close()

    file = find_file('ua', file_path_h5, training_level, 'IPSL-CM6A-LR', 'piControl')
    h5file = h5py.File(file_path_h5 + file, 'r')
    ua_data3 = h5file['x_data_sel'][:]
    ua_data3 = ua_data3[month_traintest3, :, :]
    h5file.close()

    file = find_file('zg', file_path_h5, training_level, 'IPSL-CM6A-LR', 'piControl')
    h5file = h5py.File(file_path_h5 + file, 'r')
    zg_data3 = h5file['x_data_sel'][:]
    zg_data3 = zg_data3[month_traintest3, :, :]
    h5file.close()

    file = find_file('ua', file_path_h5, training_level, 'jra', 'jra')
    h5file = h5py.File(file_path_h5 + file, 'r')
    ua_data4 = h5file['x_data_sel'][:]
    ua_data4 = ua_data4[month_traintest4, :, :]
    h5file.close()

    file = find_file('zg', file_path_h5, training_level, 'jra', 'jra')
    h5file = h5py.File(file_path_h5 + file, 'r')
    zg_data4 = h5file['x_data_sel'][:]
    zg_data4 = zg_data4[month_traintest4, :, :]
    h5file.close()

    ua_data = np.concatenate((ua_data1, ua_data2, ua_data3, ua_data4), axis=0)
    zg_data = np.concatenate((zg_data1, zg_data2, zg_data3, zg_data4), axis=0)

    seedsx = []
    seedsy = []

    for idx, month in enumerate(month_traintest):  # number of months in file

        img = ua_data[idx, :, :] + abs(np.min(ua_data[idx, :, :]))
        seed_x, seed_y = np.where(img == np.max(img))
        seedsx.append(seed_x[0].item())
        seedsy.append(seed_y[0].item())

        labeled_img2 = segmentimage2(img, rg_threshold * np.max(img), seed_x,
                                     seed_y)  # Previous region growing algorithm just for plotting
        y_rg[idx, :, :] = labeled_img2

    scaler = MinMaxScaler()
    x1 = np.zeros_like(ua_data)
    for month in np.arange(0, nmonths):
        x1[month, :, :] = scaler.fit_transform(ua_data[month, :, :])

    lat_matrix = np.vstack([nc_lat_data] * len(nc_lon_data)).T
    x2 = np.array([lat_matrix] * nmonths)
    for month in np.arange(0, nmonths):
        x2[month, :, :] = scaler.fit_transform(x2[month, :, :])

    x3 = np.zeros_like(zg_data)
    for month in np.arange(0, nmonths):
        x3[month, :, :] = scaler.fit_transform(zg_data[month, :, :])

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
            trainfull(X, y, selclass)  # Selected classifier after comparison
        y_final = np.reshape(y_final, [nmonths, len(nc_lat_data), len(nc_lon_data)])
        pickle.dump(clf, open(file_path_store + 'classifier', 'wb'))
        print('Elapsed training time: %.1f ' % traintime)
        print('Elapsed classification time: %.1f ' % classtime)

        month_plot = np.arange(0, 5)  # plot some test figures

        '''Plots'''
        lon, lat = np.meshgrid(nc_lon_data, nc_lat_data)
        gridx, gridy = lon, lat
        for idx, month in enumerate(month_traintest):  # Number of months in file
            if month in month_plot:
                plt.figure()

                plt.contourf(gridx, gridy, ua_data[idx, :, :], cmap=cm.coolwarm,
                             levels=np.linspace(0, 100, 11),
                             vmin=0, vmax=100,
                             extend='min')
                plt.ylabel('$\phi$ [' + u'\N{DEGREE SIGN}' + ']')
                plt.xlabel('$\lambda$ [' + u'\N{DEGREE SIGN}' + ']')
                plt.grid()
                cb = plt.colorbar()
                cb.set_label('u [m/s]')

                fig = plt.gcf()
                size = fig.get_size_inches()
                fig.set_size_inches(size[0], size[1])

                '''Save figure'''
                fign = pathplot + '/' + 'Traintest_ua_PNJ_' + str(month + 1) + '_a'
                plt.savefig(fign + '.png', format='png', bbox_inches='tight')

                plt.close()

                plt.figure()
                plt.contourf(gridx, gridy, ua_data[idx, :, :], cmap=cm.coolwarm,
                             levels=np.linspace(0, 100, 11),
                             vmin=0, vmax=100,
                             extend='min')
                plt.ylabel('$\phi$ [' + u'\N{DEGREE SIGN}' + ']')
                plt.xlabel('$\lambda$ [' + u'\N{DEGREE SIGN}' + ']')
                plt.grid()
                cb = plt.colorbar()
                cb.set_label('u [m/s]')

                plt.plot(nc_lon_data[seedsy[idx]],
                         nc_lat_data[seedsx[idx]], 'ys', label='classifier')
                plt.scatter(gridx, gridy, y_final[idx, :, :], c='yellow')

                plt.plot(nc_lon_data[seedsy[idx]],
                         nc_lat_data[seedsx[idx]], 'm-', label='region growing')
                plt.contour(gridx, gridy, y_rg[idx], [0], colors='m')

                plt.plot(nc_lon_data[seedsy[idx]],
                         nc_lat_data[seedsx[idx]], 'k-', label='expert')
                plt.contour(gridx, gridy, y_train[idx], [0], colors='k')

                plt.plot(nc_lon_data[seedsy[idx]],
                         nc_lat_data[seedsx[idx]], 'ro', label='max u')

                plt.legend(loc='lower left')

                fig = plt.gcf()
                size = fig.get_size_inches()
                fig.set_size_inches(size[0], size[1])

                '''Save figure'''
                fign = pathplot + '/' 'Traintest_ua_PNJ_' + str(month + 1) + '_b'
                plt.savefig(fign + '.png', format='png', bbox_inches='tight')

                plt.close()
