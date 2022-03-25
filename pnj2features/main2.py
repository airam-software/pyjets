# Plot discarded images by expert

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from pnj2f.regiongrowing import segmentimage2

rg_threshold = 0.75

h5file = h5py.File('TBD')

idx_nottrain1 = [1, 30, 31, 46]
nc_lat_data = h5file['nc_lat_data'][:]
nc_lon_data = h5file['nc_lon_data'][:]
ua_data = h5file['x_data_sel'][:]
h5file.close()

lon, lat = np.meshgrid(nc_lon_data, nc_lat_data[nc_lat_data > 0])
gridx, gridy = lon, lat
ua_dataplot = ua_data[:, nc_lat_data > 0, :]

seedsx = []
seedsy = []

for idx, month in enumerate(idx_nottrain1):
    img = ua_data[idx, :, :] + abs(np.min(ua_data[idx, :, :]))
    seed_x, seed_y = np.where(img == np.max(img))
    seedsx.append(seed_x[0].item())
    seedsy.append(seed_y[0].item())

    labeled_img2 = segmentimage2(img, rg_threshold * np.max(img), seed_x,
                                 seed_y)  # Previous region growing algorithm just for plotting
    yrg = labeled_img2[nc_lat_data > 0, :]

    plt.figure()

    plt.contourf(gridx, gridy, ua_dataplot[idx, :, :], cmap=cm.coolwarm,
                 levels=np.linspace(0, 60, 11),
                 vmin=0, vmax=60,
                 extend='both')
    plt.ylabel('$\phi$ [' + u'\N{DEGREE SIGN}' + ']')
    plt.xlabel('$\lambda$ [' + u'\N{DEGREE SIGN}' + ']')
    plt.grid()
    cb = plt.colorbar()
    cb.set_label('u [m/s]')

    fig = plt.gcf()
    size = fig.get_size_inches()
    fig.set_size_inches(size[0], size[1])

    '''Save figure'''
    fign = './plots/traintest/' + 'Discarded_ua_PNJ_' + str(month + 1) + '_a'
    plt.savefig(fign + '.png', format='png', bbox_inches='tight')

    plt.close()

    plt.figure()
    plt.contourf(gridx, gridy, ua_dataplot[idx, :, :], cmap=cm.coolwarm,
                 levels=np.linspace(0, 60, 11),
                 vmin=0, vmax=60,
                 extend='both')
    plt.ylabel('$\phi$ [' + u'\N{DEGREE SIGN}' + ']')
    plt.xlabel('$\lambda$ [' + u'\N{DEGREE SIGN}' + ']')
    plt.grid()
    cb = plt.colorbar()
    cb.set_label('u [m/s]')

    plt.plot(nc_lon_data[seedsy[idx]],
             nc_lat_data[seedsx[idx]], 'm-', label='region growing')
    plt.contour(gridx, gridy, yrg, [0], colors='m')

    plt.plot(nc_lon_data[seedsy[idx]],
             nc_lat_data[seedsx[idx]], 'ro', label='max u')

    plt.legend(loc='lower left')

    fig = plt.gcf()
    size = fig.get_size_inches()
    fig.set_size_inches(size[0], size[1])

    '''Save figure'''
    fign = './plots/traintest/' + 'Discarded_ua_PNJ_' + str(month + 1) + '_b'
    plt.savefig(fign + '.png', format='png', bbox_inches='tight')

    plt.close()
