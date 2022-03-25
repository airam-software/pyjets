# Classification by expert
# Sources:
# https://matplotlib.org/3.5.1/gallery/widgets/lasso_selector_demo_sgskip.html
# https://matplotlib.org/3.1.0/users/event_handling.html

import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

from expertssw.read_nc import find_file
from expertssw.regiongrowing import segmentimage2

warnings.filterwarnings("ignore")
not4training = []


def pnjexpert(pathh5, pathexpert, expert_level, expert_model, expert_period, month_init, rg_threshold):
    ''' Data for expert'''
    file = find_file('ua', pathh5, expert_level, expert_model, expert_period)
    h5file = h5py.File(pathh5 + file, 'r')
    nc_lat_data = h5file['nc_lat_data'][:]
    nc_lon_data = h5file['nc_lon_data'][:]
    x_data = h5file['x_data_sel'][:]
    grid_type = h5file['grid_type'][()]  # dataset.value has been deprecated
    h5file.close()

    month_vec = np.arange(0, 50)

    # # Initialization
    y_train = np.zeros_like(x_data[month_vec, :, :])
    seedsx = np.zeros((len(month_vec, )), dtype=int)
    seedsy = np.zeros((len(month_vec, )), dtype=int)
    figd = {}
    fign = {}
    figd2 = {}
    fign2 = {}

    # Data for plots
    lon, lat = np.meshgrid(nc_lon_data, nc_lat_data)

    gridx, gridy = lon, lat

    for idx, month in enumerate(month_vec):  # number of months to process

        if idx >= (month_init - 1):
            h5name = 'expertdata' + str(month + 1) + '_pruebas.h5'

            f = h5py.File(pathexpert + expert_period + '/' + expert_model + '/' \
                          + h5name, 'w')  # 'w' truncates any existing file
            f.create_group(expert_model)
            f.close()

            print('Month', month + 1)
            img = x_data[month_vec[idx], :, :] + abs(np.min(x_data[month_vec[idx], :, :]))

            seed_x, seed_y = np.where(img == np.max(img))  # Seed for region growing algorithm
            seedsx[idx] = int(seed_x[0].item())
            seedsy[idx] = int(seed_y[0].item())

            labeled_img = segmentimage2(img, rg_threshold * np.max(img), seed_x, seed_y)

            y_train[idx, :, :] = labeled_img

            # Plots region growing dictionaries
            fig = plt.figure(month)
            figd[str(idx)] = fig  # figure dictionary for figure handle
            fign[str(idx)] = pathexpert + expert_period + '/' + expert_model + '/' \
                             + 'ua_PNJ_month' + str(month_vec[idx] + 1) + '_rg'  # figure dictionary for figure name

            '''
            Plot data
            '''
            print('Press ENTER to accept selected points. \
                 \n Press "d" to delete the selected points \
                 \n Close for NEXT image')

            plot_data(fig, gridx, gridy, x_data[month_vec[idx], :, :],
                      y_train[idx], nc_lon_data, nc_lat_data, idx,
                      seed_x, seed_y, month_vec, fign[str(idx)], 'region growing')

            '''
            Get masked data selected by expert
            '''
            # Lasso selector for expert
            ax = plt.gca()
            pts = ax.scatter(gridx.flatten(), gridy.flatten(), s=0.01)  # transparent

            labeled_img2 = labeled_img.copy()

            idx_msk, idx_mskd = select_regions(ax, pts, fig)
            labeled_img2 = np.reshape(labeled_img2, (len(nc_lat_data) * len(nc_lon_data),))
            labeled_img2[idx_msk] = 1.
            labeled_img2[idx_mskd] = 0.
            labeled_img2 = np.reshape(labeled_img2, (len(nc_lat_data), len(nc_lon_data)))

            # Update y_train with new data
            y_train[idx, :, :] = labeled_img2

            f = h5py.File(pathexpert + expert_period + '/' + expert_model + '/' \
                          + h5name, 'a')  # 'w' truncates any existing file
            grp0 = f[expert_model]
            grp0.create_dataset(name='Month' + str(month + 1), data=y_train[idx, :, :])

            f.close()

            plt.close()

    for idx, month in enumerate(month_vec):
        if idx >= (month_init - 1):
            # Plots expert dictionaries
            fig = plt.figure(month + 200)
            figd2[str(idx)] = fig
            fign2[str(idx)] = pathexpert + expert_period + '/' + expert_model + '/' \
                              + 'ua_PNJ_month' + str(month_vec[idx] + 1) + '_expert'

            '''Save updated figure'''
            plot_data(fig, gridx, gridy, x_data[month_vec[idx], :, :],
                      y_train[idx], nc_lon_data, nc_lat_data, idx,
                      seedsx[idx], seedsy[idx],
                      month_vec, fign2[str(idx)], 'expert selection')

            plt.close()


def accept(event, selector, fig):
    if event.key == 'enter':

        for i in selector.ind:
            selector.masked_data_ind.append(i)
            selector.masked_data.append([selector.xys[i, 0], selector.xys[i, 1]])

        # print('Selected ', len(selector.ind), ' points')
        plt.plot(selector.xys[selector.ind, 0],
                 selector.xys[selector.ind, 1], 'r*')
        fig.canvas.draw()

    if event.key == 'd':
        for i in selector.ind:
            selector.delete_data_ind.append(i)
            selector.delete_data.append([selector.xys[i, 0], selector.xys[i, 1]])

        # print('Deleted ', len(selector.ind), ' points')
        plt.plot(selector.xys[selector.ind, 0],
                 selector.xys[selector.ind, 1], 'k*')
        fig.canvas.draw()

    # if event.key == 'q':
    #     print('Disconnecting selector')
    #     selector.disconnect()


def select_regions(ax, pts, fig):
    selector = SelectFromCollection(ax, pts)

    # Select the region
    arg1 = selector
    arg2 = fig

    fig.canvas.mpl_connect('key_press_event',
                           lambda event: accept(event, arg1,
                                                arg2))

    plt.show(block=True)  # spyder workaround
    plt.show(block=False)  # spyder workaround

    return selector.masked_data_ind, selector.delete_data_ind


class SelectFromCollection(object):
    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

        # Defined by me
        self.masked_data_ind = []
        self.masked_data = []
        self.delete_data_ind = []
        self.delete_data = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        # self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        # self.canvas.draw_idle()


def plot_data(fig, gridx, gridy, origdata, data4train, nc_lon_data,
              nc_lat_data, idx, seed_y, seed_x, month_vec, figname, llabel):
    # Plot the data

    plt.contourf(gridx, gridy, origdata,
                 levels=np.linspace(0, 100, 11),
                 vmin=0, vmax=100,
                 extend='min')
    cb = plt.colorbar()
    cb.set_label('u [m/s]')

    if llabel == 'region growing':
        plt.plot(nc_lon_data[seed_x],
                 nc_lat_data[seed_y], 'k*', label=llabel)
    else:
        plt.plot(nc_lon_data[seed_x],
                 nc_lat_data[seed_y], 'k', label=llabel)
    plt.contour(gridx, gridy, data4train, [0], colors='k')

    plt.ylabel('$\phi$ [' + u'\N{DEGREE SIGN}' + ']')
    plt.xlabel('$\lambda$ [' + u'\N{DEGREE SIGN}' + ']')
    plt.grid()
    plt.legend(loc='lower left')

    plt.title('Month ' + str(month_vec[idx] + 1))

    size = fig.get_size_inches()
    fig.set_size_inches(size[0], size[1])

    plt.savefig(figname + '.png', format='png', bbox_inches='tight')
