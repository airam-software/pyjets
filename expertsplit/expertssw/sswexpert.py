# Classification by expert for ssw events
# Sources:
# https://matplotlib.org/3.5.1/gallery/widgets/lasso_selector_demo_sgskip.html
# https://matplotlib.org/3.1.0/users/event_handling.html

import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

warnings.filterwarnings("ignore")
not4training = []

dates = ['1958-01-30',
         '1958-01-31',
         '1958-02-01',
         '1960-01-14',  # displacement
         '1960-01-15',  # displacement
         '1960-01-16',  # displacement
         '1963-01-27',
         '1963-01-28',
         '1963-01-29',
         '1965-12-08',  # displacement
         '1965-12-09',  # displacement
         '1965-12-10',  # displacement
         '1965-12-11',  # displacement
         '1965-12-12',  # displacement
         '1965-12-13',  # displacement
         '1965-12-14',  # displacement
         '1965-12-15',  # displacement
         '1965-12-16',  # displacement
         '1966-02-21',
         '1966-02-23',
         '1966-02-24',
         '1968-01-06',
         '1968-01-07',
         '1968-01-08',
         '1969-12-31',  # displacement
         '1969-01-01',  # displacement
         '1969-01-02',  # displacement
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
         '1980-02-27',  # displacement
         '1980-02-28',  # displacement
         '1980-02-29',  # displacement
         '1981-02-03',  # displacement
         '1981-02-04',  # displacement
         '1981-02-05',  # displacement
         '1984-12-31',
         '1984-02-23',  # displacement
         '1984-02-24',  # displacement
         '1984-02-25',  # displacement
         '1985-01-01',
         '1985-01-02',
         '1987-12-06',
         '1987-12-07',
         '1987-12-08',
         '1987-01-22',  # displacement
         '1987-01-23',  # displacement
         '1987-01-24',  # displacement
         '1988-03-13',
         '1988-03-14',
         '1988-03-15',
         '1989-02-20',
         '1989-02-21',
         '1989-02-22',
         '1988-12-14',  # displacement
         '1988-12-15',  # displacement
         '1988-12-16',  # displacement
         '1999-02-25',
         '1999-02-26',
         '1999-02-27',
         '2001-12-30',  # displacement
         '2001-12-31',  # displacement
         '2002-01-01',  # displacement
         '2002-01-02'  # displacement
         ]


def sswexpert(day_init):
    ''' Data for expert'''

    expert_model = 'jra'
    expert_period = 'jra'
    pathexpert = './results/'

    h5file = h5py.File('./zg010_jra-55_1958-2015_dm_remapped.h5', 'r')
    nc_lat_data = h5file['nc_lat_data'][:]
    nc_lon_data = h5file['nc_lon_data'][:]
    x_data = h5file['x_data_sel'][:]

    grid_type = h5file['grid_type'][()]  # dataset.value has been deprecated
    h5file.close()

    n_days = x_data.shape[0]
    day_vec = np.arange(0, n_days)

    # Initialization
    y_train = np.zeros_like(x_data[day_vec, :, :])

    seedsx1 = np.zeros((len(day_vec, )), dtype=int)
    seedsy1 = np.zeros((len(day_vec, )), dtype=int)

    seedsx2 = np.zeros((len(day_vec, )), dtype=int)
    seedsy2 = np.zeros((len(day_vec, )), dtype=int)

    figd = {}
    fign = {}
    figd2 = {}
    fign2 = {}

    # Data for plots
    lon, lat = np.meshgrid(nc_lon_data, nc_lat_data)

    gridx, gridy = lon, lat

    for idx, day in enumerate(day_vec):  # number of months to process

        if idx >= (day_init - 1):
            h5name = 'expertdatassw' + str(day + 1) + '.h5'

            f = h5py.File(pathexpert + expert_period + '/' + expert_model + '/' \
                          + h5name, 'w')  # 'w' truncates any existing file
            f.create_group(expert_model)
            f.close()

            print('Day', day + 1)

            img = np.negative(x_data[day_vec[idx], :, :])
            img1 = img.copy()
            img2 = img.copy()

            idx1 = (nc_lon_data > 180) + (nc_lon_data <= 10)
            img1[:, idx1] = -1E6
            seed_x1, seed_y1 = np.where(img1 == np.max(img1))  # Seed for region growing algorithm
            seedsx1[idx] = int(seed_x1[0].item())
            seedsy1[idx] = int(seed_y1[0].item())

            idx2 = (nc_lon_data <= 180) + (nc_lon_data > 350)
            img2[:, idx2] = -1E6
            seed_x2, seed_y2 = np.where(img2 == np.max(img2))  # Seed for region growing algorithm
            seedsx2[idx] = int(seed_x2[0].item())
            seedsy2[idx] = int(seed_y2[0].item())

            labeled_img = np.zeros_like(img)
            y_train[idx, :, :] = labeled_img

            # Plots region growing dictionaries
            fig = plt.figure(day)
            figd[str(idx)] = fig  # figure dictionary for figure handle
            fign[str(idx)] = pathexpert + expert_period + '/' + expert_model + '/' \
                             + 'zg_SSW_month' + str(day_vec[idx] + 1) + '_rg'  # figure dictionary for figure name

            '''
            Plot data
            '''
            print('Press ENTER to accept selected points. \
                 \n Press "d" to delete the selected points \
                 \n Close for NEXT image')

            plot_data(fig, gridx, gridy, x_data[day_vec[idx], :, :],
                      y_train[idx], nc_lon_data, nc_lat_data, idx,
                      seed_x1, seed_y1, day_vec, fign[str(idx)], 'region growing',
                      seed_x2, seed_y2)

            '''
            Get masked data selected by expert
            '''
            # Lasso selector for expert
            ax = plt.gca()
            pts = ax.scatter(gridx.flatten(), gridy.flatten(), s=0.01)  # transparent

            labeled_img3 = labeled_img.copy()

            idx_msk, idx_mskd = select_regions(ax, pts, fig)
            labeled_img3 = np.reshape(labeled_img3, (len(nc_lat_data) * len(nc_lon_data),))
            labeled_img3[idx_msk] = 1.
            labeled_img3[idx_mskd] = 0.
            labeled_img3 = np.reshape(labeled_img3, (len(nc_lat_data), len(nc_lon_data)))

            # Update y_train with new data
            y_train[idx, :, :] = labeled_img3

            f = h5py.File(pathexpert + expert_period + '/' + expert_model + '/' \
                          + h5name, 'a')  # 'w' truncates any existing file
            grp0 = f[expert_model]
            grp0.create_dataset(name='day' + str(day + 1), data=y_train[idx, :, :])

            f.close()

            plt.close()

    for idx, day in enumerate(day_vec):
        if idx >= (day_init - 1):
            # Plots expert dictionaries
            fig = plt.figure(day + 200)
            figd2[str(idx)] = fig
            fign2[str(idx)] = pathexpert + expert_period + '/' + expert_model + '/' \
                              + 'zg_SSW_month' + str(day_vec[idx] + 1) + '_expert'

            '''Save updated figure'''
            plot_data(fig, gridx, gridy, x_data[day_vec[idx], :, :],
                      y_train[idx], nc_lon_data, nc_lat_data, idx,
                      seedsx1[idx], seedsy1[idx],
                      day_vec, fign2[str(idx)], 'expert selection',
                      seedsx2[idx], seedsy2[idx])

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
              nc_lat_data, idx, seed_y1, seed_x1, day_vec, figname, llabel,
              seed_y2, seed_x2):
    # Plot the data

    plt.contourf(gridx, gridy, origdata,
                 levels=np.linspace(28800, 31900, 11),
                 vmin=28800, vmax=31900,
                 extend='min')
    cb = plt.colorbar()
    cb.set_label('zg [m]')

    if llabel == 'region growing':
        plt.plot(nc_lon_data[seed_x1],
                 nc_lat_data[seed_y1], 'k*', label=llabel)
        plt.plot(nc_lon_data[seed_x2],
                 nc_lat_data[seed_y2], 'k*')
    else:
        plt.plot(nc_lon_data[seed_x1],
                 nc_lat_data[seed_y1], 'k', label=llabel)
        plt.plot(nc_lon_data[seed_x2],
                 nc_lat_data[seed_y2], 'k', label=llabel)
    plt.contour(gridx, gridy, data4train, [0], colors='k')

    plt.ylabel('$\phi$ [' + u'\N{DEGREE SIGN}' + ']')
    plt.xlabel('$\lambda$ [' + u'\N{DEGREE SIGN}' + ']')
    plt.grid()
    plt.legend(loc='lower left')

    plt.title('Day ' + str(day_vec[idx] + 1) + ' ' + dates[idx])

    size = fig.get_size_inches()
    fig.set_size_inches(size[0], size[1])

    plt.savefig(figname + '.png', format='png', bbox_inches='tight')
