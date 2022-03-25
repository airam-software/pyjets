# Process netcdf data

import datetime as dt
import fnmatch
import os

import h5py
import numpy as np
import pandas as pd

try:
    from netCDF4 import Dataset, num2date

    print("module 'netCDF4' is installed")
except ModuleNotFoundError:
    print("module 'netCDF4' is not installed")


def convertnc2h5(study_variable, model_name, period, level, path):
    if period == 'piControl' or period == 'abrupt-4xCO2':

        file_name = find_file(study_variable, path, level, model_name, period)
        file_name = file_name[:-3]

        nc = Dataset(path + '/' + file_name + '.nc')  # Read nc file
        nc_time = nc.variables['time']

        dates = num2date(nc_time[:], units=nc_time.units,
                         calendar=nc_time.calendar)  # Select dates for three months separately
        print('-----')
        print(study_variable, model_name, period)
        print(dates[0], dates[-1], dates.shape[0] / 12)

        lat_names = ['lat', 'latitude', 'nav_lat']
        lon_names = ['lon', 'longitude', 'nav_lon']

        lat_name = [i for i in nc.variables.keys() if i in lat_names]  # Check if any item in one list is in another
        lat_name = lat_name[0]
        if not lat_name:
            print('Error')
            exit()

        lon_name = [i for i in nc.variables.keys() if i in lon_names]
        lon_name = lon_name[0]
        if not lon_name:
            print('Error')
            exit()

        nc_lat = nc.variables[lat_name]
        nc_lon = nc.variables[lon_name]
        nc_lat_data = nc_lat[:].data
        nc_lon_data = nc_lon[:].data

        x = nc.variables[study_variable]
        x_data = np.squeeze(x[:].data)

        if len(nc_lat_data.shape) == 1:
            grid_type = 'sq'
            if nc_lat_data.shape[0] != x_data.shape[1] or \
                    nc_lon_data.shape[0] != x_data.shape[2]:
                print('Inconsistency latitude or longitude')
        if len(nc_lat_data.shape) == 2:
            grid_type = 'nonsq'
            if nc_lat_data.shape[0] != x_data.shape[1]:
                print('Inconsistency latitude')
            if nc_lon_data.shape[1] != x_data.shape[2]:
                print('Inconsistency longitude')

        x_data_sel, dates_selection = select_season(x_data, dates, 'winter')

        nc.close()

    elif period == 'jra':

        file_name = find_file(study_variable, path, level, model_name, period)
        file_name = file_name[:-3]

        if study_variable == 'ua':
            variable = 'UGRD_GDS0_ISBL'
        elif study_variable == 'zg':
            variable = 'HGT_GDS0_ISBL'

        nc = Dataset(path + '/' + file_name + '.nc')  # Read nc file

        nc_time = nc.variables['initial_time0_hours']

        # '''Select only some years'''
        # year_start = 1958
        # year_end = 2010
        # time_start = date2index(dt.datetime(year_start, 1, 1), nc_time, select='nearest')
        # time_end = date2index(dt.datetime(year_end, 12, 16), nc_time, select='nearest')
        # dates = num2date(nc_time[time_start:time_end + 1], units=nc_time.units, calendar=nc_time.calendar)
        # print(dates[0], dates[-1], dates.shape[0] / 12)

        '''All dates'''
        dates = num2date(nc_time[:], units=nc_time.units, calendar=nc_time.calendar)
        print('-----')
        print(study_variable, model_name, period)
        print(dates[0], dates[-1], dates.shape[0] / 12)

        nc_lat = nc.variables['lat']  # Other option nc_lat = nc.variables['g0_lat_2']
        nc_lat_data = nc_lat[:].data

        nc_lon = nc.variables['lon']  # Other option: nc_lon = nc.variables['g0_lon_3']
        nc_lon_data = nc_lon[:].data

        x = nc.variables[variable]
        x_data = np.squeeze(x[:].data)

        if len(nc_lat_data.shape) == 1:
            grid_type = 'sq'
            if nc_lat_data.shape[0] != x_data.shape[1] or \
                    nc_lon_data.shape[0] != x_data.shape[2]:
                print('Inconsistency latitude or longitude')
        if len(nc_lat_data.shape) == 2:
            grid_type = 'nonsq'
            if nc_lat_data.shape[0] != x_data.shape[1]:
                print('Inconsistency latitude')
            if nc_lon_data.shape[1] != x_data.shape[2]:
                print('Inconsistency longitude')

        nc.close()

        x_data_sel, dates_selection = select_season(x_data, dates, 'winter')

    '''Save data as hdf5'''
    hf = h5py.File(path + '/h5files/' + file_name + '.h5', 'w')
    hf.create_dataset('nc_lat_data', data=nc_lat_data)
    hf.create_dataset('nc_lon_data', data=nc_lon_data)
    hf.create_dataset('grid_type', data=grid_type)
    hf.create_dataset('x_data_sel', data=x_data_sel)
    hf.close()

    return nc_lat_data, nc_lon_data, grid_type, x_data_sel


def h5fileinfo(study_variable, model_name, period, level, path):
    if period == 'piControl' or period == 'abrupt-4xCO2':

        if study_variable == 'tos':
            str_aux = '_Omon_'
        else:
            str_aux = '_Amon_'

        for file in os.listdir(path):
            if fnmatch.fnmatch(file, study_variable + level +
                                     str_aux + model_name + '_' + period + '_*'):
                file_name = file

        h5name = file_name

    elif period == 'jra' and study_variable == 'ua':
        h5name = 'ua010_jra-55_195801-201512_remapped.h5'
    elif period == 'jra' and study_variable == 'zg':
        h5name = 'zg010_jra-55_195801-201606_mm_remapped.h5'

    h5file = h5py.File(path + h5name, 'r')

    nc_lat_data = h5file['nc_lat_data'][:]
    nc_lon_data = h5file['nc_lon_data'][:]
    grid_type = h5file['grid_type'].value
    x_data_sel = h5file['x_data_sel'][:]

    h5file.close()

    return nc_lat_data.shape, nc_lon_data.shape, grid_type, x_data_sel.shape


def select_season(x_data, dates, season):
    df = pd.DataFrame({'Date': dates})

    df['Date'] = [dt.date(i.year, i.month, i.day) for i in df.Date]  # Convert DatetimeNoLeap to date object

    df['Month'] = [i.month for i in df.Date]  # Add new columns to dataframe

    if season == 'winter':
        df1 = df.loc[df['Month'].isin([1, 2, 12])]

    dates_selection = df1.index.values

    x_data = x_data[dates_selection, :, :]

    df1.to_excel('./results/dates.xlsx')

    return x_data, df1['Date']


def find_file(study_variable, path, level, model_name, period):
    if study_variable == 'tos':
        str_aux = '_Omon_'
    else:
        str_aux = '_Amon_'

    if period != 'jra':
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, study_variable + level +
                                     str_aux + model_name + '_' + period + '_*'):
                file_name = file
    else:
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, study_variable + level + '_' +
                                     model_name + '-55_*'):
                file_name = file

    return file_name


def ncfileinfo(study_variable, model_name, period, level, path):
    if period == 'piControl' or period == 'abrupt-4xCO2':

        file_name = find_file(study_variable, path, level, model_name, period)
        file_name = file_name[:-3]

        nc = Dataset(path + '/' + file_name + '.nc')  # Read nc file
        nc_time = nc.variables['time']

        dates = num2date(nc_time[:], units=nc_time.units,
                         calendar=nc_time.calendar)  # Select dates for three months separately
        print('-----')
        print(study_variable, model_name, period)
        print(dates[0], dates[-1], dates.shape[0] / 12)

        lat_names = ['lat', 'latitude', 'nav_lat']
        lon_names = ['lon', 'longitude', 'nav_lon']

        lat_name = [i for i in nc.variables.keys() if i in lat_names]  # Check if any item in one list is in another
        lat_name = lat_name[0]
        if not lat_name:
            print('Error')
            exit()

        lon_name = [i for i in nc.variables.keys() if i in lon_names]
        lon_name = lon_name[0]
        if not lon_name:
            print('Error')
            exit()

        nc_lat = nc.variables[lat_name]
        nc_lon = nc.variables[lon_name]
        nc_lat_data = nc_lat[:].data
        nc_lon_data = nc_lon[:].data

        x = nc.variables[study_variable]
        x_data = np.squeeze(x[:].data)

        if len(nc_lat_data.shape) == 1:
            grid_type = 'sq'
            if nc_lat_data.shape[0] != x_data.shape[1] or \
                    nc_lon_data.shape[0] != x_data.shape[2]:
                print('Inconsistency latitude or longitude')
        if len(nc_lat_data.shape) == 2:
            grid_type = 'nonsq'
            if nc_lat_data.shape[0] != x_data.shape[1]:
                print('Inconsistency latitude')
            if nc_lon_data.shape[1] != x_data.shape[2]:
                print('Inconsistency longitude')

        x_data_sel, dates_selection = select_season(x_data, dates, 'winter')

        nc.close()

    elif period == 'jra':

        file_name = find_file(study_variable, path, level, model_name, period)
        file_name = file_name[:-3]

        if study_variable == 'ua':
            variable = 'UGRD_GDS0_ISBL'
        elif study_variable == 'zg':
            variable = 'HGT_GDS0_ISBL'

        nc = Dataset(path + '/' + file_name + '.nc')  # Read nc file

        nc_time = nc.variables['initial_time0_hours']

        '''All dates'''
        dates = num2date(nc_time[:], units=nc_time.units, calendar=nc_time.calendar)
        print('-----')
        print(study_variable, model_name, period)
        print(dates[0], dates[-1], dates.shape[0] / 12)

        nc_lat = nc.variables['lat']  # Other option nc_lat = nc.variables['g0_lat_2']
        nc_lat_data = nc_lat[:].data

        nc_lon = nc.variables['lon']  # Other option: nc_lon = nc.variables['g0_lon_3']
        nc_lon_data = nc_lon[:].data

        x = nc.variables[variable]
        x_data = np.squeeze(x[:].data)

        if len(nc_lat_data.shape) == 1:
            grid_type = 'sq'
            if nc_lat_data.shape[0] != x_data.shape[1] or \
                    nc_lon_data.shape[0] != x_data.shape[2]:
                print('Inconsistency latitude or longitude')
        if len(nc_lat_data.shape) == 2:
            grid_type = 'nonsq'
            if nc_lat_data.shape[0] != x_data.shape[1]:
                print('Inconsistency latitude')
            if nc_lon_data.shape[1] != x_data.shape[2]:
                print('Inconsistency longitude')

        nc.close()

        x_data_sel, dates_selection = select_season(x_data, dates, 'winter')

    return dates_selection
