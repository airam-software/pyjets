# Average magnitudes over a region

import numpy as np


def get_avg_mag(x_climatology, labeled_img, weight_factor, grid_type):
    if grid_type == 'sq':
        avgmag = np.average(x_climatology[labeled_img >= 1],
                            weights=weight_factor[
                                labeled_img >= 1])
    elif grid_type == 'nonsq':
        masked_data = np.ma.masked_array(x_climatology, np.isnan(x_climatology))
        avgmag = np.average(masked_data[labeled_img >= 1],
                            weights=weight_factor[
                                labeled_img >= 1])

    return avgmag


def get_avg_lat(labeled_img, nc_lat_data, weight_factor, grid_type):
    if grid_type == 'sq':
        lat_matrix = np.zeros_like(labeled_img)
        for i in np.arange(0, labeled_img.shape[0]):
            lat_matrix[i, :] = nc_lat_data[i]

    elif grid_type == 'nonsq':
        lat_matrix = nc_lat_data

    avglat = np.average(lat_matrix[labeled_img >= 1],
                        weights=weight_factor[
                            labeled_img >= 1])

    return avglat


def get_avg_lon(labeled_img, nc_lon_data, weight_factor, cross_opt, grid_type):
    lon_matrix = np.zeros_like(labeled_img)

    if grid_type == 'sq':
        if cross_opt == 'yes':  # Longitude crosses over 0 and 360 deg
            for j in np.arange(0, labeled_img.shape[1]):
                if 0 <= nc_lon_data[j] < 180:
                    lon_matrix[:, j] = nc_lon_data[j]
                else:
                    lon_matrix[:, j] = nc_lon_data[j] - 360
            avglon = (np.average(lon_matrix[labeled_img >= 1],
                                 weights=weight_factor[
                                     labeled_img >= 1]))

        elif cross_opt == 'no':
            for j in np.arange(0, labeled_img.shape[1]):
                lon_matrix[:, j] = nc_lon_data[j]

            avglon = np.average(lon_matrix[labeled_img >= 1],
                                weights=weight_factor[
                                    labeled_img >= 1])

    elif grid_type == 'nonsq':

        if cross_opt == 'yes':  # Longitude crosses over 0 and 360 deg
            for i in np.arange(0, labeled_img.shape[0]):
                for j in np.arange(0, labeled_img.shape[1]):
                    if 0 <= nc_lon_data[i, j] < 180:
                        lon_matrix[i, j] = nc_lon_data[i, j]
                    else:
                        lon_matrix[i, j] = nc_lon_data[i, j] - 360
            avglon = (np.average(lon_matrix[labeled_img >= 1],
                                 weights=weight_factor[
                                     labeled_img >= 1]))
        elif cross_opt == 'no':
            lon_matrix = nc_lon_data
            avglon = np.average(lon_matrix[labeled_img >= 1],
                                weights=weight_factor[
                                    labeled_img >= 1])

    return avglon
