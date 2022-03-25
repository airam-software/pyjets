# Region growing and image segmentation

import numpy as np
from scipy import ndimage


def intersect(*lists):
    return list(set.intersection(*map(set, lists)))  # intersect an arbitrary number of lists


def get8n(x, y, shape):
    neighboring_pixels = []
    x_coord_max = shape[0] - 1
    y_coord_max = shape[1] - 1

    # Point 1
    neighboring_pixels_x = min(max(x - 1, 0), x_coord_max)
    neighboring_pixels_y = min(max(y - 1, 0), y_coord_max)
    neighboring_pixels.append((neighboring_pixels_x, neighboring_pixels_y))

    # Point 2
    neighboring_pixels_x = x
    neighboring_pixels_y = min(max(y - 1, 0), y_coord_max)
    neighboring_pixels.append((neighboring_pixels_x, neighboring_pixels_y))

    # Point 3
    neighboring_pixels_x = min(max(x + 1, 0), x_coord_max)
    neighboring_pixels_y = min(max(y - 1, 0), y_coord_max)
    neighboring_pixels.append((neighboring_pixels_x, neighboring_pixels_y))

    # Point 4
    neighboring_pixels_x = min(max(x - 1, 0), x_coord_max)
    neighboring_pixels_y = y
    neighboring_pixels.append((neighboring_pixels_x, neighboring_pixels_y))

    # Point 5
    neighboring_pixels_x = min(max(x + 1, 0), x_coord_max)
    neighboring_pixels_y = y
    neighboring_pixels.append((neighboring_pixels_x, neighboring_pixels_y))

    # Point 6
    neighboring_pixels_x = min(max(x - 1, 0), x_coord_max)
    neighboring_pixels_y = min(max(y + 1, 0), y_coord_max)
    neighboring_pixels.append((neighboring_pixels_x, neighboring_pixels_y))

    # Point 7
    neighboring_pixels_x = x
    neighboring_pixels_y = min(max(y + 1, 0), y_coord_max)
    neighboring_pixels.append((neighboring_pixels_x, neighboring_pixels_y))

    # Point 8
    neighboring_pixels_x = min(max(x + 1, 0), x_coord_max)
    neighboring_pixels_y = min(max(y + 1, 0), y_coord_max)
    neighboring_pixels.append((neighboring_pixels_x, neighboring_pixels_y))

    return neighboring_pixels


def region_growing_f(img, threshold, seed_x, seed_y):
    wrklist = []
    wrklist.append((seed_x, seed_y))  # Append seed to working list
    labeled_img1 = np.zeros_like(img)
    processed = []
    while (len(wrklist) > 0):
        pixel = wrklist[0]
        labeled_img1[pixel[0], pixel[1]] = 1  # The first item of the list is part of the selected region
        for coord in get8n(pixel[0], pixel[1], img.shape):
            if img[coord[0], coord[1]] >= threshold:
                labeled_img1[coord[0], coord[1]] = 1
                if coord not in processed:
                    wrklist.append(coord)

                processed.append(coord)
        wrklist.pop(
            0)

    labeled_img = labeled_img1

    return labeled_img


def region_growing_2(img, threshold, seed_x, seed_y, nc_lat_data, nc_lon_data, grid_type):
    wrklist = []
    wrklist.append((seed_x, seed_y))  # Append seed to working list
    labeled_img1 = np.zeros_like(img)
    processed = []

    while (len(wrklist) > 0):
        pixel = wrklist[0]
        labeled_img1[pixel[0], pixel[1]] = 1  # The first item of the list is part of the selected region
        for coord in get8n(pixel[0], pixel[1], img.shape):
            if grid_type == 'nonsq':
                if img[coord[0], coord[1]] >= threshold and \
                        abs(nc_lat_data[coord[0], coord[1]]) < 5:
                    labeled_img1[coord[0], coord[1]] = 1
                    if coord not in processed:
                        wrklist.append(coord)
                    processed.append(coord)
            elif grid_type == 'sq':
                if img[coord[0], coord[1]] >= threshold and \
                        abs(nc_lat_data[coord[0]]) < 5:
                    labeled_img1[coord[0], coord[1]] = 1
                    if coord not in processed:
                        wrklist.append(coord)
                    processed.append(coord)
        wrklist.pop(
            0)

    # Wrapping from the longitude extremes
    if labeled_img1[:, 0].any():  # If any of the elements at 0 degrees longitude have been selected
        interest_idx = np.where(labeled_img1[:, 0] == 1)
        # Overwrite seeds
        seed_x = np.asarray(interest_idx)[0, 0]
        seed_y = img.shape[1] - 1
        wrklist = []
        wrklist.append((seed_x, seed_y))  # Append seed to working list
        labeled_img2 = np.zeros_like(img)
        processed = []
        while (len(wrklist) > 0):
            pixel = wrklist[0]
            labeled_img2[pixel[0], pixel[1]] = 1  # The first item of the list is part of the selected region
            for coord in get8n(pixel[0], pixel[1], img.shape):
                if grid_type == 'nonsq':
                    if img[coord[0], coord[1]] >= threshold and \
                            abs(nc_lat_data[coord[0], coord[1]]) < 5:
                        labeled_img2[coord[0], coord[1]] = 1
                        if not coord in processed:
                            wrklist.append(coord)
                        processed.append(coord)
                    elif grid_type == 'sq':
                        if img[coord[0], coord[1]] >= threshold and \
                                abs(nc_lat_data[coord[0]]) < 5:
                            labeled_img2[coord[0], coord[1]] = 1
                            if not coord in processed:
                                wrklist.append(coord)
                            processed.append(coord)
            wrklist.pop(
                0)
    elif labeled_img1[:, -1].any():  # If any of the elements at 360 degrees longitude have been selected
        interest_idx = np.where(labeled_img1[:, -1] == 1)
        # Overwrite seeds
        seed_x = np.asarray(interest_idx)[0, 0]
        seed_y = 0  # This changes with respect to the first if
        wrklist = []
        wrklist.append((seed_x, seed_y))  # Append seed to working list
        labeled_img2 = np.zeros_like(img)
        processed = []
        while len(wrklist) > 0:
            pixel = wrklist[0]
            labeled_img2[pixel[0], pixel[1]] = 1  # The first item of the list is part of the selected region
            for coord in get8n(pixel[0], pixel[1], img.shape):
                if grid_type == 'nonsq':
                    if img[coord[0], coord[1]] >= threshold and \
                            abs(nc_lat_data[coord[0], coord[1]]) < 5:
                        labeled_img2[coord[0], coord[1]] = 1
                        if not coord in processed:
                            wrklist.append(coord)
                        processed.append(coord)
                    elif grid_type == 'sq':
                        if img[coord[0], coord[1]] >= threshold and \
                                abs(nc_lat_data[coord[0]]) < 5:
                            labeled_img2[coord[0], coord[1]] = 1
                            if not coord in processed:
                                wrklist.append(coord)
                            processed.append(coord)
            wrklist.pop(
                0)

    labeled_img = np.zeros_like(img)

    for i in np.arange(0, img.shape[0]):
        for j in np.arange(0, img.shape[1]):
            if labeled_img1[:, 0].any() or labeled_img1[:, -1].any():
                labeled_img[i, j] = max(labeled_img1[i, j], labeled_img2[i, j])
            else:
                labeled_img[i, j] = labeled_img1[i, j]

    return labeled_img


def segmentimage0(img, threshold):
    blobs = img >= threshold
    labeled_img, nlabeled_img = ndimage.label(blobs)
    labeled_img = labeled_img.astype(float)

    labeled_img[labeled_img > 0] = 1.0

    return labeled_img


def segmentimage1(img, threshold, seed_x, seed_y):
    blobs = img >= threshold
    labeled_img, nlabels = ndimage.label(blobs)
    labeled_img = labeled_img.astype(float)

    regions = {}
    for i in np.arange(0, nlabels):
        labels_aux = np.copy(labeled_img)
        labels_aux[labeled_img != i + 1] = 0
        name = str(i + 1)
        regions[name] = labels_aux
        if regions[name][seed_x, seed_y] != 0:
            labeled_img = regions[name]
            break

    labeled_img[labeled_img > 0] = 1.0

    return labeled_img


def segmentimage2(img, threshold, seed_x, seed_y):
    blobs = img >= threshold
    labeled_img, nlabels = ndimage.label(blobs)
    labeled_img = labeled_img.astype(float)

    regions = {}
    borders = {}

    flag_opt = 0

    for i in np.arange(0, nlabels):
        labels_aux = np.copy(labeled_img)
        labels_aux[labeled_img != i + 1] = 0
        name = str(i + 1)
        regions[name] = labels_aux
        if regions[name][seed_x, seed_y] != 0:
            name1 = name
            seed_region1 = regions[name1]
            if np.count_nonzero(regions[name1][:, 0]) > 0:
                flag_opt = 1
                borders[name1] = np.where(regions[name1][:, 0] != 0)
            elif np.count_nonzero(regions[name1][:, -1]) > 0:
                flag_opt = 2
                borders[name1] = np.where(regions[name1][:, -1] != 0)

    seed_region2 = np.zeros_like(seed_region1)
    for i in np.arange(0, nlabels):
        name = str(i + 1)
        if name != name1 and flag_opt != 0:
            if flag_opt == 1:
                borders[name] = np.where(regions[name][:, -1] != 0)
            elif flag_opt == 2:
                borders[name] = np.where(regions[name][:, 0] != 0)

            if np.intersect1d(borders[name1], borders[name]).size != 0:
                seed_region2 += regions[name]

    seed_region1[seed_region1 > 0] = 1.0
    seed_region2[seed_region2 > 0] = 1.0

    labeled_img = np.maximum(seed_region1, seed_region2)

    return labeled_img


def segmentimage_ssw(img, threshold, seed_x, seed_y):
    blobs = img >= threshold
    labeled_img, nlabels = ndimage.label(blobs)
    labeled_img = labeled_img.astype(float)

    regions = {}
    for i in np.arange(0, nlabels):
        labels_aux = np.copy(labeled_img)
        labels_aux[labeled_img != i + 1] = 0
        name = str(i + 1)
        regions[name] = labels_aux
        if regions[name][seed_x, seed_y].any() != 0:
            labeled_img = regions[name]
            break

    labeled_img[labeled_img > 0] = 1.0

    return labeled_img
