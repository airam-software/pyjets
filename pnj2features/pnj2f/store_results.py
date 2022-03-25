# Store results in hdf5 files

import h5py
import numpy as np


def store_pnj_vars_2h5(h5file, model_name, Month, ua_PNJ, ua_PNJ_std, PNJ_avglat, PNJ_avglon):
    f = h5py.File(h5file, 'w')  # 'w' truncates any existing file

    grp0 = f.create_group(model_name)

    '''PNJ'''
    region = 'Polar night jet'

    dtype = np.dtype([
        ('Month', np.float),
        ('ua_PNJ', np.float),
        ('PNJ_avglat', np.float),
        ('ua_PNJ_std', np.float),
        ('PNJ_avglon', np.float)
    ])
    wdata = np.zeros((len(Month),), dtype=dtype)
    wdata['Month'] = Month
    wdata['ua_PNJ'] = ua_PNJ
    wdata['PNJ_avglat'] = PNJ_avglat
    wdata['ua_PNJ_std'] = ua_PNJ_std
    wdata['PNJ_avglon'] = PNJ_avglon

    grp0.create_dataset(region, data=wdata)
