# Brief script for information about resulting polar night jet h5 files

import h5py

f = h5py.File('TBD', 'r')
print(f['jra']['Month_type1event'][:])
print(f['jra']['Month_normalPNJ'][:])

f.close()
