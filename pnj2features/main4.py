# General information about result h5 files

import h5py

f = h5py.File('TBD', 'r')
print(f['jra']['Month_type1event'][:])
print(f['jra']['Month_normalPNJ'][:])
print(f['jra']['Month_PNJ'][:])

f.close()
