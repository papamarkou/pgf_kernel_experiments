# %%

import zarr

# %%

dataset = zarr.load('gistemp1200_GHCNv4_ERSSTv5.zarr')

# %%

dataset['lat']

dataset['lon']

dataset['tempanomaly']

dataset['time']
