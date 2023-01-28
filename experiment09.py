# %%

# Data downloaded from this page:
# https://data.giss.nasa.gov/gistemp/

# Direct link to download the Zarr file:
# https://data.giss.nasa.gov/pub/gistemp/gistemp1200_GHCNv4_ERSSTv5.zarr.tar.gz

# %%

import zarr

# %%

dataset = zarr.load('gistemp1200_GHCNv4_ERSSTv5.zarr')

# %%

dataset['lat']

dataset['lon']

dataset['tempanomaly']

dataset['time']
