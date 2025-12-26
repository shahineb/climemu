# %%
import cftime
import numpy as np
import xarray as xr
from tqdm import tqdm
from getdata2_eof import getdata


# %%
def get_latlon():
    args = {"field": "T",
            "type": "del",
            "delTbar": "0",
            "month": "0",
            "em": "eof"}
    output = getdata(args)
    lat = output["lat"]
    lon = output["lon"]
    return lat, lon

def generate_eof_data(gmst, month):
    data = {}
    for field in ["T", "precip"]:
        data[field] = {}
        for typ in ["del", "std_dev"]:
            args = {"field": field,
                    "type": typ,
                    "delTbar": str(gmst),
                    "month": str(month),
                    "em": "eof"}
            output = getdata(args)["v"]
            data[field][typ] = output
    return data

def stack_dicts(dicts, time, lat, lon, axis=0):
    out = {}
    for key in dicts[0]:
        values = [d[key] for d in dicts]
        if isinstance(values[0], dict):
            out[key] = stack_dicts(values, time, lat, lon, axis=axis)
        else:
            out[key] = xr.DataArray(
                np.stack(values, axis=axis),
                dims=("time", "lat", "lon"),
                coords={"time": time, "lat": lat, "lon": lon},
                name=key,
            )
    return out


# %%
ssp245 = xr.open_dataset("mpi_ssp245_gmst.nc").tas.values

# %%
# n_total = 12 * len(ssp245)
# with tqdm(total=n_total) as pbar:
#     all_data = []
#     for year_idx in range(len(ssp245)):
#         year_data = []
#         for month in range(1, 13):
#             gmst = ssp245[year_idx]
#             data = generate_eof_data(gmst=gmst, month=month)
#             all_data.append(data)
#             _ = pbar.update(1)

# %%
time = np.array(
    [cftime.DatetimeNoLeap(y, m, 1, has_year_zero=True)
     for y in range(2015, 2101)
     for m in range(1, 13)],
    dtype=object
)
lat, lon = get_latlon()
stacked = stack_dicts(all_data, time, lat, lon)

# %%
stacked['T']['del'].to_netcdf("eof_emulator_T_del.nc")
stacked['T']['std_dev'].to_netcdf("eof_emulator_T_std_dev.nc")
stacked['precip']['del'].to_netcdf("eof_emulator_precip_del.nc")
stacked['precip']['std_dev'].to_netcdf("eof_emulator_precip_std_dev.nc")

