import pandas as pd
import numpy as np
import time
import pickle
import netCDF4 as nc

num_rows = 1000000
num_columns = 10
data = np.random.rand(num_rows, num_columns)
df = pd.DataFrame(data, columns=[f'col{i}' for i in range(num_columns)])

def benchmark_format(format_name, save_fn, load_fn, file_name):
    start_time = time.time()
    save_fn(df, file_name)
    save_time = time.time() - start_time

    start_time = time.time()
    _ = load_fn(file_name)
    load_time = time.time() - start_time

    print(f"{format_name}:")
    print(f"  Save time: {save_time:.4f} seconds")
    print(f"  Load time: {load_time:.4f} seconds")
    print("-" * 40)

benchmark_format(
    "CSV",
    lambda df, f: df.to_csv(f, index=False),
    lambda f: pd.read_csv(f),
    "data.csv"
)

benchmark_format(
    "Parquet",
    lambda df, f: df.to_parquet(f),
    lambda f: pd.read_parquet(f),
    "data.parquet"
)

benchmark_format(
    "HDF5",
    lambda df, f: df.to_hdf(f, key="df", mode="w"),
    lambda f: pd.read_hdf(f),
    "data.h5"
)

benchmark_format(
    "Feather",
    lambda df, f: df.to_feather(f),
    lambda f: pd.read_feather(f),
    "data.feather"
)

benchmark_format(
    "JSON",
    lambda df, f: df.to_json(f, orient="records"),
    lambda f: pd.read_json(f, orient="records"),
    "data.json"
)

def save_netcdf(df, file_name):
    ds = nc.Dataset(file_name, "w", format="NETCDF4")
    ds.createDimension("row", df.shape[0])
    for col in df.columns:
        var = ds.createVariable(col, "f4", ("row",))
        var[:] = df[col].values
    ds.close()

def load_netcdf(file_name):
    ds = nc.Dataset(file_name, "r")
    data = {var: ds.variables[var][:] for var in ds.variables}
    df = pd.DataFrame(data)
    ds.close()
    return df

benchmark_format(
    "NetCDF4",
    save_netcdf,
    load_netcdf,
    "data.nc"
)

benchmark_format(
    "NumPy (NPY)",
    lambda df, f: np.save(f, df.values),
    lambda f: pd.DataFrame(np.load(f), columns=df.columns),
    "data.npy"
)

benchmark_format(
    "Pickle",
    lambda df, f: pickle.dump(df, open(f, "wb")),
    lambda f: pickle.load(open(f, "rb")),
    "data.pkl"
)

import os
for file in ["data.csv", "data.parquet", "data.h5", "data.feather", "data.json", "data.nc", "data.npy", "data.pkl"]:
    os.remove(file)
