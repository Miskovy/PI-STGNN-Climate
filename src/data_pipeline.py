import os
import time
import requests
import zipfile
import io
import glob
import gc
import cdsapi
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import torch
from scipy.interpolate import griddata

# --- Configuration & Paths ---
START_YEAR = 1981
END_YEAR = 2020
MIDWEST_BBOX = [49.0, -104.0, 36.0, -80.0] # North, West, South, East

DIR_ERA5 = "../data/era5_historical/"
DIR_PRISM = "../data/prism_historical/"
DIR_TENSORS = "../data/processed_tensors/"

os.makedirs(DIR_ERA5, exist_ok=True)
os.makedirs(DIR_PRISM, exist_ok=True)
os.makedirs(DIR_TENSORS, exist_ok=True)

# --- 1. Batch Downloaders ---
def download_era5():
    print(f"Queueing ERA5 downloads from {START_YEAR} to {END_YEAR}...")
    # cdsapi automatically reads credentials from ~/.cdsapirc
    c = cdsapi.Client() 
    
    for year in range(START_YEAR, END_YEAR + 1):
        file_name = f"{DIR_ERA5}era5_midwest_{year}.nc"
        if os.path.exists(file_name):
            print(f"ERA5 Year {year} already exists. Skipping...")
            continue
            
        print(f"Requesting ERA5 data for {year}...")
        try:
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': ['geopotential', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind'],
                    'pressure_level': ['500', '850'], # 500 for Z500, 850 for moisture/wind
                    'year': str(year),
                    'month': [str(i).zfill(2) for i in range(1, 13)],
                    'day': [str(i).zfill(2) for i in range(1, 32)],
                    'time': '12:00',
                    'area': MIDWEST_BBOX,
                    'format': 'netcdf',
                },
                file_name)
        except Exception as e:
            print(f"Failed to download ERA5 for {year}: {e}")

def download_prism():
    print(f"Starting PRISM daily downloads...")
    date_range = pd.date_range(start=f'{START_YEAR}-01-01', end=f'{END_YEAR}-12-31')
    headers = {'User-Agent': 'Mozilla/5.0'}

    for single_date in date_range:
        date_str = single_date.strftime("%Y%m%d")
        year = single_date.strftime("%Y")
        year_dir = os.path.join(DIR_PRISM, year)
        os.makedirs(year_dir, exist_ok=True)
        
        expected_file = os.path.join(year_dir, f"prism_ppt_us_4km_{date_str}.tif")
        if os.path.exists(expected_file):
            continue

        url = f"https://services.nacse.org/prism/data/get/us/4km/ppt/{date_str}"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200 and response.content.startswith(b'PK'):
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall(year_dir)
            time.sleep(0.1) # Be polite to the PRISM servers
        except Exception as e:
            print(f"Network error on PRISM {date_str}: {e}")

# --- 2. Graph Mapping Utilities ---
def create_fibonacci_sphere(num_nodes=1000):
    indices = np.arange(0, num_nodes, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_nodes)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return torch.tensor(np.vstack((x, y, z)).T, dtype=torch.float32)

def map_to_graph(node_coords, grid_coords, grid_features, lon_min, lon_max, lat_min, lat_max):
    x, y, z = node_coords[:, 0], node_coords[:, 1], node_coords[:, 2]
    node_lat = np.degrees(np.arcsin(z))
    node_lon = np.degrees(np.arctan2(y, x))
    
    valid_nodes = (node_lon >= lon_min) & (node_lon <= lon_max) & \
                  (node_lat >= lat_min) & (node_lat <= lat_max)
    node_lon_lat = np.column_stack((node_lon[valid_nodes], node_lat[valid_nodes]))
    
    mapped_features = griddata(grid_coords, grid_features, node_lon_lat, method='nearest')
    return torch.tensor(mapped_features, dtype=torch.float32), valid_nodes

# --- 3. Processing and Garbage Collection Pipeline ---
def process_and_cleanup():
    print("\nStarting Data Fusion and Garbage Collection Pipeline...")
    lon_min, lon_max = MIDWEST_BBOX[1], MIDWEST_BBOX[3]
    lat_min, lat_max = MIDWEST_BBOX[2], MIDWEST_BBOX[0]
    node_coords = create_fibonacci_sphere(num_nodes=2000)

    for year in range(START_YEAR, END_YEAR + 1):
        era5_path = f'{DIR_ERA5}era5_midwest_{year}.nc'
        if not os.path.exists(era5_path):
            print(f"Missing ERA5 data for {year}, skipping processing.")
            continue
            
        print(f"Processing Year {year}...")
        ds_era5 = xr.open_dataset(era5_path)
        X_year_list, Y_year_list = [], []
        
        date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31')
        
        for single_date in date_range:
            date_str_nc = single_date.strftime("%Y-%m-%d")
            date_str_tif = single_date.strftime("%Y%m%d")
            
            # 1. Extract ERA5
            try:
                ds_day = ds_era5.sel(valid_time=date_str_nc, longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))
                lon_era5, lat_era5 = np.meshgrid(ds_day.longitude.values, ds_day.latitude.values)
                
                z500 = ds_day['z'].sel(pressure_level=500).values
                humidity = ds_day['q'].sel(pressure_level=850).values
                u_wind = ds_day['u'].sel(pressure_level=850).values
                v_wind = ds_day['v'].sel(pressure_level=850).values
                
                era5_coords = np.column_stack((lon_era5.ravel(), lat_era5.ravel()))
                era5_features = np.column_stack((z500.ravel(), humidity.ravel(), u_wind.ravel(), v_wind.ravel()))
            except KeyError:
                continue 
                
            # 2. Extract PRISM
            prism_files = glob.glob(f'{DIR_PRISM}{year}/*{date_str_tif}*.tif')
            if not prism_files: continue
                
            prism_file = prism_files[0]
            da_prism = rioxarray.open_rasterio(prism_file, masked=True).squeeze()
            da_prism = da_prism.rio.clip_box(minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max)
            
            regional_precip = np.nanmean(da_prism.values)
            # Temporary label; will be overwritten by 95th percentile logic in train.py
            is_extreme = 1.0 if regional_precip > 15.0 else 0.0 
            
            # 3. Map to spherical graph
            node_features, _ = map_to_graph(node_coords.numpy(), era5_coords, era5_features, lon_min, lon_max, lat_min, lat_max)
            X_year_list.append(node_features)
            Y_year_list.append(torch.tensor([is_extreme], dtype=torch.float32))
            
            # 4. Clean up daily PRISM files
            da_prism.close()
            os.remove(prism_file)
            for extra_file in glob.glob(f'{DIR_PRISM}{year}/*{date_str_tif}*'):
                os.remove(extra_file)

        # 5. Save yearly tensors
        if X_year_list:
            X_tensor = torch.stack(X_year_list)
            Y_tensor = torch.stack(Y_year_list)
            torch.save(X_tensor, f'{DIR_TENSORS}X_{year}.pt')
            torch.save(Y_tensor, f'{DIR_TENSORS}Y_{year}.pt')
            print(f"Saved tensors for {year}. Shape: {X_tensor.shape}")
            
        # 6. Clean up yearly ERA5 file
        ds_era5.close()
        os.remove(era5_path)
        gc.collect()

if __name__ == "__main__":
    download_era5()
    download_prism()
    process_and_cleanup()
    print("Data pipeline executed successfully!")
