# MODIS to Sentinel-2/Landsat Albedo Conversion

This repository contains Python code to generate high-resolution (10m) albedo maps from Sentinel-2 or Landsat data using machine learning, based on the methodology described in:

Liu, X., Ren, H., Li, X., & Zhang, X. (2023). Performance assessment of four data-driven machine learning models: a case to generate Sentinel-2 albedo at 10 meters. Remote Sensing, 15(10), 2684.

## Features

- BRDF correction using Ross-Li kernel-driven models
- Spectral indices calculation for enhanced predictions
- Four machine learning models:
  - Random Forest (RF)
  - XGBoost (XGB)
  - Support Vector Regression (SVR)
  - Neural Network (NN)
- Hyperparameter optimization via grid search
- Validation tools and visualizations

## Requirements

```
numpy
pandas
rasterio
scikit-learn
xgboost
matplotlib
```

Install requirements using:

```bash
pip install -r requirements.txt
```

## Usage

### Basic usage

```bash
python modis_sentinel_albedo.py /path/to/modis.tif /path/to/sentinel.tif /path/to/metadata.xml /path/to/output.tif
```

### With validation plot and grid search

```bash
python modis_sentinel_albedo.py /path/to/modis.tif /path/to/sentinel.tif /path/to/metadata.xml /path/to/output.tif --plot validation_plot.png --grid-search
```

## Input Data

- **MODIS data**: MCD43A1 BRDF/Albedo product (500m resolution)
- **Sentinel-2 data**: L2A surface reflectance product (10m resolution)
- **Metadata**: XML file containing sun-sensor geometry information

## Algorithm Steps

1. Load MODIS and Sentinel-2 data
2. Apply BRDF correction using Ross-Li kernel models
3. Align datasets (reproject MODIS to Sentinel-2 grid)
4. Calculate spectral indices (NDVI, NDWI, EVI, SAVI, BSI)
5. Create training dataset
6. Train and evaluate multiple ML models
7. Generate high-resolution albedo map using best model
8. Save results and validation plots

## Citation

If you use this code, please cite:

```
Liu, X., Ren, H., Li, X., & Zhang, X. (2023). Performance assessment of four data-driven machine learning models: a case to generate Sentinel-2 albedo at 10 meters. Remote Sensing, 15(10), 2684.
```