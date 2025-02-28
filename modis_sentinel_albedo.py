"""
MODIS to Sentinel-2 Albedo Conversion using Machine Learning
Based on: Liu et al. (2023) - Performance assessment of four data-driven machine learning models:
a case to generate Sentinel-2 albedo at 10 meters
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


def calculate_ross_kernel(solar_zenith, view_zenith, relative_azimuth):
    """
    Calculate the Ross-Thick volumetric kernel.
    All angles in radians.
    """
    solar_zenith = np.radians(solar_zenith)
    view_zenith = np.radians(view_zenith)
    relative_azimuth = np.radians(relative_azimuth)
    
    cos_phase = np.cos(solar_zenith) * np.cos(view_zenith) + \
                np.sin(solar_zenith) * np.sin(view_zenith) * np.cos(relative_azimuth)
    
    phase = np.arccos(np.clip(cos_phase, -1, 1))
    
    # Ross-Thick kernel
    k_vol = ((np.pi/2 - phase) * np.cos(phase) + np.sin(phase)) / \
            (np.cos(solar_zenith) + np.cos(view_zenith)) - np.pi/4
    
    return k_vol


def calculate_li_kernel(solar_zenith, view_zenith, relative_azimuth):
    """
    Calculate the Li-Sparse geometric kernel.
    All angles in radians.
    """
    solar_zenith = np.radians(solar_zenith)
    view_zenith = np.radians(view_zenith)
    relative_azimuth = np.radians(relative_azimuth)
    
    # Fixed parameters
    h_b_r = 2  # height-to-crown center/crown radius ratio
    b_r = 1    # vertical-to-horizontal crown radius ratio
    
    # Li-Sparse geometric kernel calculation
    D = np.sqrt((np.tan(solar_zenith)**2 + np.tan(view_zenith)**2 - 
                 2*np.tan(solar_zenith)*np.tan(view_zenith)*np.cos(relative_azimuth)))
    
    cost = h_b_r * np.sqrt(D**2 + (np.tan(solar_zenith)*np.tan(view_zenith)*np.sin(relative_azimuth))**2)
    
    t = np.arccos(np.clip(cost, -1, 1))
    
    O = (1/np.pi) * (t - np.sin(t)*np.cos(t)) * (b_r / h_b_r)
    
    return O


def apply_brdf_correction(sentinel_bands, sun_zenith, view_zenith, relative_azimuth, brdf_parameters):
    """
    Apply BRDF correction to Sentinel-2 bands using MODIS BRDF parameters
    """
    corrected_bands = []
    for i, band in enumerate(sentinel_bands):
        k_iso = brdf_parameters[i, 0]  # isotropic kernel parameter
        k_vol = brdf_parameters[i, 1]  # volumetric kernel parameter 
        k_geo = brdf_parameters[i, 2]  # geometric kernel parameter
        
        # Calculate kernels based on illumination/viewing geometry
        f_iso = 1.0
        f_vol = calculate_ross_kernel(sun_zenith, view_zenith, relative_azimuth)
        f_geo = calculate_li_kernel(sun_zenith, view_zenith, relative_azimuth)
        
        # Apply correction
        corrected_band = band * (k_iso + k_vol*f_vol + k_geo*f_geo)
        corrected_bands.append(corrected_band)
    
    return np.array(corrected_bands)


def load_modis_data(modis_path):
    """
    Load MODIS MCD43A1 BRDF/Albedo product
    """
    with rasterio.open(modis_path) as src:
        modis_albedo = src.read(1)  # Black-sky albedo at local solar noon
        modis_transform = src.transform
        modis_crs = src.crs
        modis_meta = src.meta
    
    return modis_albedo, modis_transform, modis_crs, modis_meta


def load_sentinel_data(sentinel_path):
    """
    Load Sentinel-2 L2A surface reflectance data
    """
    with rasterio.open(sentinel_path) as src:
        sentinel_bands = src.read()
        sentinel_transform = src.transform
        sentinel_crs = src.crs
        sentinel_meta = src.meta
        band_descriptions = src.descriptions
    
    return sentinel_bands, sentinel_transform, sentinel_crs, sentinel_meta, band_descriptions


def load_metadata(metadata_path):
    """
    Load metadata containing sun-sensor geometry
    """
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(metadata_path)
    root = tree.getroot()
    
    # Extract solar and viewing angles from metadata
    # This will need to be adapted based on actual metadata format
    
    # Example extraction (adjust according to actual XML structure)
    solar_zenith = float(root.find('.//SOLAR_ZENITH_ANGLE').text)
    solar_azimuth = float(root.find('.//SOLAR_AZIMUTH_ANGLE').text)
    view_zenith = float(root.find('.//VIEWING_ZENITH_ANGLE').text)
    view_azimuth = float(root.find('.//VIEWING_AZIMUTH_ANGLE').text)
    
    # Calculate relative azimuth
    relative_azimuth = abs(solar_azimuth - view_azimuth)
    if relative_azimuth > 180:
        relative_azimuth = 360 - relative_azimuth
    
    return solar_zenith, view_zenith, relative_azimuth


def align_datasets(modis_data, modis_transform, modis_crs, 
                   sentinel_bands, sentinel_transform, sentinel_crs, sentinel_meta):
    """
    Resample and align MODIS data to Sentinel-2 resolution and grid
    """
    # Create destination array for resampled data
    dst_shape = (sentinel_bands.shape[1], sentinel_bands.shape[2])
    dst_modis = np.zeros(dst_shape, dtype=np.float32)
    
    # Perform reprojection
    reproject(
        source=modis_data,
        destination=dst_modis,
        src_transform=modis_transform,
        src_crs=modis_crs,
        dst_transform=sentinel_transform,
        dst_crs=sentinel_crs,
        resampling=Resampling.bilinear
    )
    
    return dst_modis


def create_spectral_indices(sentinel_bands):
    """
    Calculate spectral indices from Sentinel-2 bands
    Band mapping (approximate):
    - B2 (Blue): index 0
    - B3 (Green): index 1
    - B4 (Red): index 2
    - B8 (NIR): index 3
    - B11 (SWIR1): index 4
    - B12 (SWIR2): index 5
    """
    # Ensure no division by zero
    epsilon = 1e-10
    
    # NDVI (Normalized Difference Vegetation Index)
    ndvi = (sentinel_bands[3] - sentinel_bands[2]) / (sentinel_bands[3] + sentinel_bands[2] + epsilon)
    
    # NDWI (Normalized Difference Water Index)
    ndwi = (sentinel_bands[1] - sentinel_bands[3]) / (sentinel_bands[1] + sentinel_bands[3] + epsilon)
    
    # EVI (Enhanced Vegetation Index)
    evi = 2.5 * ((sentinel_bands[3] - sentinel_bands[2]) / 
                 (sentinel_bands[3] + 6 * sentinel_bands[2] - 7.5 * sentinel_bands[0] + 1 + epsilon))
    
    # SAVI (Soil Adjusted Vegetation Index)
    savi = ((sentinel_bands[3] - sentinel_bands[2]) / 
            (sentinel_bands[3] + sentinel_bands[2] + 0.5 + epsilon)) * 1.5
    
    # BSI (Bare Soil Index)
    bsi = ((sentinel_bands[4] + sentinel_bands[2]) - (sentinel_bands[3] + sentinel_bands[0])) / \
          ((sentinel_bands[4] + sentinel_bands[2]) + (sentinel_bands[3] + sentinel_bands[0]) + epsilon)
    
    # Return stacked indices
    return np.vstack([ndvi, ndwi, evi, savi, bsi])


def create_training_data(aligned_modis, sentinel_bands, spectral_indices=None, mask=None):
    """
    Create training dataset from aligned MODIS albedo and Sentinel-2 bands
    """
    # Stack all features
    if spectral_indices is not None:
        features = np.vstack([sentinel_bands, spectral_indices])
    else:
        features = sentinel_bands.copy()
    
    # Reshape for ML (samples x features)
    X = features.reshape(features.shape[0], -1).T
    y = aligned_modis.flatten()
    
    # Apply mask if provided (e.g., cloud mask)
    if mask is not None:
        mask_flat = mask.flatten()
        X = X[mask_flat, :]
        y = y[mask_flat]
    else:
        # Remove invalid pixels (NaN or negative albedo)
        valid_mask = ~np.isnan(y) & (y >= 0) & (y <= 1)
        X = X[valid_mask, :]
        y = y[valid_mask]
    
    return X, y


def train_models(X, y, use_grid_search=True):
    """
    Train multiple ML models and evaluate performance
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    predictions = {}
    scores = {}
    
    if use_grid_search:
        # Random Forest with hyperparameter tuning
        rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5)
        rf.fit(X_train_scaled, y_train)
        models['RF'] = rf.best_estimator_
        
        # XGBoost with hyperparameter tuning
        xgb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
        xgb_model = GridSearchCV(xgb.XGBRegressor(random_state=42), xgb_params, cv=5)
        xgb_model.fit(X_train_scaled, y_train)
        models['XGB'] = xgb_model.best_estimator_
        
        # SVR with hyperparameter tuning
        svr_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1]}
        svr = GridSearchCV(SVR(), svr_params, cv=5)
        svr.fit(X_train_scaled, y_train)
        models['SVR'] = svr.best_estimator_
        
        # Neural Network with hyperparameter tuning
        nn_params = {'hidden_layer_sizes': [(100,), (100, 50), (50, 25, 10)]}
        nn = GridSearchCV(MLPRegressor(random_state=42, max_iter=1000), nn_params, cv=5)
        nn.fit(X_train_scaled, y_train)
        models['NN'] = nn.best_estimator_
    else:
        # Random Forest without hyperparameter tuning
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        models['RF'] = rf
        
        # XGBoost without hyperparameter tuning
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb_model.fit(X_train_scaled, y_train)
        models['XGB'] = xgb_model
        
        # SVR without hyperparameter tuning
        svr = SVR()
        svr.fit(X_train_scaled, y_train)
        models['SVR'] = svr
        
        # Neural Network without hyperparameter tuning
        nn = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        nn.fit(X_train_scaled, y_train)
        models['NN'] = nn
    
    # Evaluate all models
    for name, model in models.items():
        predictions[name] = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, predictions[name]))
        r2 = r2_score(y_test, predictions[name])
        scores[name] = {'RMSE': rmse, 'R2': r2}
        print(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Find best model
    best_model_name = max(scores, key=lambda k: scores[k]['R2'])
    print(f"Best model: {best_model_name}")
    
    return models, scaler, best_model_name, scores


def generate_albedo(model, scaler, sentinel_data, spectral_indices=None):
    """
    Apply ML model to generate high-resolution albedo from Sentinel-2 data
    """
    # Stack all features
    if spectral_indices is not None:
        features = np.vstack([sentinel_data, spectral_indices])
    else:
        features = sentinel_data.copy()
    
    # Reshape for prediction
    orig_shape = sentinel_data.shape[1:]
    X_new = features.reshape(features.shape[0], -1).T
    
    # Scale features
    X_new_scaled = scaler.transform(X_new)
    
    # Predict
    predicted_albedo = model.predict(X_new_scaled)
    
    # Reshape back to image dimensions
    albedo_map = predicted_albedo.reshape(orig_shape)
    
    return albedo_map


def save_results(albedo_map, output_path, reference_meta):
    """
    Save predicted albedo map as GeoTIFF
    """
    # Update metadata for output
    output_meta = reference_meta.copy()
    output_meta.update({
        'count': 1,
        'dtype': 'float32',
        'nodata': np.nan
    })
    
    with rasterio.open(output_path, 'w', **output_meta) as dst:
        dst.write(albedo_map.astype(np.float32), 1)
    
    print(f"Saved albedo map to {output_path}")


def plot_results(y_true, predictions, output_path=None):
    """
    Create scatterplots of predicted vs true albedo for each model
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(n_models*5, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Add 1:1 line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_true, p(y_true), 'b-')
        
        # Add stats
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f"RMSE: {rmse:.4f}\nR²: {r2:.4f}", 
                transform=ax.transAxes, verticalalignment='top')
        
        ax.set_xlabel('MODIS Albedo')
        ax.set_ylabel(f'Predicted Albedo ({name})')
        ax.set_title(f'{name} Model')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Saved validation plot to {output_path}")
    else:
        plt.show()


def main(modis_path, sentinel_path, metadata_path, output_path, 
         plot_path=None, use_grid_search=False):
    """
    Main workflow for MODIS to Sentinel-2 albedo conversion
    """
    print("Loading MODIS data...")
    modis_data, modis_transform, modis_crs, modis_meta = load_modis_data(modis_path)
    
    print("Loading Sentinel-2 data...")
    sentinel_data, sentinel_transform, sentinel_crs, sentinel_meta, band_descriptions = load_sentinel_data(sentinel_path)
    
    print("Loading metadata...")
    solar_zenith, view_zenith, relative_azimuth = load_metadata(metadata_path)
    
    print("Aligning datasets...")
    aligned_modis = align_datasets(modis_data, modis_transform, modis_crs,
                                  sentinel_data, sentinel_transform, sentinel_crs, sentinel_meta)
    
    print("Calculating spectral indices...")
    spectral_indices = create_spectral_indices(sentinel_data)
    
    print("Creating training data...")
    X, y = create_training_data(aligned_modis, sentinel_data, spectral_indices)
    
    print("Training models...")
    models, scaler, best_model_name, scores = train_models(X, y, use_grid_search)
    
    print("Generating high-resolution albedo...")
    albedo_map = generate_albedo(models[best_model_name], scaler, sentinel_data, spectral_indices)
    
    print("Saving results...")
    save_results(albedo_map, output_path, sentinel_meta)
    
    # Plot validation results
    if plot_path:
        print("Creating validation plots...")
        # Get test predictions for plotting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        
        test_predictions = {}
        for name, model in models.items():
            test_predictions[name] = model.predict(X_test_scaled)
        
        plot_results(y_test, test_predictions, plot_path)
    
    print("Done!")
    return models, scaler, best_model_name, scores


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert MODIS albedo to Sentinel-2 resolution using ML")
    parser.add_argument("modis_path", help="Path to MODIS albedo file")
    parser.add_argument("sentinel_path", help="Path to Sentinel-2 bands file")
    parser.add_argument("metadata_path", help="Path to Sentinel-2 metadata file")
    parser.add_argument("output_path", help="Path for output high-resolution albedo file")
    parser.add_argument("--plot", help="Path for validation plot", default=None)
    parser.add_argument("--grid-search", action="store_true", help="Use grid search for hyperparameter tuning")
    
    args = parser.parse_args()
    
    main(args.modis_path, args.sentinel_path, args.metadata_path, args.output_path,
         args.plot, args.grid_search)