# UTCI Calculator for 3D Models

## Overview

This project enables thermal comfort analysis of outdoor spaces by calculating UTCI values for 3D models. It uses Radiance for solar radiation calculations and the Ladybug Tools library for thermal comfort metrics. The tool can process 3D models in GLTF/GLB format and weather data from EPW files to predict thermal comfort conditions at specific times.

## Features

- Load and analyze 3D models in GLTF/GLB format
- Extract geometries, clean, and convert to Radiance format
- Generate accurate sky models from EPW weather data
- Calculate solar irradiance using Radiance ray-tracing
- Compute UTCI values for outdoor thermal comfort analysis
- Process large models efficiently with batch processing


## Installation

1. Clone this repository
2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Ensure Radiance is installed and accessible from your command line 

## File Formats and Required Inputs

### Input Files

1. **3D Model Files**
   - Supported formats: GLTF (.gltf) or GLB (.glb)
   - The model should consist of triangulated meshes
   - Models can include both surface meshes and line elements (base maps)
   - For best results, use cleaned models with unnecessary elements removed

2. **Weather Data**
   - Required format: EnergyPlus Weather (EPW) files
   - Must include standard weather data: temperature, humidity, wind speed, solar radiation
   - The tool extracts data for the specified hour of year (1-8760)

3. **Output Directory**
   - A directory where calculation results and intermediate files will be saved
   - Will be created if it doesn't exist

### Parameters

- `gltf_path`: Path to the GLTF/GLB model file
- `epw_path`: Path to the EPW weather file
- `output_dir`: Directory for output files
- `hour_of_year`: Hour of the year (1-8760) for which to calculate UTCI
- `ground_albedo`: Ground reflectance (default: 0.2)
- `solar_absorptance`: Solar absorptance of surfaces (default: 0.7)
- `batch_size`: Number of vertices to process in each batch (default: 10000)

## Tools

### GLTF/GLB Analyzer

The repository includes a `gltf_analyzer.py` script for examining the structure of GLTF/GLB files:

```bash
python gltf_analyzer.py path/to/file.glb
```

### GLTF/GLB Cleaner

The `gltf_cleaner.py` script helps examine and clean GLTF/GLB files by removing line geometries (often used for base maps):

```bash
python gltf_cleaner.py path/to/file.glb
```

## Testing

The project includes a comprehensive test suite:

```bash
pytest test_utci_calculator.py -v
```

## How It Works

1. **Model Preparation**: Loads a 3D model in GLTF/GLB format and extracts vertex data
2. **Weather Data**: Extracts weather conditions from an EPW file for the specified hour
3. **Sky Generation**: Creates a Radiance sky model based on solar position and radiation values
4. **Radiance Calculation**: Uses Radiance's rtrace to calculate solar irradiance at each vertex
5. **UTCI Calculation**: Computes UTCI values based on air temperature, mean radiant temperature, humidity, and wind speed