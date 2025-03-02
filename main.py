#!/usr/bin/env python3
"""
Main UTCI Calculator and Visualization Tool for GLTF/GLB models.
This script integrates the UTCI calculation with visualization capabilities.

Hardcoded paths for convenience.
"""

import os
import time
import logging
from pathlib import Path

# Import the UTCI calculator module
import utci_calculator
from utci_calculator import (
    gltf_to_honeybee_model,
    calculate_utci_from_honeybee_model,
    calculate_utci_from_gltf_epw,
    set_log_level
)

# Import the visualization module
from utci_visualization import create_utci_visualization_set

# Hardcoded paths (matching the original script)
CURRENT_DIR = Path.cwd()
GLB_FILE = CURRENT_DIR / "data/rec_model_no_curve.glb"
EPW_FILE = CURRENT_DIR / "data/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw"
OUTPUT_DIRECTORY = CURRENT_DIR / "output"


def save_sensor_positions(sensor_grid, output_file):
    """Save sensor positions to a CSV file for visualization."""
    with open(output_file, "w") as f:
        f.write("x,y,z\n")
        for sensor in sensor_grid.sensors:
            try:
                # Try multiple access methods for compatibility
                if hasattr(sensor, 'pos') and hasattr(sensor.pos, 'x'):
                    pos = sensor.pos
                    f.write(f"{pos.x},{pos.y},{pos.z}\n")
                elif hasattr(sensor, 'pos') and isinstance(sensor.pos, (list, tuple)):
                    f.write(f"{sensor.pos[0]},{sensor.pos[1]},{sensor.pos[2]}\n")
                else:
                    # Try dictionary access
                    f.write(f"{sensor['pos'][0]},{sensor['pos'][1]},{sensor['pos'][2]}\n")
            except Exception as e:
                logging.warning(f"Could not process sensor position: {e}")
                # Use a placeholder if all methods fail
                f.write("0,0,0\n")


def run_utci_analysis_with_visualization(
    gltf_path=GLB_FILE, 
    epw_path=EPW_FILE, 
    output_dir=OUTPUT_DIRECTORY, 
    hour_of_year=12,
    grid_size=0.5,         # Smaller grid size for better ground-level resolution
    offset=0.1,
    solar_absorptance=0.7,
    clean_geometry=True,
    use_centroids=True,
    max_sensors=20000,     # More sensors for better coverage
    visualize=True,
    views=None,
    height_filter=1.5      # Height threshold for ground level (pedestrian height)
):
    """
    Run a complete UTCI analysis including visualization.
    
    Args:
        gltf_path: Path to GLTF/GLB file.
        epw_path: Path to EPW weather file.
        output_dir: Directory to save outputs.
        hour_of_year: Hour of the year to analyze (1-8760).
        grid_size: Size of the sensor grid cells.
        offset: Distance to offset sensors from surfaces.
        solar_absorptance: Solar absorptance of surfaces (0-1).
        clean_geometry: Whether to clean the geometry.
        use_centroids: Use centroids as fallback for meshing.
        max_sensors: Maximum number of sensor points.
        visualize: Whether to create visualizations.
        views: List of views to generate ('top', 'front', 'side', 'combined').
    
    Returns:
        Dictionary with paths to results and visualizations.
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up result paths
    result_paths = {
        "utci_csv": str(output_dir / "utci_results.csv"),
        "utci_txt": str(output_dir / "utci_values.txt"),
        "sensor_positions": str(output_dir / "sensor_positions.csv"),
        "visualizations": {}
    }
    
    # Convert GLTF to Honeybee Model
    logging.info(f"Loading and converting GLTF file: {gltf_path}")
    start_time = time.time()
    hb_model = gltf_to_honeybee_model(gltf_path, min_area=1e-6, clean_geometry=clean_geometry)
    logging.info(f"Conversion completed in {time.time() - start_time:.2f} seconds.")
    
    # Run UTCI calculation
    logging.info(f"Starting UTCI calculation for hour {hour_of_year}...")
    start_time = time.time()
    
    # The main UTCI calculation function from the original script
    utci_values = calculate_utci_from_honeybee_model(
        hb_model,
        epw_path,
        output_dir,
        hour_of_year,
        grid_size=grid_size,
        offset=offset,
        solar_absorptance=solar_absorptance,
        use_centroids=use_centroids,
        max_sensors=max_sensors
    )
    
    # Get the sensor grid (this would be cleaner if the original function returned it)
    # For now, we need to create it again, which is inefficient but works
    sensor_grid = utci_calculator.create_sensor_grid(
        hb_model, 
        grid_size=grid_size, 
        offset=offset, 
        use_centroids=use_centroids, 
        max_sensors=max_sensors
    )
    
    calculation_time = time.time() - start_time
    logging.info(f"UTCI calculation completed in {calculation_time:.2f} seconds.")
    
    # Save sensor positions for visualization
    save_sensor_positions(sensor_grid, result_paths["sensor_positions"])
    logging.info(f"Saved sensor positions to: {result_paths['sensor_positions']}")
    
    # Create visualizations if requested
    if visualize:
        try:
            if views is None:
                views = ['top', 'front', 'side', 'combined']
                
            logging.info(f"Creating visualizations for views: {', '.join(views)}")
            start_time = time.time()
            
            visualization_paths = create_utci_visualization_set(
                result_paths["utci_csv"], 
                result_paths["sensor_positions"],
                output_dir,
                views=views,
                focus_ground_level=True,
                height_threshold=1.5  # Consider points below 1.5m as "ground level" for pedestrians
            )
            
            result_paths["visualizations"] = visualization_paths
            logging.info(f"Visualizations created in {time.time() - start_time:.2f} seconds.")
            
            # Print visualization paths
            for view, path in visualization_paths.items():
                logging.info(f"  {view.capitalize()} view: {path}")
                
        except ImportError as e:
            logging.warning(f"Could not create visualizations. Required packages not installed: {e}")
        except Exception as e:
            logging.error(f"Error creating visualizations: {e}")
            import traceback
            logging.debug(traceback.format_exc())
    
    return result_paths


def main():
    """Main function to run the UTCI analysis with hardcoded parameters."""
    # Set logging level
    set_log_level("INFO")
    
    # Define parameters (can be modified here instead of command line arguments)
    params = {
        "gltf_path": GLB_FILE,
        "epw_path": EPW_FILE,
        "output_dir": OUTPUT_DIRECTORY,
        "hour_of_year": 12,  # Noon
        "grid_size": 0.5,     # Smaller grid size for better resolution at ground level
        "offset": 0.1,
        "solar_absorptance": 0.7,
        "clean_geometry": True,
        "use_centroids": True,
        "max_sensors": 20000,  # More sensors for better ground coverage
        "visualize": True,
        "views": ["top"]      # Only generate top view for pedestrian-level comfort
    }
    
    logging.info(f"Starting Pedestrian-Level UTCI Analysis")
    logging.info(f"Model: {params['gltf_path']}")
    logging.info(f"EPW: {params['epw_path']}")
    logging.info(f"Output: {params['output_dir']}")
    logging.info(f"Focus: Ground-level pedestrian thermal comfort (height ≤ 1.5m)")
    
    # Run the analysis
    try:
        start_time = time.time()
        
        result_paths = run_utci_analysis_with_visualization(**params)
        
        total_time = time.time() - start_time
        logging.info(f"UTCI analysis and visualization completed in {total_time:.2f} seconds.")
        logging.info(f"Results saved to: {params['output_dir']}")
        
    except Exception as e:
        logging.error(f"Error in UTCI analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())