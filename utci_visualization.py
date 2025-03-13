# python utci_visualization.py output/utci_results.csv
#!/usr/bin/env python3
"""UTCI Visualization Tools for Ground-Level Thermal Comfort Analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pathlib import Path
import json
import os
import logging
import argparse
from mpl_toolkits.mplot3d import Axes3D

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def debug_sensor_positions(sensor_positions_file, output_dir):
    """
    Debug and fix issues with sensor positions, especially z-value problems.
    
    Args:
        sensor_positions_file: Path to CSV file with sensor positions
        output_dir: Directory to save diagnostic plots
    
    Returns:
        Fixed sensor positions array
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sensor positions
    df = pd.read_csv(sensor_positions_file)
    
    # Check for expected column names
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        # Try to use the first three columns
        if len(df.columns) >= 3:
            print(f"Column names not as expected: {df.columns}")
            # Rename the first three columns to x, y, z
            column_mapping = {df.columns[0]: 'x', df.columns[1]: 'y', df.columns[2]: 'z'}
            df = df.rename(columns=column_mapping)
            print(f"Renamed columns to: {df.columns}")
        else:
            raise ValueError(f"Not enough columns in {sensor_positions_file}")
    
    # Create a 3D scatter plot to visualize the sensor positions
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot sensors as a scatter plot
    scatter = ax.scatter(df['x'], df['y'], df['z'], c=df['z'], cmap='viridis', 
                         s=10, alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Sensor Positions')
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Z Coordinate (m)')
    
    # Save the figure
    plt.savefig(output_dir / "sensor_positions_3d.png", dpi=300, bbox_inches='tight')
    
    # Print z-value statistics
    z_min = df['z'].min()
    z_max = df['z'].max()
    z_mean = df['z'].mean()
    z_std = df['z'].std()
    
    print(f"Z-value statistics:")
    print(f"  Min: {z_min:.2f}m")
    print(f"  Max: {z_max:.2f}m")
    print(f"  Mean: {z_mean:.2f}m")
    print(f"  Std: {z_std:.2f}m")
    
    # Check for extreme outliers
    z_range = z_max - z_min
    
    # Create histogram of z-values
    plt.figure(figsize=(10, 6))
    plt.hist(df['z'], bins=50)
    plt.axvline(z_mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {z_mean:.2f}m')
    plt.axvline(z_min, color='g', linestyle='dashed', linewidth=1, label=f'Min: {z_min:.2f}m')
    plt.axvline(z_max, color='purple', linestyle='dashed', linewidth=1, label=f'Max: {z_max:.2f}m')
    plt.xlabel('Z Coordinate (m)')
    plt.ylabel('Count')
    plt.title('Distribution of Z Values')
    plt.legend()
    plt.savefig(output_dir / "z_value_histogram.png", dpi=300, bbox_inches='tight')
    
    # If extreme outliers are detected, fix them
    outlier_threshold = z_mean + 3*z_std
    
    if z_range > 10 or z_max > outlier_threshold:
        print(f"Extreme z-value range detected: {z_range:.2f}m")
        print(f"Possible outliers above: {outlier_threshold:.2f}m")
        
        # Count potential outliers
        outlier_count = sum(df['z'] > outlier_threshold)
        outlier_percentage = (outlier_count / len(df)) * 100
        print(f"Potential outliers: {outlier_count} ({outlier_percentage:.1f}%)")
        
        if outlier_percentage < 5:
            # If less than 5% are outliers, remove them
            print(f"Removing {outlier_count} outlier points")
            df_fixed = df[df['z'] <= outlier_threshold].copy()
        else:
            # If many outliers, probably a scale issue - normalize z coordinates
            print("Many outliers detected - attempting scale normalization")
            df_fixed = df.copy()
            
            # Reset z-values to be within a reasonable range
            z_reasonable_max = z_min + 5.0  # Assume most buildings/models aren't more than 5m tall
            
            # Apply scaling to compress z-values
            df_fixed['z'] = z_min + (df['z'] - z_min) * (z_reasonable_max - z_min) / (z_max - z_min)
            
            # Report statistics after scaling
            z_fixed_min = df_fixed['z'].min()
            z_fixed_max = df_fixed['z'].max()
            z_fixed_range = z_fixed_max - z_fixed_min
            print(f"Fixed z-value range: {z_fixed_range:.2f}m ({z_fixed_min:.2f}m to {z_fixed_max:.2f}m)")
    else:
        print("Z-value range looks reasonable")
        df_fixed = df
    
    # Save fixed positions
    fixed_path = output_dir / "sensor_positions_fixed.csv"
    df_fixed.to_csv(fixed_path, index=False)
    print(f"Fixed sensor positions saved to: {fixed_path}")
    
    # Create a 3D scatter plot of the fixed positions
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot fixed sensors
    scatter = ax.scatter(df_fixed['x'], df_fixed['y'], df_fixed['z'], c=df_fixed['z'], cmap='viridis', 
                         s=10, alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Fixed 3D Sensor Positions')
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Z Coordinate (m)')
    
    # Save the figure
    plt.savefig(output_dir / "sensor_positions_3d_fixed.png", dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    return df_fixed.values


def visualize_utci_results(utci_values, sensor_positions, output_dir, 
                          filename="utci_map.png", view='top', 
                          resolution=1000, interpolation='linear',
                          focus_ground_level=True, height_threshold=1.5):
    """
    Create a UTCI visualization map as a PNG image with improved ground-level filtering.
    
    Args:
        utci_values: Array of UTCI values in Celsius.
        sensor_positions: Array of (x, y, z) coordinates corresponding to each UTCI value.
        output_dir: Directory to save the output image.
        filename: Name of the output PNG file.
        view: Perspective for 2D projection ('top', 'front', or 'side').
        resolution: Resolution of the output image in pixels.
        interpolation: Interpolation method ('linear', 'nearest', etc.).
        focus_ground_level: Whether to filter for ground-level points.
        height_threshold: Height in meters above minimum z to consider as ground level.
    
    Returns:
        Path to the created visualization image.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    # Convert inputs to numpy arrays if they aren't already
    utci_values = np.array(utci_values)
    sensor_positions = np.array(sensor_positions)
    
    # Log initial z-value distribution for diagnostics
    z_values = sensor_positions[:, 2]
    logging.info(f"Initial z-value distribution: min={z_values.min():.2f}m, max={z_values.max():.2f}m, mean={z_values.mean():.2f}m")
    
    # Filter for ground level if requested and it's a top view
    if focus_ground_level and view == 'top':
        # Find the minimum z-value as a reference for "ground level"
        min_z = z_values.min()
        
        # Set ground threshold based on minimum z plus the height threshold
        ground_threshold = min_z + height_threshold
        ground_mask = z_values <= ground_threshold
        
        ground_count = sum(ground_mask)
        logging.info(f"Ground points (z ≤ {ground_threshold:.2f}m): {ground_count} out of {len(z_values)}")
        
        # If we have too few ground points, use all points
        if ground_count < 10:  # Arbitrary threshold
            logging.warning(f"Too few ground-level points (only {ground_count}). Using all points.")
        else:
            # Filter to only use ground-level points
            sensor_positions = sensor_positions[ground_mask]
            utci_values = utci_values[ground_mask]
            logging.info(f"Focusing on {ground_count} ground-level points (z ≤ {ground_threshold:.2f}m)")
            
            # Log filtered z-value distribution
            z_values = sensor_positions[:, 2]
            logging.info(f"Filtered z-value distribution: min={z_values.min():.2f}m, max={z_values.max():.2f}m, mean={z_values.mean():.2f}m")
    
    # Determine which coordinates to use based on view
    if view == 'top':
        x_idx, y_idx = 0, 1  # x, y coordinates (top view)
    elif view == 'front':
        x_idx, y_idx = 0, 2  # x, z coordinates (front view)
    elif view == 'side':
        x_idx, y_idx = 1, 2  # y, z coordinates (side view)
    else:
        raise ValueError(f"Invalid view: {view}. Choose 'top', 'front', or 'side'.")
    
    # Extract the relevant coordinates
    x = sensor_positions[:, x_idx]
    y = sensor_positions[:, y_idx]
    
    # Log coordinate range for debugging
    logging.info(f"{view.capitalize()} view: x range [{x.min():.2f}, {x.max():.2f}], y range [{y.min():.2f}, {y.max():.2f}]")
    
    # Check if we have enough unique points for interpolation
    unique_points = np.unique(np.column_stack((x, y)), axis=0)
    if len(unique_points) < 4:
        logging.warning(f"Not enough unique points for interpolation (only {len(unique_points)}). Using scatter plot only.")
        use_interpolation = False
    else:
        use_interpolation = True
    
    # Calculate the bounds of the visualization
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Add a small margin
    margin = 0.05
    x_margin = max(0.1, (x_max - x_min) * margin) # At least 0.1m margin
    y_margin = max(0.1, (y_max - y_min) * margin) # At least 0.1m margin
    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin
    
    # Create a custom colormap for UTCI values
    # Based on standard UTCI thermal stress categories
    utci_colors = [
        (0.0, 'darkblue'),     # -40°C: Extreme cold stress
        (0.1, 'blue'),         # -27°C: Very strong cold stress
        (0.2, 'skyblue'),      # -13°C: Strong cold stress
        (0.3, 'lightblue'),    # 0°C: Moderate cold stress
        (0.4, 'palegreen'),    # 9°C: Slight cold stress
        (0.5, 'green'),        # 18°C: No thermal stress
        (0.6, 'yellow'),       # 26°C: Moderate heat stress
        (0.7, 'orange'),       # 32°C: Strong heat stress
        (0.8, 'red'),          # 38°C: Very strong heat stress
        (0.9, 'darkred'),      # 46°C: Extreme heat stress
        (1.0, 'purple')        # >46°C: Beyond extreme heat stress
    ]
    utci_cmap = LinearSegmentedColormap.from_list('utci', utci_colors)
    
    # Calculate min and max for color scaling
    # Use standard UTCI range with some buffer
    vmin = max(-40, utci_values.min() - 5)
    vmax = min(50, utci_values.max() + 5)
    
    # Create the figure
    plt.figure(figsize=(12, 10))
    
    # If we have enough points for interpolation, create a contour plot
    if use_interpolation:
        # Create a grid for interpolation
        grid_resolution = resolution
        xi = np.linspace(x_min, x_max, grid_resolution)
        yi = np.linspace(y_min, y_max, grid_resolution)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate UTCI values onto the grid using only x and y coordinates
        from scipy.interpolate import griddata
        zi_grid = griddata((x, y), utci_values, (xi_grid, yi_grid), method=interpolation, fill_value=np.nan)
        
        # Create the contour plot with higher resolution for pedestrian comfort maps
        contour_levels = 30 if view == 'top' and focus_ground_level else 20
        contour = plt.contourf(xi_grid, yi_grid, zi_grid, contour_levels, cmap=utci_cmap, 
                              vmin=vmin, vmax=vmax, extend='both', alpha=0.9)
    
    # Add scatter plot of sensor positions
    scatter_size = 25 if len(utci_values) < 100 else (10 if view == 'top' and focus_ground_level else 5)
    scatter = plt.scatter(x, y, c=utci_values, s=scatter_size, 
                         cmap=utci_cmap, vmin=vmin, vmax=vmax, 
                         edgecolors='k', linewidths=0.3, alpha=0.7)
    
    # Add a colorbar - use the contour if available, otherwise the scatter
    if use_interpolation:
        cbar = plt.colorbar(contour, label='UTCI (°C)')
    else:
        cbar = plt.colorbar(scatter, label='UTCI (°C)')
    
    # Add UTCI thermal stress category labels to the colorbar
    stress_categories = [
        (-40, "Extreme cold stress"),
        (-27, "Very strong cold stress"),
        (-13, "Strong cold stress"),
        (0, "Moderate cold stress"),
        (9, "Slight cold stress"),
        (26, "No thermal stress"),
        (32, "Moderate heat stress"),
        (38, "Strong heat stress"),
        (46, "Very strong heat stress"),
        (50, "Extreme heat stress")
    ]
    
    # Only add labels that are within our visualized range
    tick_positions = []
    tick_labels = []
    for temp, label in stress_categories:
        if vmin <= temp <= vmax:
            tick_positions.append(temp)
            tick_labels.append(f"{temp}°C: {label}")
    
    if tick_positions:
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
    
    # For top view focusing on pedestrians, add a more prominent title
    if view == 'top' and focus_ground_level:
        plt.title(f"Pedestrian-Level Thermal Comfort (Height ≤ {height_threshold}m)", fontsize=14, fontweight='bold')
    else:
        plt.title(f"UTCI Map - {view.capitalize()} View")
    
    # Add labels
    plt.xlabel(f"{'X' if x_idx == 0 else 'Y'} coordinate (m)")
    plt.ylabel(f"{'Y' if y_idx == 1 else 'Z'} coordinate (m)")
    
    # Add statistics as text
    stats_text = (
        f"Min: {utci_values.min():.1f}°C\n"
        f"Max: {utci_values.max():.1f}°C\n"
        f"Mean: {utci_values.mean():.1f}°C\n"
        f"Points: {len(utci_values)}"
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add equal aspect ratio for more accurate spatial representation
    plt.axis('equal')
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def create_utci_visualization_set(utci_csv_path, sensor_positions_file=None, output_dir=None, views=None, 
                                focus_ground_level=True, height_threshold=1.5, debug=False):
    """
    Create a set of UTCI visualizations from results.
    
    Args:
        utci_csv_path: Path to the CSV file with UTCI results.
        sensor_positions_file: Path to file with sensor positions (if None, will try to infer).
        output_dir: Directory to save visualizations (if None, will use same directory as CSV).
        views: List of views to generate ('top', 'front', 'side', 'combined'). If None, all views.
        focus_ground_level: Whether to focus on ground-level points for pedestrian thermal comfort.
        height_threshold: Height in meters below which points are considered at "ground level".
        debug: If True, run diagnostic checks on sensor positions.
    
    Returns:
        Dictionary with paths to the created visualization images.
    """
    # Prepare paths
    utci_path = Path(utci_csv_path)
    if output_dir is None:
        output_dir = utci_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load UTCI values
    try:
        df = pd.read_csv(utci_path)
        
        # Check if we have the expected column names
        if 'utci_celsius' in df.columns:
            utci_values = df['utci_celsius'].values
        else:
            # Try the second column as UTCI values
            utci_values = df.iloc[:, 1].values
            logging.info(f"Using column '{df.columns[1]}' for UTCI values")
    except Exception as e:
        logging.error(f"Error loading UTCI values: {e}")
        raise
    
    # Try to find sensor positions file if not provided
    if sensor_positions_file is None:
        # Look for sensor positions in same directory with common naming patterns
        possible_files = [
            utci_path.parent / "sensor_positions.csv",
            utci_path.parent / "sensor_positions_fixed.csv",  # Look for fixed positions first
            utci_path.parent / "sensor_positions.json",
            utci_path.parent / "sensors.pts",
            utci_path.parent / (utci_path.stem.replace("utci_results", "sensor_positions") + ".csv")
        ]
        
        for file in possible_files:
            if file.exists():
                sensor_positions_file = file
                logging.info(f"Found sensor positions file: {file}")
                break
    
    # If we still don't have sensor positions, generate synthetic ones for demonstration
    if sensor_positions_file is None or not Path(sensor_positions_file).exists():
        logging.warning(f"No sensor positions file found. Creating synthetic positions for visualization purposes.")
        # Create grid-like positions for demonstration
        n = len(utci_values)
        side_length = int(np.sqrt(n))
        remainder = n - (side_length * side_length)
        
        x = np.tile(np.arange(side_length), side_length)
        y = np.repeat(np.arange(side_length), side_length)
        
        # Handle remaining points
        if remainder > 0:
            extra_x = np.arange(remainder)
            extra_y = np.repeat(side_length, remainder)
            x = np.concatenate([x, extra_x])
            y = np.concatenate([y, extra_y])
        
        # Add small variations to make visualization more interesting
        np.random.seed(42)  # For reproducibility
        x = x + np.random.normal(0, 0.1, size=n)
        y = y + np.random.normal(0, 0.1, size=n)
        z = np.zeros(n)  # Set z to zero for synthetic positions
        
        sensor_positions = np.column_stack((x, y, z))
    else:
        # Load sensor positions from file
        sensor_positions_file = Path(sensor_positions_file)
        
        if sensor_positions_file.suffix == '.json':
            with open(sensor_positions_file, 'r') as f:
                data = json.load(f)
                # Extract positions based on common JSON structures
                if isinstance(data, list) and 'pos' in data[0]:
                    sensor_positions = np.array([item['pos'] for item in data])
                elif 'positions' in data:
                    sensor_positions = np.array(data['positions'])
                else:
                    raise ValueError(f"Unsupported JSON format in {sensor_positions_file}")
        
        elif sensor_positions_file.suffix == '.csv':
            df_pos = pd.read_csv(sensor_positions_file)
            # Try to identify position columns
            if all(col in df_pos.columns for col in ['x', 'y', 'z']):
                sensor_positions = df_pos[['x', 'y', 'z']].values
            else:
                # Try first three columns
                sensor_positions = df_pos.iloc[:, 0:3].values
        
        elif sensor_positions_file.suffix == '.pts':
            # Parse Radiance pts file format
            positions = []
            with open(sensor_positions_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        values = line.strip().split()
                        if len(values) >= 3:
                            positions.append([float(values[0]), float(values[1]), float(values[2])])
            sensor_positions = np.array(positions)
        
        else:
            raise ValueError(f"Unsupported file format: {sensor_positions_file.suffix}")
    
    # Run diagnostic checks if requested
    if debug:
        try:
            logging.info("Running diagnostic checks on sensor positions...")
            fixed_positions = debug_sensor_positions(sensor_positions_file, output_dir)
            # Use the fixed positions for visualization
            sensor_positions = fixed_positions
            logging.info("Using fixed sensor positions from diagnostic check")
        except Exception as e:
            logging.error(f"Error during diagnostic checks: {e}")
            logging.info("Continuing with original sensor positions")
    
    # Ensure we have the right number of positions
    if len(sensor_positions) != len(utci_values):
        logging.warning(
            f"Number of sensor positions ({len(sensor_positions)}) "
            f"doesn't match number of UTCI values ({len(utci_values)})"
        )
        
        # Adjust by truncating the longer array
        if len(sensor_positions) > len(utci_values):
            logging.warning("Truncating sensor positions to match UTCI values")
            sensor_positions = sensor_positions[:len(utci_values)]
        else:
            logging.warning("Truncating UTCI values to match sensor positions")
            utci_values = utci_values[:len(sensor_positions)]
    
    # Check for NaN values in UTCI data and handle them
    nan_mask = np.isnan(utci_values)
    if np.any(nan_mask):
        nan_count = np.sum(nan_mask)
        logging.warning(f"Found {nan_count} NaN values in UTCI data. Removing these points.")
        sensor_positions = sensor_positions[~nan_mask]
        utci_values = utci_values[~nan_mask]
    
    # Determine which views to create
    if views is None:
        views = ['top', 'front', 'side', 'combined']
    
    # Create visualizations for different views
    output_files = {}
    for view in ['top', 'front', 'side']:
        if view in views:
            filename = f"utci_map_{view}.png"
            if view == 'top' and 'top' in views:
                # For top view, use ground-level focus if requested
                output_path = visualize_utci_results(
                    utci_values, 
                    sensor_positions, 
                    output_dir,
                    filename=filename,
                    view=view,
                    focus_ground_level=focus_ground_level,
                    height_threshold=height_threshold,
                    resolution=1500  # Higher resolution for top view
                )
            else:
                output_path = visualize_utci_results(
                    utci_values, 
                    sensor_positions, 
                    output_dir,
                    filename=filename,
                    view=view
                )
            output_files[view] = output_path
    
    # Create a combined view if requested
        if 'combined' in views:
            combined_output_path = output_dir / "utci_combined_views.png"
            
            # Create a figure with 3 subplots
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # Set up common parameters
            view_options = ['top', 'front', 'side']
            labels = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
            vmin = max(-40, utci_values.min() - 5)
            vmax = min(50, utci_values.max() + 5)
            utci_colors = [
                (0.0, 'darkblue'),     # -40°C
                (0.1, 'blue'),         # -27°C
                (0.2, 'skyblue'),      # -13°C
                (0.3, 'lightblue'),    # 0°C
                (0.4, 'palegreen'),    # 9°C
                (0.5, 'green'),        # 18°C
                (0.6, 'yellow'),       # 26°C
                (0.7, 'orange'),       # 32°C
                (0.8, 'red'),          # 38°C
                (0.9, 'darkred'),      # 46°C
                (1.0, 'purple')        # >46°C
            ]
            utci_cmap = LinearSegmentedColormap.from_list('utci', utci_colors)
            
            # Define stress categories for the colorbar
            stress_categories = [
                (-40, "Extreme cold stress"),
                (-27, "Very strong cold stress"),
                (-13, "Strong cold stress"),
                (0, "Moderate cold stress"),
                (9, "Slight cold stress"),
                (26, "No thermal stress"),
                (32, "Moderate heat stress"),
                (38, "Strong heat stress"),
                (46, "Very strong heat stress"),
                (50, "Extreme heat stress")
            ]
        

        
        # Create each subplot
        for i, (view, (xlabel, ylabel)) in enumerate(zip(view_options, labels)):
            ax = axs[i]
            
            # Determine which coordinates to use based on view
            if view == 'top':
                x_idx, y_idx = 0, 1  # x, y coordinates
                
                # For top view, filter to ground level if requested
                if focus_ground_level:
                    # Find the minimum z-value as a reference for "ground level"
                    min_z = sensor_positions[:, 2].min()
                    ground_threshold = min_z + height_threshold
                    ground_mask = sensor_positions[:, 2] <= ground_threshold
                    
                    if sum(ground_mask) >= 10:  # Only filter if we have enough ground points
                        plot_positions = sensor_positions[ground_mask]
                        plot_utci = utci_values[ground_mask]
                    else:
                        plot_positions = sensor_positions
                        plot_utci = utci_values
                else:
                    plot_positions = sensor_positions
                    plot_utci = utci_values
                    
            elif view == 'front':
                x_idx, y_idx = 0, 2  # x, z coordinates
                plot_positions = sensor_positions
                plot_utci = utci_values
            else:  # side
                x_idx, y_idx = 1, 2  # y, z coordinates
                plot_positions = sensor_positions
                plot_utci = utci_values
            
            # Extract coordinates
            x = plot_positions[:, x_idx]
            y = plot_positions[:, y_idx]
            
            # Create scatter plot
            scatter = ax.scatter(x, y, c=plot_utci, s=10, cmap=utci_cmap, 
                               vmin=vmin, vmax=vmax, alpha=0.7)
            
            # Add labels
            ax.set_xlabel(f"{xlabel} coordinate (m)")
            ax.set_ylabel(f"{ylabel} coordinate (m)")
            
            # Add title - mention ground level filtering for top view
            if view == 'top' and focus_ground_level and sum(ground_mask) >= 10:
                ax.set_title(f"Top View (z ≤ {ground_threshold:.1f}m)")
            else:
                ax.set_title(f"{view.capitalize()} View")
            
            # Equal aspect ratio for proper visualization
            ax.set_aspect('equal')
        
        # Add a colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('UTCI (°C)')
        
        # Add stress category labels to colorbar
        tick_positions = []
        tick_labels = []
        for temp, label in stress_categories:
            if vmin <= temp <= vmax:
                tick_positions.append(temp)
                tick_labels.append(f"{temp}°C")
        
        if tick_positions:
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)
        
        # Add main title
        plt.suptitle("UTCI Thermal Comfort Analysis", fontsize=16)
        
        # Add stats text box
        stats_text = (
            f"Min: {utci_values.min():.1f}°C\n"
            f"Max: {utci_values.max():.1f}°C\n"
            f"Mean: {utci_values.mean():.1f}°C\n"
            f"Points: {len(utci_values)}"
        )
        fig.text(0.01, 0.02, stats_text, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
        plt.savefig(combined_output_path, dpi=300)
        plt.close()
        
        output_files['combined'] = str(combined_output_path)
    
    # Create a 3D visualization if requested
    if '3d' in views:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            sensor_positions[:, 0], 
            sensor_positions[:, 1], 
            sensor_positions[:, 2],
            c=utci_values, 
            cmap=utci_cmap,
            vmin=vmin, 
            vmax=vmax,
            s=15,
            alpha=0.8
        )
        
        # Add labels
        ax.set_xlabel('X coordinate (m)')
        ax.set_ylabel('Y coordinate (m)')
        ax.set_zlabel('Z coordinate (m)')
        ax.set_title('3D UTCI Visualization')
        
        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('UTCI (°C)')
        
        # Add stress category labels to colorbar
        tick_positions = []
        tick_labels = []
        for temp, label in stress_categories:
            if vmin <= temp <= vmax:
                tick_positions.append(temp)
                tick_labels.append(f"{temp}°C")
        
        if tick_positions:
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)
        
        # Save the figure
        output_path = output_dir / "utci_map_3d.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        output_files['3d'] = str(output_path)
    
    return output_files


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create UTCI visualizations")
    parser.add_argument("utci_file", help="Path to UTCI results CSV file")
    parser.add_argument("--positions", help="Path to sensor positions file (CSV, JSON or PTS)")
    parser.add_argument("--output", help="Output directory for visualizations")
    parser.add_argument("--views", help="Comma-separated list of views to generate (top,front,side,combined,3d)")
    parser.add_argument("--height", type=float, default=1.5, help="Height threshold for ground level (default: 1.5m)")
    parser.add_argument("--no-ground-focus", action="store_true", help="Disable ground level focus")
    parser.add_argument("--debug", action="store_true", help="Run diagnostic checks on sensor positions")
    
    args = parser.parse_args()
    
    # Set up views
    if args.views:
        views = args.views.split(',')
    else:
        views = ['top', 'front', 'side', 'combined']
        
    focus_ground_level = not args.no_ground_focus
    
    try:
        output_files = create_utci_visualization_set(
            args.utci_file,
            args.positions,
            args.output,
            views=views,
            focus_ground_level=focus_ground_level,
            height_threshold=args.height,
            debug=args.debug
        )
        print(f"Created UTCI visualizations:")
        for view, path in output_files.items():
            print(f"  {view.capitalize()} view: {path}")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()