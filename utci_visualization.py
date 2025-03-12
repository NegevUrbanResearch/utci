# python utci_visualization.py /Users/noamgal/DSProjects/utci/output/utci_results.csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pathlib import Path
import json
import os
import logging

def visualize_utci_results(utci_values, sensor_positions, output_dir, 
                          filename="utci_map.png", view='top', 
                          resolution=1000, interpolation='linear',
                          focus_ground_level=True, height_threshold=1.5):
    """
    Create a UTCI visualization map as a PNG image.
    
    Args:
        utci_values: Array of UTCI values in Celsius.
        sensor_positions: Array of (x, y, z) coordinates corresponding to each UTCI value.
        output_dir: Directory to save the output image.
        filename: Name of the output PNG file.
        view: Perspective for 2D projection ('top', 'front', or 'side').
        resolution: Resolution of the output image in pixels.
        interpolation: Interpolation method ('linear', 'nearest', etc.).
    
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
    
    # Filter for ground level if requested and it's a top view
    if focus_ground_level and view == 'top':
        # Find points that are close to ground level (assuming z coordinate is at index 2)
        z_values = sensor_positions[:, 2]
        ground_mask = z_values <= height_threshold
        
        # If we have too few ground points, use all points
        if sum(ground_mask) < 10:  # Arbitrary threshold
            logging.warning(f"Too few ground-level points (only {sum(ground_mask)} below {height_threshold}m). Using all points.")
        else:
            # Filter to only use ground-level points
            sensor_positions = sensor_positions[ground_mask]
            utci_values = utci_values[ground_mask]
            logging.info(f"Focusing on {sum(ground_mask)} ground-level points (below {height_threshold}m)")
    
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
    
    # Calculate the bounds of the visualization
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Add a small margin
    margin = 0.05
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin
    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin
    
    # Create a grid for interpolation
    grid_resolution = resolution
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate UTCI values onto the grid
    from scipy.interpolate import griddata
    zi_grid = griddata((x, y), utci_values, (xi_grid, yi_grid), method=interpolation)
    
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
    
    # Create the contour plot with higher resolution for pedestrian comfort maps
    contour_levels = 30 if view == 'top' and focus_ground_level else 20
    contour = plt.contourf(xi_grid, yi_grid, zi_grid, contour_levels, cmap=utci_cmap, 
                          vmin=vmin, vmax=vmax, extend='both', alpha=0.9)
    
    # Add a colorbar
    cbar = plt.colorbar(contour, label='UTCI (°C)')
    
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
    
    # Add scatter plot of sensor positions
    plt.scatter(x, y, c=utci_values, s=10 if view == 'top' and focus_ground_level else 5, 
               cmap=utci_cmap, vmin=vmin, vmax=vmax, 
               edgecolors='k', linewidths=0.3, alpha=0.7)
    
    # For top view focusing on pedestrians, add a more prominent title
    if view == 'top' and focus_ground_level:
        plt.title(f"Pedestrian-Level Thermal Comfort (Height ≤ {height_threshold}m)", fontsize=14, fontweight='bold')
    
    # Add labels and title
    plt.xlabel(f"{'X' if x_idx == 0 else 'Y'} coordinate (m)")
    plt.ylabel(f"{'Y' if y_idx == 1 else 'Z'} coordinate (m)")
    plt.title(f"UTCI Map - {view.capitalize()} View")
    
    # Add statistics as text
    stats_text = (
        f"Min: {utci_values.min():.1f}°C\n"
        f"Max: {utci_values.max():.1f}°C\n"
        f"Mean: {utci_values.mean():.1f}°C"
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def create_utci_visualization_set(utci_csv_path, sensor_positions_file=None, output_dir=None, views=None, 
                              focus_ground_level=True, height_threshold=1.5):
    """
    Create a set of UTCI visualizations from results.
    
    Args:
        utci_csv_path: Path to the CSV file with UTCI results.
        sensor_positions_file: Path to file with sensor positions (if None, will try to infer).
        output_dir: Directory to save visualizations (if None, will use same directory as CSV).
        views: List of views to generate ('top', 'front', 'side', 'combined'). If None, all views are generated.
        focus_ground_level: Whether to focus on ground-level points for pedestrian thermal comfort.
        height_threshold: Height in meters below which points are considered at "ground level".
    
    Returns:
        Dictionary with paths to the created visualization images.
    """
    # Prepare paths
    utci_path = Path(utci_csv_path)
    if output_dir is None:
        output_dir = utci_path.parent
    else:
        output_dir = Path(output_dir)
    
    # Load UTCI values
    df = pd.read_csv(utci_path)
    utci_values = df['utci_celsius'].values
    
    # Try to find sensor positions file if not provided
    if sensor_positions_file is None:
        # Look for sensor positions in same directory with common naming patterns
        possible_files = [
            utci_path.parent / "sensor_positions.csv",
            utci_path.parent / "sensor_positions.json",
            utci_path.parent / "sensors.pts",
            utci_path.parent / (utci_path.stem.replace("utci_results", "sensor_positions") + ".csv")
        ]
        
        for file in possible_files:
            if file.exists():
                sensor_positions_file = file
                break
    
    # If we still don't have sensor positions, generate synthetic ones for demonstration
    if sensor_positions_file is None or not Path(sensor_positions_file).exists():
        print(f"Warning: No sensor positions file found. Creating synthetic positions for visualization purposes.")
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
    
    # Ensure we have the right number of positions
    if len(sensor_positions) != len(utci_values):
        raise ValueError(
            f"Number of sensor positions ({len(sensor_positions)}) "
            f"doesn't match number of UTCI values ({len(utci_values)})"
        )
    
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
    combined_output_path = output_dir / "utci_combined_views.png"
    create_combined = 'combined' in views
    
    # Create the combined view only if requested
    if create_combined:
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
        
        # Create each subplot
        for i, (view, (xlabel, ylabel)) in enumerate(zip(view_options, labels)):
            ax = axs[i]
            
            # Determine which coordinates to use based on view
            if view == 'top':
                x_idx, y_idx = 0, 1  # x, y coordinates
            elif view == 'front':
                x_idx, y_idx = 0, 2  # x, z coordinates
            else:  # side
                x_idx, y_idx = 1, 2  # y, z coordinates
            
            # Extract coordinates
            x = sensor_positions[:, x_idx]
            y = sensor_positions[:, y_idx]
            
            # Create scatter plot
            scatter = ax.scatter(x, y, c=utci_values, s=10, cmap=utci_cmap, 
                               vmin=vmin, vmax=vmax, alpha=0.7)
            
            # Add labels
            ax.set_xlabel(f"{xlabel} coordinate (m)")
            ax.set_ylabel(f"{ylabel} coordinate (m)")
            ax.set_title(f"{view.capitalize()} View")
            
            # Equal aspect ratio for proper visualization
            ax.set_aspect('equal')
        
        # Add a colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label('UTCI (°C)')
        
        # Add main title
        plt.suptitle("UTCI Thermal Comfort Analysis", fontsize=16)
        
        # Add stats text box
        stats_text = (
            f"Min: {utci_values.min():.1f}°C\n"
            f"Max: {utci_values.max():.1f}°C\n"
            f"Mean: {utci_values.mean():.1f}°C"
        )
        fig.text(0.01, 0.02, stats_text, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
        plt.savefig(combined_output_path, dpi=300)
        plt.close()
        
        output_files['combined'] = str(combined_output_path)
    
    return output_files


def add_visualization_to_utci_calculator():
    """
    Add this visualization module to the UTCI calculator pipeline.
    
    Returns:
        Code snippet that can be added to the main calculator script.
    """
    # This is a code snippet that can be added to the main UTCI calculator
    return '''
# Add to imports at the top of the file
from utci_visualization import create_utci_visualization_set

# Add this at the end of calculate_utci_from_honeybee_model() function
# after saving the CSV file:

# Create visualizations if matplotlib is available
try:
    # Save sensor positions for visualization
    sensor_positions_file = os.path.join(output_dir, "sensor_positions.csv")
    with open(sensor_positions_file, "w") as f:
        f.write("x,y,z\\n")
        for sensor in sensor_grid.sensors:
            if hasattr(sensor, 'pos') and hasattr(sensor.pos, 'x'):
                pos = sensor.pos
                f.write(f"{pos.x},{pos.y},{pos.z}\\n")
            elif hasattr(sensor, 'pos') and isinstance(sensor.pos, (list, tuple)):
                f.write(f"{sensor.pos[0]},{sensor.pos[1]},{sensor.pos[2]}\\n")
            else:
                f.write(f"{sensor['pos'][0]},{sensor['pos'][1]},{sensor['pos'][2]}\\n")
    
    # Create visualizations
    visualizations = create_utci_visualization_set(
        utci_file, 
        sensor_positions_file,
        output_dir
    )
    logging.info(f"Created UTCI visualizations: {', '.join(visualizations.keys())}")
except ImportError:
    logging.warning("Could not create visualizations. Make sure matplotlib, scipy and pandas are installed.")
except Exception as e:
    logging.warning(f"Error creating visualizations: {e}")
'''


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Create UTCI visualizations")
    parser.add_argument("utci_file", help="Path to UTCI results CSV file")
    parser.add_argument("--positions", help="Path to sensor positions file (CSV, JSON or PTS)")
    parser.add_argument("--output", help="Output directory for visualizations")
    args = parser.parse_args()
    
    try:
        output_files = create_utci_visualization_set(
            args.utci_file,
            args.positions,
            args.output
        )
        print(f"Created UTCI visualizations:")
        for view, path in output_files.items():
            print(f"  {view.capitalize()} view: {path}")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()