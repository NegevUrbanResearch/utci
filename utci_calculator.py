import os
import time
import tempfile
import subprocess
from pathlib import Path
import json
import struct
import numpy as np
import warnings
from datetime import datetime

# Honeybee imports (only the ones that are confirmed to work)
from honeybee.model import Model
from honeybee.shade import Shade
from honeybee.typing import clean_string
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_radiance.lightsource.sky import CertainIrradiance
from honeybee_radiance.config import folders as rad_folders

# Ladybug imports
from ladybug.epw import EPW
from ladybug.location import Location
from ladybug.sunpath import Sunpath
from ladybug_comfort.utci import universal_thermal_climate_index
from ladybug_geometry.geometry3d import Point3D, Face3D, Vector3D


def _load_gltf_or_glb(filepath):
    """Load either a GLB or a GLTF file, handling external buffers.
    
    Args:
        filepath: Path to the GLTF/GLB file.
        
    Returns:
        tuple: (json_data, bin_data) containing the JSON data and binary data.
    """
    filepath = Path(filepath)

    if filepath.suffix.lower() == '.glb':
        # GLB loading
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            if magic != b'glTF':
                raise ValueError("Not a valid GLB file")
            version, length = struct.unpack('<II', f.read(8))
            chunk_length, chunk_type = struct.unpack('<II', f.read(8))
            if chunk_type != 0x4E4F534A:  # JSON
                raise ValueError("Expected JSON chunk")
            json_data = json.loads(f.read(chunk_length))
            chunk_length, chunk_type = struct.unpack('<II', f.read(8))
            if chunk_type != 0x004E4942:  # BIN
                raise ValueError("Expected BIN chunk")
            bin_data = f.read(chunk_length)
        return json_data, bin_data

    elif filepath.suffix.lower() == '.gltf':
        # GLTF loading
        with open(filepath, 'r') as f:
            json_data = json.load(f)

        if 'buffers' not in json_data or len(json_data['buffers']) == 0:
            raise ValueError("No buffers found in GLTF file.")

        buffer_uri = json_data['buffers'][0]['uri']

        if buffer_uri.startswith('data:'):
            # Handle data URI
            import base64
            data_header, data_encoded = buffer_uri.split(',', 1)
            bin_data = base64.b64decode(data_encoded)
        else:
            # Load from external file
            bin_filepath = filepath.parent / buffer_uri
            if not bin_filepath.exists():
                raise FileNotFoundError(f"Binary file not found: {bin_filepath}")
            with open(bin_filepath, 'rb') as f:
                bin_data = f.read()
        return json_data, bin_data

    else:
        raise ValueError("Unsupported file extension. Must be .glb or .gltf")


def gltf_to_honeybee_model(gltf_path, min_area=1e-6, clean_geometry=True):
    """Convert a GLTF/GLB file to a Honeybee Model.
    
    Args:
        gltf_path: Path to the GLTF/GLB file.
        min_area: Minimum face area to consider valid (default: 1e-6)
        clean_geometry: Whether to clean/validate geometry before adding (default: True)
        
    Returns:
        honeybee.model.Model: A Honeybee Model containing geometry from the GLTF/GLB.
    """
    print(f"Loading GLTF/GLB file: {gltf_path}")
    
    # Load the GLTF/GLB
    json_data, bin_data = _load_gltf_or_glb(gltf_path)
    
    # Create Honeybee Model
    model_name = Path(gltf_path).stem
    model_name = clean_string(model_name)
    hb_model = Model(model_name)
    
    # Extract vertices
    vertices = []
    triangle_indices = []
    
    for mesh in json_data.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            if 'POSITION' not in primitive['attributes']:
                continue

            # Get position accessor and buffer view
            pos_accessor = json_data['accessors'][primitive['attributes']['POSITION']]
            pos_buffer_view = json_data['bufferViews'][pos_accessor['bufferView']]
            
            # Calculate offsets and lengths
            pos_offset = pos_buffer_view.get('byteOffset', 0)
            pos_length = pos_buffer_view['byteLength']
            
            # Extract position data
            pos_data = bin_data[pos_offset:pos_offset + pos_length]
            positions = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 3)
            
            # Store the offset for vertex indexing
            vertex_offset = len(vertices)
            vertices.extend(positions)
            
            if 'indices' in primitive:
                # Handle indexed geometry
                idx_accessor = json_data['accessors'][primitive['indices']]
                idx_buffer_view = json_data['bufferViews'][idx_accessor['bufferView']]
                idx_offset = idx_buffer_view.get('byteOffset', 0)
                idx_length = idx_buffer_view['byteLength']
                
                # Map component type to numpy dtype
                dtype_map = {
                    5121: np.uint8,
                    5123: np.uint16,
                    5125: np.uint32
                }
                idx_dtype = dtype_map[idx_accessor['componentType']]
                
                # Extract indices
                idx_data = bin_data[idx_offset:idx_offset + idx_length]
                indices = np.frombuffer(idx_data, dtype=idx_dtype)
                
                # Group indices into triangles
                for i in range(0, len(indices), 3):
                    if i+2 < len(indices):
                        # Add vertex_offset to account for multiple primitives
                        triangle_indices.append([
                            indices[i] + vertex_offset,
                            indices[i+1] + vertex_offset,
                            indices[i+2] + vertex_offset
                        ])
            else:
                # Non-indexed geometry - use sequential indices
                for i in range(0, len(positions), 3):
                    if i+2 < len(positions):
                        triangle_indices.append([
                            i + vertex_offset,
                            i+1 + vertex_offset,
                            i+2 + vertex_offset
                        ])
    
    vertices = np.array(vertices)
    
    # Process triangles into Honeybee Shades
    print(f"Creating Honeybee Model with {len(triangle_indices)} triangles...")
    
    valid_faces = 0
    invalid_faces = 0
    
    for i, triangle in enumerate(triangle_indices):
        # Create the 3 points of the triangle
        try:
            pt1 = Point3D(vertices[triangle[0]][0], vertices[triangle[0]][1], vertices[triangle[0]][2])
            pt2 = Point3D(vertices[triangle[1]][0], vertices[triangle[1]][1], vertices[triangle[1]][2])
            pt3 = Point3D(vertices[triangle[2]][0], vertices[triangle[2]][1], vertices[triangle[2]][2])
            
            # Check for degenerate triangles (where points are too close together)
            if clean_geometry:
                # Calculate distances between points
                dist1 = pt1.distance_to_point(pt2)
                dist2 = pt2.distance_to_point(pt3)
                dist3 = pt3.distance_to_point(pt1)
                
                # Skip if any points are too close together
                if dist1 < 1e-6 or dist2 < 1e-6 or dist3 < 1e-6:
                    invalid_faces += 1
                    continue
                
                # Skip colinear points (where the triangle has effectively no area)
                # Calculate vectors between points
                vec1 = Vector3D(pt2.x - pt1.x, pt2.y - pt1.y, pt2.z - pt1.z)
                vec2 = Vector3D(pt3.x - pt1.x, pt3.y - pt1.y, pt3.z - pt1.z)
                
                # Calculate cross product magnitude to check for colinearity
                cross_prod = vec1.cross(vec2)
                cross_magnitude = cross_prod.magnitude
                
                if cross_magnitude < 1e-6:
                    invalid_faces += 1
                    continue
            
            # Create a Face3D from the 3 points
            face_geo = Face3D([pt1, pt2, pt3])
            
            # Skip faces with small areas
            if face_geo.area < min_area:
                invalid_faces += 1
                continue
            
            # Create a Honeybee Shade
            shade = Shade(f"shade_{i}", face_geo)
            
            # Add shade to the model
            hb_model.add_shade(shade)
            valid_faces += 1
            
        except Exception as e:
            print(f"Error creating face from triangle {i}: {e}")
            invalid_faces += 1
    
    print(f"Created Honeybee Model with {len(hb_model.shades)} shades")
    print(f"Valid faces: {valid_faces}, Invalid/skipped faces: {invalid_faces}")
    
    return hb_model


def _run_radiance_command(command, cwd):
    """Run a Radiance command and capture output.
    
    Args:
        command: Radiance command to run.
        cwd: Working directory.
        
    Returns:
        subprocess.CompletedProcess: Result of the command.
    """
    print(f"Running command: {command}")
    
    try:
        process = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.stdout:
            print(f"Command output: {process.stdout[:100]}...")
        
        if process.stderr:
            print(f"Command error: {process.stderr}")
            
        return process
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        raise


def _generate_sky_file(output_dir, location, month, day, hour, direct_normal, diffuse_horizontal):
    """Generate a Radiance sky file.
    
    Args:
        output_dir: Directory to write the sky file.
        location: Ladybug Location object.
        month, day, hour: Date and time for the sky.
        direct_normal: Direct normal irradiance in W/m².
        diffuse_horizontal: Diffuse horizontal irradiance in W/m².
        
    Returns:
        str: Path to the generated sky file.
    """
    sky_file = os.path.join(output_dir, "sky.rad")
    
    # Create gensky command
    gensky_cmd = (
        f"gensky -ang {location.latitude} {location.longitude} "
        f"{month} {day} {hour + 0.5} "  # Add 0.5 to center the hour
        f"+s -B {direct_normal} -g {diffuse_horizontal}"
    )
    
    # Write sky file
    with open(sky_file, "w") as f:
        # Add the gensky command as a comment
        f.write(f"# {gensky_cmd}\n\n")
        
        # Add the sky definition
        f.write("void light solar\n0\n0\n3 1e6 1e6 1e6\n\n")
        f.write("solar source sun\n0\n0\n4 0 0 1 0.533\n\n")
        f.write("void light sky_mat\n0\n0\n3 1e1 1e1 1e1\n\n")
        f.write("sky_mat source sky\n0\n0\n4 0 0 1 180\n\n")
        f.write("void plastic ground_mat\n0\n0\n5 .2 .2 .2 0 0\n\n")
        f.write("ground_mat source ground\n0\n0\n4 0 0 -1 180\n")
    
    return sky_file


def _create_radiance_model(hb_model, output_dir):
    """Create Radiance model files from Honeybee model.
    
    Args:
        hb_model: Honeybee Model.
        output_dir: Directory to write files.
        
    Returns:
        str: Path to the Radiance model file.
    """
    rad_file = os.path.join(output_dir, "model.rad")
    
    with open(rad_file, "w") as f:
        # Add material definition
        f.write("void plastic red\n0\n0\n5 1 0 0 0.2 0\n\n")
        
        # Add geometry for each shade in the model
        for i, shade in enumerate(hb_model.shades):
            try:
                vertices = shade.geometry.vertices
                
                # Skip shades with too few vertices
                if len(vertices) < 3:
                    continue
                    
                f.write(f"red polygon shade_{i}\n0\n0\n{3 * len(vertices)}\n")
                
                for vertex in vertices:
                    f.write(f"{vertex.x} {vertex.y} {vertex.z} ")
                
                f.write("\n\n")
            except Exception as e:
                print(f"Error writing shade {i} to Radiance file: {e}")
    
    return rad_file
# Fix for the _create_sensor_file function
def _create_sensor_file(sensor_grid, output_dir):
    """Create Radiance sensor points file.
    
    Args:
        sensor_grid: Honeybee SensorGrid.
        output_dir: Directory to write the file.
        
    Returns:
        str: Path to the sensor file.
    """
    sensor_file = os.path.join(output_dir, "sensors.pts")
    
    with open(sensor_file, "w") as f:
        for sensor in sensor_grid.sensors:
            # Check if pos is a tuple or a Point3D object
            if hasattr(sensor.pos, 'x'):
                # It's a Point3D object
                pos = sensor.pos
                dir = sensor.dir
                f.write(f"{pos.x} {pos.y} {pos.z} {dir.x} {dir.y} {dir.z}\n")
            else:
                # It's probably a tuple
                pos = sensor.pos
                dir = sensor.dir
                f.write(f"{pos[0]} {pos[1]} {pos[2]} {dir[0]} {dir[1]} {dir[2]}\n")
    
    return sensor_file

# Alternative implementation if the above doesn't work
def _create_sensor_file_alt(sensor_grid, output_dir):
    """Create Radiance sensor points file (alternative implementation).
    
    Args:
        sensor_grid: Honeybee SensorGrid.
        output_dir: Directory to write the file.
        
    Returns:
        str: Path to the sensor file.
    """
    sensor_file = os.path.join(output_dir, "sensors.pts")
    
    with open(sensor_file, "w") as f:
        for sensor in sensor_grid.sensors:
            # Get position and direction values safely
            try:
                # Try accessing as object attributes first
                if hasattr(sensor, 'pos') and hasattr(sensor.pos, 'x'):
                    pos_x, pos_y, pos_z = sensor.pos.x, sensor.pos.y, sensor.pos.z
                    dir_x, dir_y, dir_z = sensor.dir.x, sensor.dir.y, sensor.dir.z
                elif hasattr(sensor, 'pos') and isinstance(sensor.pos, (list, tuple)):
                    # Access as list/tuple
                    pos_x, pos_y, pos_z = sensor.pos[0], sensor.pos[1], sensor.pos[2]
                    dir_x, dir_y, dir_z = sensor.dir[0], sensor.dir[1], sensor.dir[2]
                else:
                    # Try dictionary format
                    pos_x, pos_y, pos_z = sensor['pos'][0], sensor['pos'][1], sensor['pos'][2]
                    dir_x, dir_y, dir_z = sensor['dir'][0], sensor['dir'][1], sensor['dir'][2]
                
                f.write(f"{pos_x} {pos_y} {pos_z} {dir_x} {dir_y} {dir_z}\n")
            except Exception as e:
                print(f"Error writing sensor to file: {e}")
                print(f"Sensor type: {type(sensor)}")
                print(f"Sensor data: {sensor}")
                continue
    
    return sensor_file

# Modified create_sensor_grid function to reduce logging
def create_sensor_grid(hb_model, grid_size=0.5, offset=0.1, use_centroids=False):
    """Create a sensor grid on all shades in the Honeybee Model.
    
    Args:
        hb_model: A Honeybee Model.
        grid_size: Size of the sensor grid in model units.
        offset: Offset distance from the geometry surface.
        use_centroids: If True, use face centroids instead of meshing when meshing fails.
        
    Returns:
        SensorGrid: A Honeybee Radiance SensorGrid object.
    """
    # Create a grid of sensor points on each shade
    all_positions = []
    all_directions = []
    
    # Track how many shades we successfully process
    processed_shades = 0
    skipped_shades = 0
    used_centroids = 0
    
    # Process each shade in the model
    for i, shade in enumerate(hb_model.shades):
        try:
            # Progress logging (less verbose)
            if i % 1000 == 0:
                print(f"Processing shade {i}/{len(hb_model.shades)}...")
                
            # Check if the shade is valid for meshing
            if len(shade.geometry.vertices) < 3:
                skipped_shades += 1
                continue
                
            # Get the normal vector of the shade
            normal = shade.normal
            
            # Calculate the area of the shade
            area = shade.geometry.area
            if area < 1e-6:  # Skip extremely small faces
                skipped_shades += 1
                continue
            
            # Try creating a sensor grid on the shade
            mesh_success = False
            
            # First attempt: Try meshing with the standard approach
            try:
                shade_mesh = shade.geometry.mesh_grid(grid_size, offset=offset, generate_centroids=True)
                
                # Check if the mesh has face centroids
                if hasattr(shade_mesh, 'face_centroids') and len(shade_mesh.face_centroids) > 0:
                    # Extract positions and directions
                    positions = [Point3D(*p) for p in shade_mesh.face_centroids]
                    
                    # Use the normal vector for all sensor directions
                    directions = [normal] * len(positions)
                    
                    all_positions.extend(positions)
                    all_directions.extend(directions)
                    processed_shades += 1
                    mesh_success = True
            except AssertionError as e:
                # If meshing fails, we'll try the fallback method if enabled
                if not use_centroids:
                    skipped_shades += 1
            except Exception as e:
                # Just log the type of error, not the full message for every shade
                if i % 1000 == 0:
                    print(f"Error meshing shade {i}: {type(e).__name__}")
                
            # Second attempt: If meshing failed and use_centroids is True, 
            # just use the face centroid as a sensor point
            if not mesh_success and use_centroids:
                try:
                    # Use the face centroid with an offset in the normal direction
                    centroid = shade.geometry.centroid
                    if offset != 0:
                        # Move the centroid in the direction of the normal by the offset amount
                        moved_centroid = Point3D(
                            centroid.x + normal.x * offset,
                            centroid.y + normal.y * offset,
                            centroid.z + normal.z * offset
                        )
                    else:
                        moved_centroid = centroid
                    
                    all_positions.append(moved_centroid)
                    all_directions.append(normal)
                    used_centroids += 1
                    
                except Exception as e:
                    skipped_shades += 1
                    
        except Exception as e:
            # Handle any other errors
            skipped_shades += 1
            if i % 1000 == 0:
                print(f"Error processing shade {i}: {type(e).__name__}")
    
    # Check if we have any points to create a grid
    if not all_positions:
        print(f"No valid sensor points generated. Processed: {processed_shades}, Skipped: {skipped_shades}")
        
        # If we have no valid points, create a minimal grid with a single point
        # This allows the workflow to continue, even with minimal data
        print("Creating fallback sensor grid with a single point")
        center = Point3D(0, 0, 0)
        normal = Vector3D(0, 0, 1)
        all_positions = [center]
        all_directions = [normal]
    
    # Print sample of positions (not all of them)
    if len(all_positions) > 0:
        sample_size = min(5, len(all_positions))
        print(f"Sample of positions (showing {sample_size} of {len(all_positions)}):")
        for i in range(sample_size):
            print(f"  Position {i}: {all_positions[i]}")
    
    # Create a sensor grid with all positions and directions
    try:
        # Try the standard 3-argument constructor (name, positions, directions)
        sensor_grid = SensorGrid('utci_grid', all_positions, all_directions)
    except TypeError as e:
        print(f"Error creating SensorGrid with standard constructor: {e}")
        # Try alternative constructor forms based on the error
        try:
            # Try without providing identifier (may be a default parameter in some versions)
            sensor_grid = SensorGrid(all_positions, all_directions)
        except Exception as e2:
            print(f"Error creating SensorGrid with alternative constructor: {e2}")
            # Last resort - create a very simple grid with explicit arguments
            print("Attempting to create minimal sensor grid...")
            try:
                # Convert Point3D and Vector3D objects to arrays
                pos_arrays = [p.to_array() if hasattr(p, 'to_array') else p for p in all_positions]
                dir_arrays = [d.to_array() if hasattr(d, 'to_array') else d for d in all_directions]
                
                sensor_grid = SensorGrid.from_dict({
                    'type': 'SensorGrid', 
                    'identifier': 'utci_grid',
                    'display_name': 'UTCI Grid',
                    'sensors': [{'pos': p, 'dir': d} for p, d in zip(pos_arrays, dir_arrays)]
                })
            except Exception as e3:
                print(f"Error creating minimal sensor grid: {e3}")
                # If everything fails, create an absolute minimal grid
                sensor_grid = SensorGrid.from_dict({
                    'type': 'SensorGrid', 
                    'identifier': 'utci_grid',
                    'sensors': [{'pos': [0, 0, 0], 'dir': [0, 0, 1]}]
                })
    
    # Summary
    print(f"Created sensor grid with {len(all_positions)} sensor points")
    print(f"Processed {processed_shades} shades with meshing")
    if use_centroids:
        print(f"Used centroids for {used_centroids} shades")
    print(f"Skipped {skipped_shades} shades")
    
    return sensor_grid

# Add a helper function to inspect SensorGrid structure
def inspect_sensor_grid(sensor_grid):
    """Inspect the structure of a sensor grid to help with debugging.
    
    Args:
        sensor_grid: A SensorGrid object to inspect.
    """
    print(f"SensorGrid type: {type(sensor_grid)}")
    print(f"SensorGrid attributes: {dir(sensor_grid)}")
    
    if hasattr(sensor_grid, 'sensors') and len(sensor_grid.sensors) > 0:
        print(f"Number of sensors: {len(sensor_grid.sensors)}")
        
        # Inspect first sensor
        first_sensor = sensor_grid.sensors[0]
        print(f"First sensor type: {type(first_sensor)}")
        print(f"First sensor attributes: {dir(first_sensor)}")
        
        # Check position and direction format
        if hasattr(first_sensor, 'pos'):
            print(f"Position type: {type(first_sensor.pos)}")
            print(f"Position value: {first_sensor.pos}")
            
            if hasattr(first_sensor, 'dir'):
                print(f"Direction type: {type(first_sensor.dir)}")
                print(f"Direction value: {first_sensor.dir}")

# Modify the calculate_utci_from_honeybee_model function to use the fixed functions
def calculate_utci_from_honeybee_model(
    hb_model, 
    epw_path, 
    output_dir,
    hour_of_year, 
    grid_size=0.5,
    offset=0.1,
    solar_absorptance=0.7,
    ground_albedo=0.2,
    use_centroids=True
):
    """Calculate UTCI using direct Radiance commands.
    
    Args:
        hb_model: A Honeybee Model.
        epw_path: Path to the EPW file.
        output_dir: Path to the output directory.
        hour_of_year: Hour of the year for which UTCI will be calculated.
        grid_size: Size of the sensor grid in model units.
        offset: Offset distance for the sensor grid.
        solar_absorptance: Solar absorptance coefficient.
        ground_albedo: Ground reflectance.
        use_centroids: If True, use face centroids when meshing fails.
        
    Returns:
        np.ndarray: Array of UTCI values for each sensor point.
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load EPW file
    print(f"Loading EPW file: {epw_path}")
    epw = EPW(epw_path)
    location = Location(
        epw.location.city, 
        epw.location.state, 
        epw.location.country,
        epw.location.latitude, 
        epw.location.longitude, 
        epw.location.time_zone,
        epw.location.elevation
    )
    
    # Check if the hour is daytime
    sunpath = Sunpath.from_location(location)
    sun = sunpath.calculate_sun_from_hoy(hour_of_year - 1)
    if not sun.is_during_day:
        print(f"Hour {hour_of_year} is not during the day. Returning empty array.")
        return np.array([])
    
    # Get weather data for the hour
    air_temp = epw.dry_bulb_temperature[hour_of_year - 1]
    rel_humidity = epw.relative_humidity[hour_of_year - 1]
    wind_speed = epw.wind_speed[hour_of_year - 1]
    direct_normal = epw.direct_normal_radiation[hour_of_year - 1]
    diffuse_horizontal = epw.diffuse_horizontal_radiation[hour_of_year - 1]
    
    # Calculate the month, day, and hour from the hour of year
    hour_index = hour_of_year - 1
    day_of_year = hour_index // 24
    hour = hour_index % 24
    
    # Approximate calculation for month and day using a non-leap year
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Calculate month and day
    month = 1  # Start with January
    remaining_days = day_of_year
    
    for i, days in enumerate(days_in_month):
        if remaining_days < days:
            # Found the month
            month = i + 1  # Months are 1-indexed
            day = remaining_days + 1  # Days are 1-indexed
            break
        remaining_days -= days
    
    # Create sensor grid with option to use centroids as fallback
    sensor_grid = create_sensor_grid(hb_model, grid_size, offset, use_centroids=use_centroids)
    
    # Debug: Inspect the sensor grid structure
    inspect_sensor_grid(sensor_grid)
    
    # Setup Radiance working directory
    rad_dir = output_dir / "radiance"
    rad_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate Radiance files
    sky_file = _generate_sky_file(rad_dir, location, month, day, hour, direct_normal, diffuse_horizontal)
    model_file = _create_radiance_model(hb_model, rad_dir)
    
    # Use the fixed version of create_sensor_file
    try:
        sensor_file = _create_sensor_file(sensor_grid, rad_dir)
    except Exception as e:
        print(f"Error with standard sensor file creation: {e}")
        print("Trying alternative sensor file creation method...")
        sensor_file = _create_sensor_file_alt(sensor_grid, rad_dir)
    
    octree_file = os.path.join(rad_dir, "scene.oct")
    
    # Create octree
    oconv_cmd = f"oconv {model_file} {sky_file} > {octree_file}"
    _run_radiance_command(oconv_cmd, rad_dir)
    
    # Calculate irradiance
    results_file = os.path.join(rad_dir, "results.dat")
    rtrace_cmd = (
        f"rtrace -I -h -ab 2 -ad 5000 -lw 2e-05 "
        f"{octree_file} < {sensor_file} > {results_file}"
    )
    _run_radiance_command(rtrace_cmd, rad_dir)
    
    # Parse results
    irradiance_values = []
    with open(results_file, "r") as f:
        for line in f:
            values = line.strip().split()
            if len(values) >= 3:
                # Sum RGB values for total irradiance
                irr = sum(float(v) for v in values[:3])
                irradiance_values.append(irr)
    
    irradiance_values = np.array(irradiance_values)
    
    # Validate results
    if len(irradiance_values) == 0:
        print("No irradiance values calculated.")
        return np.array([])
        
    if len(irradiance_values) != len(sensor_grid.sensors):
        print(f"Warning: Mismatch between sensor count ({len(sensor_grid.sensors)}) "
              f"and result count ({len(irradiance_values)}).")
    
    # Calculate UTCI
    # MRT calculation
    longwave_mrt = air_temp
    shortwave_mrt_delta = (solar_absorptance * irradiance_values / 5.67e-8)**0.25 - 273.15
    
    # Handle edge cases with np.nan_to_num
    shortwave_mrt_delta = np.nan_to_num(shortwave_mrt_delta, nan=0.0, posinf=30.0, neginf=0.0)
    
    # Calculate mean radiant temperature
    mean_radiant_temperature = (shortwave_mrt_delta**4 + longwave_mrt**4)**0.25
    mean_radiant_temperature = np.nan_to_num(mean_radiant_temperature, nan=air_temp, posinf=air_temp+30, neginf=air_temp)
    
    # Calculate UTCI
    utci_values = universal_thermal_climate_index(
        air_temp, mean_radiant_temperature, wind_speed, rel_humidity
    )
    
    # Validate UTCI values
    min_utci, max_utci = np.min(utci_values), np.max(utci_values)
    if min_utci < -50 or max_utci > 50:
        warnings.warn(f"Extreme UTCI values detected: Min={min_utci:.2f}°C, Max={max_utci:.2f}°C")
        # Clip to reasonable range
        utci_values = np.clip(utci_values, -50, 50)
    
    print(f"UTCI calculation completed with {len(utci_values)} values")
    print(f"  UTCI range: {np.min(utci_values):.2f}°C to {np.max(utci_values):.2f}°C")
    print(f"  UTCI mean: {np.mean(utci_values):.2f}°C")
    
    return utci_values


def calculate_utci_from_gltf_epw(
    gltf_path, 
    epw_path, 
    output_dir, 
    hour_of_year, 
    ground_albedo=0.2, 
    solar_absorptance=0.7, 
    grid_size=0.5,
    offset=0.1,
    clean_geometry=True,
    use_centroids=True
):
    """Calculates UTCI from a GLB/GLTF model and EPW file, using direct Radiance calls.
    
    Args:
        gltf_path: Path to the GLTF/GLB file.
        epw_path: Path to the EPW file.
        output_dir: Path to the output directory.
        hour_of_year: Hour of the year for which UTCI will be calculated.
        ground_albedo: Ground reflectance.
        solar_absorptance: Solar absorptance coefficient.
        grid_size: Size of the sensor grid in model units.
        offset: Offset distance for sensor points.
        clean_geometry: Whether to clean/validate geometry while loading.
        use_centroids: If True, use face centroids when meshing fails.
        
    Returns:
        np.ndarray: Array of UTCI values for each sensor point.
    """
    # Convert GLTF to Honeybee Model with geometry cleaning
    hb_model = gltf_to_honeybee_model(gltf_path, min_area=1e-6, clean_geometry=clean_geometry)
    
    # Run the calculation with improved sensor grid creation
    utci_values = calculate_utci_from_honeybee_model(
        hb_model,
        epw_path,
        output_dir,
        hour_of_year,
        grid_size=grid_size,
        offset=offset,
        solar_absorptance=solar_absorptance,
        ground_albedo=ground_albedo,
        use_centroids=use_centroids
    )
    
    return utci_values


def example_usage():
    """Example usage with GLB file."""
    current_dir = Path.cwd()
    glb_file = Path("data/rec_model_no_curve.glb")
    epw_file = current_dir / "data/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw"
    output_directory = current_dir / "output"
    hour = 12  # Noon for daytime calculation
    
    # Ensure the file exists before proceeding
    if not glb_file.exists():
        print(f"Error: GLB file not found at {glb_file}")
        print("Please check the file path and try again")
        return
    
    print(f"Processing GLB file: {glb_file}")
    start_time = time.time()
    
    # Calculate UTCI values with improved geometry handling
    utci_values = calculate_utci_from_gltf_epw(
        str(glb_file), 
        str(epw_file), 
        str(output_directory), 
        hour,
        grid_size=1.0,           # Larger grid size for faster processing
        offset=0.1,              # Offset distance
        clean_geometry=True,     # Clean geometry during import
        use_centroids=True       # Use centroids when meshing fails
    )
    
    elapsed_time = time.time() - start_time
    
    if len(utci_values) > 0:
        print(f"Calculated UTCI values for {len(utci_values):,} points in {elapsed_time:.2f} seconds.")
        print(f"UTCI statistics:")
        print(f"  Min: {np.min(utci_values):.2f}°C")
        print(f"  Max: {np.max(utci_values):.2f}°C")
        print(f"  Mean: {np.mean(utci_values):.2f}°C")
        print(f"  Median: {np.median(utci_values):.2f}°C")
        
        # Save the results
        output_file = output_directory / "utci_values.txt"
        output_directory.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_file, utci_values)
        print(f"UTCI values saved to: {output_file}")
    else:
        print(f'No UTCI values to show for hour {hour}')


if __name__ == "__main__":
    example_usage()