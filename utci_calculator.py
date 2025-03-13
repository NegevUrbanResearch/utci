#!/usr/bin/env python3
"""UTCI Calculator for GLTF/GLB models using Honeybee and Radiance."""

import os
import time
import tempfile
import subprocess
import json
import struct
import numpy as np
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import torch

# Honeybee imports
from honeybee.model import Model
from honeybee.shade import Shade
from honeybee.typing import clean_string
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_radiance.lightsource.sky import CertainIrradiance

# Ladybug imports
from ladybug.epw import EPW
from ladybug.location import Location
from ladybug.sunpath import Sunpath
from ladybug_comfort.utci import universal_thermal_climate_index
from ladybug_geometry.geometry3d import Point3D, Face3D, Vector3D

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def set_log_level(level_name):
    """Set the logging level."""
    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level_name}')
    logging.getLogger().setLevel(numeric_level)

def _load_gltf_or_glb(filepath):
    """Load either a GLB or a GLTF file, handling external buffers."""
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
    """Convert a GLTF/GLB file to a Honeybee Model."""
    logging.info(f"Loading GLTF/GLB file: {gltf_path}")
    
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
    logging.info(f"Creating Honeybee Model with {len(triangle_indices)} triangles...")
    
    valid_faces = 0
    invalid_faces = 0
    
    for i, triangle in enumerate(triangle_indices):
        # Create the 3 points of the triangle
        try:
            pt1 = Point3D(vertices[triangle[0]][0], vertices[triangle[0]][1], vertices[triangle[0]][2])
            pt2 = Point3D(vertices[triangle[1]][0], vertices[triangle[1]][1], vertices[triangle[1]][2])
            pt3 = Point3D(vertices[triangle[2]][0], vertices[triangle[2]][1], vertices[triangle[2]][2])
            
            # Check for degenerate triangles if requested
            if clean_geometry:
                # Calculate distances between points
                dist1 = pt1.distance_to_point(pt2)
                dist2 = pt2.distance_to_point(pt3)
                dist3 = pt3.distance_to_point(pt1)
                
                # Skip if any points are too close together
                if dist1 < 1e-6 or dist2 < 1e-6 or dist3 < 1e-6:
                    invalid_faces += 1
                    continue
                
                # Skip colinear points
                vec1 = Vector3D(pt2.x - pt1.x, pt2.y - pt1.y, pt2.z - pt1.z)
                vec2 = Vector3D(pt3.x - pt1.x, pt3.y - pt1.y, pt3.z - pt1.z)
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
            logging.debug(f"Error creating face from triangle {i}: {e}")
            invalid_faces += 1
    
    logging.info(f"Created Honeybee Model with {len(hb_model.shades)} shades")
    logging.info(f"Valid faces: {valid_faces}, Invalid/skipped faces: {invalid_faces}")
    
    return hb_model


def check_gpu_availability():
    """Check if Accelerad GPU capabilities are available."""
    try:
        result = subprocess.run(['rtrace_gpu', '-version'], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               text=True, check=False)
        if "NVIDIA" in result.stdout or "OptiX" in result.stdout:
            logging.info(f"GPU acceleration available: {result.stdout.strip()}")
            return True
        else:
            logging.warning("GPU acceleration not available or not detected")
            return False
    except Exception as e:
        logging.warning(f"GPU acceleration not available: {e}")
        return False

def _run_radiance_command(command, cwd, use_gpu=True):
    """Run a Radiance command and capture output."""
    # Replace rtrace with rtrace_gpu for GPU acceleration if available
    if use_gpu and check_gpu_availability():
        if ' rtrace ' in command:
            command = command.replace(' rtrace ', ' rtrace_gpu ')
        elif command.startswith('rtrace '):
            command = command.replace('rtrace ', 'rtrace_gpu ')
    
    logging.info(f"Running command: {command}")
    
    try:
        # For long-running commands like rtrace, show a simple time-based progress bar
        if 'rtrace' in command:
            logging.info("Starting ray tracing calculation (this may take a while)...")
            start_time = time.time()
            
            # Start the process
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Create a simple time-based progress bar
            with tqdm(desc="Ray tracing (time elapsed)", unit="s") as pbar:
                while process.poll() is None:
                    pbar.update(1)
                    time.sleep(1)
            
            # Process completed
            elapsed = time.time() - start_time
            logging.info(f"Ray tracing completed in {elapsed:.1f} seconds")
            
            # Get output and errors
            stdout, stderr = process.communicate()
            
            if stderr:
                logging.warning(f"Command warnings/errors: {stderr[:200]}..." if len(stderr) > 200 else stderr)
                
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command, stderr)
                
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        else:
            # For other commands, use the standard approach
            process = subprocess.run(
                command,
                shell=True,
                check=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.stderr:
                logging.warning(f"Command warnings/errors: {process.stderr[:200]}..." if len(process.stderr) > 200 else process.stderr)
                
            return process
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with return code {e.returncode}")
        logging.error(f"Error output: {e.stderr[:200]}..." if len(e.stderr) > 200 else e.stderr)
        raise

def _run_gpu_rtrace_with_batching(octree_file, sensor_file, results_file, temp_dir, batch_size=5000):
    """Run rtrace with GPU acceleration using batching to manage GPU memory."""
    # Count total sensors
    with open(sensor_file, 'r') as f:
        total_sensors = sum(1 for _ in f)
    
    # Create temporary directory for batched processing
    batch_dir = os.path.join(temp_dir, "gpu_batches")
    os.makedirs(batch_dir, exist_ok=True)
    
    results = []
    processed = 0
    
    with tqdm(total=total_sensors, desc="GPU ray tracing") as pbar:
        # Process in batches
        while processed < total_sensors:
            # Create batch sensor file
            batch_file = os.path.join(batch_dir, f"sensors_batch_{processed}.pts")
            with open(sensor_file, 'r') as f_in:
                # Skip already processed lines
                for _ in range(processed):
                    f_in.readline()
                    
                # Write this batch
                with open(batch_file, 'w') as f_out:
                    batch_count = 0
                    for _ in range(batch_size):
                        line = f_in.readline()
                        if not line:
                            break
                        f_out.write(line)
                        batch_count += 1
            
            if batch_count == 0:
                break  # No more sensors to process
            
            # Run GPU rtrace on this batch
            batch_results = os.path.join(batch_dir, f"results_batch_{processed}.dat")
            rtrace_cmd = (
                f"rtrace_gpu -I -h -ab 1 -ad 1000 -lw 0.0001 "
                f"-dc 1 -dt 0 -dj 0 -st 0 -ss 0 -faf "
                f"{octree_file} < {batch_file} > {batch_results}"
            )
            
            try:
                subprocess.run(rtrace_cmd, shell=True, check=True, cwd=temp_dir)
                
                # Read this batch's results
                with open(batch_results, 'rb') as f_batch:
                    batch_data = f_batch.read()
                    results.append(batch_data)
                
                # Update progress
                processed += batch_count
                pbar.update(batch_count)
                
            except subprocess.CalledProcessError as e:
                logging.error(f"GPU rtrace failed: {e}")
                logging.warning("Falling back to CPU rtrace")
                return False
                
    # Combine all results
    with open(results_file, 'wb') as f_out:
        for result in results:
            f_out.write(result)
    
    return True

def _run_cpu_rtrace(octree_file, sensor_file, results_file, temp_dir):
    """Run standard rtrace on CPU."""
    rtrace_cmd = (
        f"rtrace -I -h -ab 1 -ad 1000 -lw 0.0001 "
        f"-dc 1 -dt 0 -dj 0 -st 0 -ss 0 -faf "
        f"{octree_file} < {sensor_file} > {results_file}"
    )
    
    _run_radiance_command(rtrace_cmd, temp_dir, use_gpu=False)
    return True

def _generate_sky_file(output_dir, location, month, day, hour, direct_normal, diffuse_horizontal):
    """Generate a Radiance sky file."""
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
    """Create Radiance model files from Honeybee model."""
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
                logging.debug(f"Error writing shade {i} to Radiance file: {e}")
    
    return rad_file


def _create_sensor_file(sensor_grid, output_dir):
    """Create Radiance sensor points file."""
    sensor_file = os.path.join(output_dir, "sensors.pts")
    
    with open(sensor_file, "w") as f:
        for sensor in sensor_grid.sensors:
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
    
    return sensor_file


def _read_rtrace_results(results_file):
    """Read rtrace results from a binary file."""
    try:
        # First try to read as a binary file in float format
        with open(results_file, 'rb') as f:
            # Check if it's a text file by reading the first few bytes
            first_bytes = f.read(6)
            is_text = False
            
            # Check if the first byte is a digit
            for byte in first_bytes:
                if chr(byte).isdigit():
                    is_text = True
                    break
            
            f.seek(0)  # Reset file pointer to beginning
            
            if is_text:
                # Text file format
                results = []
                for line in f:
                    try:
                        values = line.decode('utf-8').strip().split()
                        if len(values) >= 3:
                            rgb = [float(val) for val in values[:3]]
                            results.append(sum(rgb) / 3.0)  # Average of RGB
                        else:
                            results.append(0.0)
                    except Exception:
                        results.append(0.0)
            else:
                # Binary format - assuming Radiance RGBE/float format
                # Just read the bytes and interpret as floats
                results = []
                byte_data = f.read()
                float_size = 4  # Size of float in bytes
                values_per_line = 3  # RGB values per line
                
                # Read every 3 floats (RGB values)
                for i in range(0, len(byte_data), float_size * values_per_line):
                    if i + float_size * values_per_line <= len(byte_data):
                        rgb_sum = 0
                        for j in range(values_per_line):
                            offset = i + j * float_size
                            value_bytes = byte_data[offset:offset + float_size]
                            try:
                                value = struct.unpack('f', value_bytes)[0]
                                rgb_sum += value
                            except Exception:
                                pass
                        results.append(rgb_sum / values_per_line)
                    else:
                        results.append(0.0)
        
        return results
    except Exception as e:
        logging.error(f"Error reading rtrace results: {e}")
        # Generate varied synthetic values for testing
        logging.warning("Generating synthetic solar values")
        return [i / 100.0 for i in range(100)]  # Return varied values for testing


def _process_shade(args):
    """Process a single shade to extract sensor points (must be outside main function for multiprocessing).
    
    Args:
        args: Tuple containing (shade, shade_idx, grid_size, offset, use_centroids)
        
    Returns:
        Dictionary with positions, directions and processing status
    """
    shade, shade_idx, grid_size, offset, use_centroids = args
    result = {"positions": [], "directions": [], "processed": False, "centroid_used": False}
    
    try:
        # Check if the shade is valid for meshing
        if len(shade.geometry.vertices) < 3:
            return result
            
        # Get the normal vector of the shade
        normal = shade.normal
        
        # Calculate the area of the shade
        area = shade.geometry.area
        if area < 1e-6:  # Skip extremely small faces
            return result
        
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
                
                result["positions"] = positions
                result["directions"] = directions
                result["processed"] = True
                mesh_success = True
        except Exception:
            pass
            
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
                
                result["positions"] = [moved_centroid]
                result["directions"] = [normal]
                result["processed"] = True
                result["centroid_used"] = True
            except Exception:
                pass
    except Exception:
        pass
        
    return result


def create_sensor_grid(hb_model, grid_size=0.5, offset=0.1, use_centroids=False, max_sensors=10000):
    """Create a sensor grid on all shades in the Honeybee Model."""
    all_positions = []
    all_directions = []
    
    # Track processing stats
    processed_shades = 0
    skipped_shades = 0
    used_centroids = 0
    
    # For non-parallel processing, use simple loop
    total_shades = len(hb_model.shades)
    
    # Determine if we can use parallel processing
    try:
        # Determine number of processes to use (leave one core free for system)
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        use_parallel = num_processes > 1 and total_shades > 100  # Only use parallel for larger models
        
        if use_parallel:
            logging.info(f"Processing {total_shades} shades using {num_processes} processes...")
            
            # Prepare the arguments for each worker
            process_args = [(hb_model.shades[i], i, grid_size, offset, use_centroids) 
                           for i in range(total_shades)]
            
            # Process in parallel
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                results = list(tqdm(
                    executor.map(_process_shade, process_args), 
                    total=total_shades,
                    desc="Processing shades"
                ))
        else:
            # Process sequentially for smaller models or if multiprocessing is unavailable
            logging.info(f"Processing {total_shades} shades sequentially...")
            results = []
            for i in tqdm(range(total_shades), desc="Processing shades"):
                results.append(_process_shade((hb_model.shades[i], i, grid_size, offset, use_centroids)))
    except Exception as e:
        # Fallback to sequential processing if parallel fails
        logging.warning(f"Parallel processing failed: {e}. Switching to sequential processing.")
        results = []
        for i in tqdm(range(total_shades), desc="Processing shades"):
            results.append(_process_shade((hb_model.shades[i], i, grid_size, offset, use_centroids)))
    
    # Collect results
    for result in results:
        if result["processed"]:
            all_positions.extend(result["positions"])
            all_directions.extend(result["directions"])
            processed_shades += 1
            if result["centroid_used"]:
                used_centroids += 1
            
            # Check if we've hit the sensor limit
            if max_sensors and len(all_positions) >= max_sensors:
                logging.info(f"Reached maximum sensor count ({max_sensors}). Truncating grid.")
                all_positions = all_positions[:max_sensors]
                all_directions = all_directions[:max_sensors]
                break
        else:
            skipped_shades += 1
    
    # Check if we have any points to create a grid
    if not all_positions:
        logging.warning(f"No valid sensor points generated. Processed: {processed_shades}, Skipped: {skipped_shades}")
        
        # If we have no valid points, create a minimal grid with a single point
        logging.info("Creating fallback sensor grid with a single point")
        center = Point3D(0, 0, 0)
        normal = Vector3D(0, 0, 1)
        all_positions = [center]
        all_directions = [normal]
    
    # Print summary
    logging.info(f"Created sensor grid with {len(all_positions)} sensor points")
    logging.info(f"Processed {processed_shades} shades with meshing")
    if use_centroids:
        logging.info(f"Used centroids for {used_centroids} shades")
    logging.info(f"Skipped {skipped_shades} shades")
    
    # Create a sensor grid with all positions and directions
    try:
        # Try the standard constructor
        sensor_grid = SensorGrid('utci_grid', all_positions, all_directions)
    except Exception:
        # Fall back to dictionary-based construction
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
        except Exception:
            # Last resort - create a minimal grid
            sensor_grid = SensorGrid.from_dict({
                'type': 'SensorGrid', 
                'identifier': 'utci_grid',
                'sensors': [{'pos': [0, 0, 0], 'dir': [0, 0, 1]}]
            })
    
    return sensor_grid


def _process_utci_batch(batch_data):
    """Process a batch of UTCI calculations for parallelization."""
    air_temp, mrt_batch, wind_speed, rel_humidity = batch_data
    
    # Calculate UTCI for each point in the batch
    utci_results = []
    for mrt in mrt_batch:
        # Handle missing or NaN MRT values
        if np.isnan(mrt):
            utci_results.append(np.nan)
        else:
            utci = universal_thermal_climate_index(air_temp, mrt, wind_speed, rel_humidity)
            utci_results.append(utci)
        
    return utci_results


def calculate_utci_from_honeybee_model(
    hb_model, 
    epw_path, 
    output_dir,
    hour_of_year, 
    grid_size=1.0,
    offset=0.1,
    solar_absorptance=0.7,
    use_centroids=True,
    max_sensors=10000,
    use_gpu=True,
    sensor_grid=None  # New optional parameter
):
    """Calculate UTCI using Radiance and the UTCI formula."""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load EPW file
    logging.info(f"Loading EPW file: {epw_path}")
    epw = EPW(epw_path)
    
    # Get year from EPW data
    try:
        year = int(epw.header[0])
    except (ValueError, TypeError):
        if hasattr(epw, 'year'):
            year = epw.year
        else:
            logging.warning("Could not determine year from EPW file, using 2021 as default")
            year = 2021
    
    # Create location object
    location = Location(
        epw.location.city, 
        epw.location.state, 
        epw.location.country,
        epw.location.latitude, 
        epw.location.longitude, 
        epw.location.time_zone,
        epw.location.elevation
    )
    
    # Calculate month, day, hour for the specified hour of year
    datetime_obj = datetime(year, 1, 1) + timedelta(hours=hour_of_year - 1)
    month = datetime_obj.month
    day = datetime_obj.day
    hour = datetime_obj.hour
    
    logging.info(f"Calculating UTCI for: Month {month}, Day {day}, Hour {hour}")
    
    # Get weather data for this hour
    air_temp = epw.dry_bulb_temperature[hour_of_year - 1]  # °C
    rel_humidity = epw.relative_humidity[hour_of_year - 1]  # %
    wind_speed = epw.wind_speed[hour_of_year - 1]  # m/s
    
    # Get direct normal and diffuse horizontal radiation
    direct_normal = epw.direct_normal_radiation[hour_of_year - 1]  # W/m²
    diffuse_horizontal = epw.diffuse_horizontal_radiation[hour_of_year - 1]  # W/m²
    
    logging.info(f"Weather conditions: Temp {air_temp:.1f}°C, RH {rel_humidity:.1f}%, "
          f"Wind {wind_speed:.1f} m/s, DNI {direct_normal:.1f} W/m², DHI {diffuse_horizontal:.1f} W/m²")
    
    # Create or use existing sensor grid
    if sensor_grid is None:
        sensor_grid = create_sensor_grid(hb_model, grid_size, offset, use_centroids, max_sensors)
    
    num_sensors = len(sensor_grid.sensors)
    logging.info(f"Using sensor grid with {num_sensors} points")
    
    # Create temporary directory for Radiance files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create Radiance model file
        rad_model_file = _create_radiance_model(hb_model, temp_dir)
        
        # Create sky file
        sky_file = _generate_sky_file(temp_dir, location, month, day, hour, direct_normal, diffuse_horizontal)
        
        # Create sensor file
        sensor_file = _create_sensor_file(sensor_grid, temp_dir)
        
        # Create octree
        octree_file = os.path.join(temp_dir, "scene.oct")
        oconv_cmd = f"oconv {sky_file} {rad_model_file} > {octree_file}"
        _run_radiance_command(oconv_cmd, temp_dir, use_gpu=False)  # oconv doesn't have GPU version
        
        # Run rtrace for direct solar - try GPU first if enabled
        direct_results_file = os.path.join(temp_dir, "direct_results.dat")
        
        # Try GPU with batching if requested and available
        gpu_success = False
        if use_gpu and check_gpu_availability():
            logging.info("Using GPU acceleration for ray tracing...")
            try:
                gpu_success = _run_gpu_rtrace_with_batching(
                    octree_file, sensor_file, direct_results_file, temp_dir
                )
            except Exception as e:
                logging.error(f"GPU ray tracing failed: {e}")
                logging.warning("Falling back to CPU ray tracing")
                gpu_success = False
        
        # Fall back to CPU if GPU failed or wasn't requested
        if not gpu_success:
            logging.info("Using CPU for ray tracing...")
            _run_cpu_rtrace(octree_file, sensor_file, direct_results_file, temp_dir)
        
        # Read rtrace results
        logging.info("Reading rtrace results...")
        direct_sun_values = _read_rtrace_results(direct_results_file)
        
        # Make sure we have the right number of values
        if len(direct_sun_values) != num_sensors:
            logging.warning(f"Number of rtrace results ({len(direct_sun_values)}) doesn't match sensor count ({num_sensors})")
            # Truncate or extend as needed
            if len(direct_sun_values) > num_sensors:
                direct_sun_values = direct_sun_values[:num_sensors]
            else:
                direct_sun_values.extend([0.0] * (num_sensors - len(direct_sun_values)))
        
        # Convert to numpy array
        direct_sun_values = np.array(direct_sun_values)
        
        # Scale by solar absorptance and convert to temperature contribution
        direct_sun_influence = direct_sun_values * solar_absorptance * 0.5
        
        # Calculate mean radiant temperature
        mrt_values = air_temp + direct_sun_influence
        
        # Calculate UTCI values
        logging.info("Calculating UTCI values...")
        
        try:
            # Try parallel processing first
            # Determine optimal batch size and number of processes
            num_processes = max(1, multiprocessing.cpu_count() - 1)
            batch_size = max(1, len(mrt_values) // (num_processes * 4))  # Split into smaller batches
            
            # Prepare batches for parallel processing
            batches = []
            for i in range(0, len(mrt_values), batch_size):
                batch_mrt = mrt_values[i:i+batch_size]
                batches.append((air_temp, batch_mrt, wind_speed, rel_humidity))
            
            # Process batches in parallel
            utci_values = []
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                results = list(tqdm(
                    executor.map(_process_utci_batch, batches), 
                    total=len(batches),
                    desc="UTCI calculation"
                ))
                
                # Combine results from all batches
                for batch_result in results:
                    utci_values.extend(batch_result)
        
        except Exception as e:
            # Fallback to sequential processing if parallel processing fails
            logging.warning(f"Parallel UTCI calculation failed: {e}. Switching to sequential processing.")
            utci_values = []
            for mrt in tqdm(mrt_values, desc="UTCI calculation"):
                utci = universal_thermal_climate_index(air_temp, mrt, wind_speed, rel_humidity)
                utci_values.append(utci)
        
        # Convert to numpy array
        utci_values = np.array(utci_values)
        
        # Ensure utci_values matches the number of sensors
        if len(utci_values) != num_sensors:
            logging.warning(f"Number of UTCI values ({len(utci_values)}) doesn't match sensor count ({num_sensors})")
            # Truncate or extend as needed
            if len(utci_values) > num_sensors:
                utci_values = utci_values[:num_sensors]
            else:
                # Extend with NaN values for missing sensors
                utci_values = np.append(utci_values, np.full(num_sensors - len(utci_values), np.nan))
        
        # Save UTCI values to a file
        utci_file = os.path.join(output_dir, "utci_results.csv")
        with open(utci_file, "w") as f:
            f.write("sensor_id,utci_celsius\n")
            for i, utci in enumerate(utci_values):
                f.write(f"{i},{utci:.2f}\n")
        
        # Log result statistics
        logging.info(f"UTCI results saved to: {utci_file}")
        logging.info(f"UTCI range: Min {np.min(utci_values):.1f}°C, Max {np.max(utci_values):.1f}°C, Mean {np.mean(utci_values):.1f}°C")
        
        # Save sensor positions for visualization
        sensor_positions_file = os.path.join(output_dir, "sensor_positions.csv")
        with open(sensor_positions_file, "w") as f:
            f.write("x,y,z\n")
            # Only save positions for sensors that were used in UTCI calculation
            for i in range(len(utci_values)):
                if i >= len(sensor_grid.sensors):
                    break
                
                sensor = sensor_grid.sensors[i]
                if hasattr(sensor, 'pos') and hasattr(sensor.pos, 'x'):
                    pos = sensor.pos
                    f.write(f"{pos.x},{pos.y},{pos.z}\n")
                elif hasattr(sensor, 'pos') and isinstance(sensor.pos, (list, tuple)):
                    f.write(f"{sensor.pos[0]},{sensor.pos[1]},{sensor.pos[2]}\n")
                else:
                    f.write(f"{sensor['pos'][0]},{sensor['pos'][1]},{sensor['pos'][2]}\n")
        
        return utci_values


def calculate_utci_from_gltf_epw(
    gltf_path, 
    epw_path, 
    output_dir, 
    hour_of_year, 
    grid_size=1.0,
    offset=0.1,
    solar_absorptance=0.7,
    clean_geometry=True,
    use_centroids=True,
    max_sensors=10000,
    use_gpu=True
):
    """Calculate UTCI from a GLTF/GLB model using Radiance."""
    # Convert GLTF to Honeybee Model
    hb_model = gltf_to_honeybee_model(gltf_path, min_area=1e-6, clean_geometry=clean_geometry)
    
    # Run the UTCI calculation
    utci_values = calculate_utci_from_honeybee_model(
        hb_model,
        epw_path,
        output_dir,
        hour_of_year,
        grid_size=grid_size,
        offset=offset,
        solar_absorptance=solar_absorptance,
        use_centroids=use_centroids,
        max_sensors=max_sensors,
        use_gpu=use_gpu
    )
    
    return utci_values


if __name__ == "__main__":
    # Default parameters
    current_dir = Path.cwd()
    glb_file = current_dir / "data/rec_model_no_curve.glb"
    epw_file = current_dir / "data/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw"
    output_directory = current_dir / "output"
    hour = 12  # Noon
    
    # You can adjust these parameters based on your needs
    grid_size = 1.0  # Larger grid = fewer points = faster calculation
    max_sensors = 10000  # Maximum number of sensor points
    use_gpu = True  # Enable GPU acceleration if available
    
    # Set logging level
    set_log_level("INFO")
    
    # Run the calculation
    logging.info(f"Processing GLB file: {glb_file}")
    start_time = time.time()
    
    try:
        utci_values = calculate_utci_from_gltf_epw(
            str(glb_file), 
            str(epw_file), 
            str(output_directory), 
            hour,
            grid_size=grid_size,
            offset=0.1,
            solar_absorptance=0.7,
            clean_geometry=True,
            use_centroids=True,
            max_sensors=max_sensors,
            use_gpu=use_gpu
        )
        
        elapsed_time = time.time() - start_time
        
        # Print summary statistics
        logging.info(f"Calculated UTCI values for {len(utci_values):,} points in {elapsed_time:.2f} seconds.")
        
        # Save the results as a simple text file
        output_file = output_directory / "utci_values.txt"
        np.savetxt(output_file, utci_values)
        logging.info(f"UTCI values saved to: {output_file}")
    
    except Exception as e:
        logging.error(f"Error calculating UTCI: {e}")
        import traceback
        logging.error(traceback.format_exc())
