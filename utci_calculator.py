from ladybug.epw import EPW
from ladybug_comfort.utci import universal_thermal_climate_index
from ladybug.location import Location
from ladybug.sunpath import Sunpath
from typing import Dict, Tuple, List
import time
from tqdm import tqdm
import os
import subprocess
import struct
import json
from pathlib import Path
import numpy as np
import pandas as pd  # Although not used directly, good practice to keep


def calculate_utci_from_gltf_epw(
    gltf_path: str,
    epw_path: str,
    output_dir: str,
    hour_of_year: int,
    ground_albedo: float = 0.2,
    solar_absorptance: float = 0.7
) -> np.ndarray:
    """Calculates UTCI from a GLB model and EPW file, using Radiance."""
    gltf_path = Path(gltf_path)
    
    # Simplified validation - only accept .glb files
    if gltf_path.suffix.lower() != '.glb':
        raise ValueError(f"File must have .glb extension, got: {gltf_path}")
    
    # Validate GLB format
    with open(gltf_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'glTF':
            raise ValueError(f"File {gltf_path} is not a valid GLB file")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists
    radiance_dir = output_dir / "radiance"
    radiance_dir.mkdir(parents=True, exist_ok=True)  # Ensure radiance dir exists

    # --- 1. Load EPW and get weather data ---
    try:
        epw = EPW(epw_path)
        location = Location(epw.location.city, epw.location.state, epw.location.country,
                             epw.location.latitude, epw.location.longitude, epw.location.time_zone,
                             epw.location.elevation)
    except Exception as e:
        print(f"Error loading EPW file: {e}")
        return np.array([])

    sunpath = Sunpath.from_location(location)
    sun = sunpath.calculate_sun_from_hoy(hour_of_year - 1)  # Corrected indexing
    if not sun.is_during_day:
        print(f"Hour {hour_of_year} is not during the day. Returning empty array.")
        return np.array([])

    air_temp = epw.dry_bulb_temperature.values[hour_of_year - 1]
    rel_humidity = epw.relative_humidity.values[hour_of_year - 1]
    wind_speed = epw.wind_speed.values[hour_of_year - 1]
    direct_normal_irradiance = epw.direct_normal_radiation.values[hour_of_year - 1]
    diffuse_horizontal_irradiance = epw.diffuse_horizontal_radiation.values[hour_of_year - 1]

    # --- 2. Load GLTF/GLB Model and create Radiance geometry ---
    try:
        json_data, bin_data = _load_gltf_or_glb(gltf_path)
        vertices = _extract_vertices(json_data, bin_data)
        if len(vertices) == 0:
            print("Warning: No vertices extracted from the GLTF/GLB model.")
            return np.array([])
        if len(vertices) % 3 != 0 :
            raise ValueError("Number of vertices must be divisible by 3 (triangles).")
        print(f"Successfully extracted {len(vertices):,} vertices")

        # Add validation for vertex count
        print(f"\nProcessing {len(vertices):,} vertices...")
        
        # Ensure points file ends with newline
        points_file = radiance_dir / "points.pts"
        with open(points_file, "w") as f:
            for vertex in vertices:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
            f.write("\n")  # Add final newline

        rad_file = radiance_dir / "model.rad"
        with open(rad_file, "w") as f:
            f.write("void plastic red\n0\n0\n5 1 0 0 0.2 0\n")  # Material definition

            for i in range(0, len(vertices), 3):
                f.write("red polygon face_{}\n0\n0\n".format(i // 3))
                f.write("9\n")
                f.write(f"{vertices[i][0]} {vertices[i][1]} {vertices[i][2]} ")
                f.write(f"{vertices[i+1][0]} {vertices[i+1][1]} {vertices[i+1][2]} ")
                f.write(f"{vertices[i+2][0]} {vertices[i+2][1]} {vertices[i+2][2]}\n")

    except Exception as e:
        print(f"Error loading or processing model: {e}")
        return np.array([])

    # Generate and validate sky model
    sky_file = _generate_sky_from_epw(epw, hour_of_year, location, radiance_dir)
    if sky_file is None:
        print("Error: Failed to generate sky model")
        return np.array([])
    
    # --- 4. Radiance Calculations ---
    oct_file = radiance_dir / "scene.oct"  # Absolute path
    _run_radiance_command(f"oconv -f {rad_file} {sky_file} > {oct_file}", radiance_dir)

    direct_ill_file = radiance_dir / "direct.ill"
    _run_radiance_command(
        f"rtrace -w -h -ab 0 -ad 1 -lw 0.0001 -n {os.cpu_count()} "
        f"-x {len(vertices)} "
        f"-y 1 "
        f"{oct_file} < {points_file} > {direct_ill_file}",
        radiance_dir
    )

    indirect_ill_file = radiance_dir / "indirect.ill"
    _run_radiance_command(
        f"rtrace -h -ab 1 -ad 1000 -lw 0.0001 -n {os.cpu_count()} "
        f"-x {len(vertices)} "  # Specify expected number of points
        f"-y 1 "  # Process one point at a time
        f"{oct_file} < {points_file} > {indirect_ill_file}",
        radiance_dir
    )

    # --- 5. Load Radiance Results and Calculate MRT ---
    try:
        direct_irradiance = _load_ill_file(direct_ill_file)
        indirect_irradiance = _load_ill_file(indirect_ill_file)
    except Exception as e:
        print(f"Error loading Radiance results: {e}")
        return np.array([])

    # Improve irradiance validation
    if len(direct_irradiance) != len(vertices) or len(indirect_irradiance) != len(vertices):
        print(f"\nWarning: Mismatch in number of values:")
        print(f"  Vertices: {len(vertices):,}")
        print(f"  Direct irradiance values: {len(direct_irradiance):,}")
        print(f"  Indirect irradiance values: {len(indirect_irradiance):,}")
        min_len = min(len(vertices), len(direct_irradiance), len(indirect_irradiance))
        print(f"  Using first {min_len:,} values for calculation")
        vertices = vertices[:min_len]
        direct_irradiance = direct_irradiance[:min_len]
        indirect_irradiance = indirect_irradiance[:min_len]
    longwave_mrt = air_temp
    shortwave_mrt_delta = (solar_absorptance * (direct_irradiance + indirect_irradiance) / 5.67e-8)**0.25 - 273.15
    mean_radiant_temperature = (shortwave_mrt_delta**4 + longwave_mrt**4)**0.25


    # --- 6. Calculate UTCI ---
    utci_values = universal_thermal_climate_index(
        air_temp, mean_radiant_temperature, wind_speed, rel_humidity
    )
    return utci_values



def _run_radiance_command(command: str, cwd: Path):
    """Runs a Radiance command, handling errors and warnings."""

    try:
        print(f"Running Radiance command:\n{command}\n")
        log_file_path = cwd / "radiance_log.txt"

        # Split command if it contains file redirection
        if ">" in command:
            cmd_parts = command.split(">")
            base_command = cmd_parts[0].strip()
            output_file = Path(cwd / cmd_parts[1].strip())  # Keep using cwd

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as out_file:
                process = subprocess.Popen(
                    base_command,
                    shell=True,
                    stdout=out_file,
                    stderr=subprocess.PIPE,  # Capture stderr
                    text=True,
                    cwd=cwd,
                )
                stderr_output = process.stderr.read()  # Read stderr
                if stderr_output:
                    print(f"Standard Error:\n{stderr_output}")
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"Command: {command}\n")
                        log_file.write(f"STDERR: {stderr_output}\n")
        else:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,  # Capture stdout
                stderr=subprocess.PIPE,  # Capture stderr
                text=True,
                cwd=cwd,
            )

            stdout_output, stderr_output = process.communicate()  # Get both stdout and stderr
            
            # Log outputs
            with open(log_file_path, "a") as log_file: # Append to log file
                log_file.write(f"\nCommand: {command}\n")
                if stdout_output:
                    print(f"Standard Output:\n{stdout_output}")
                    log_file.write(f"STDOUT: {stdout_output}\n")
                if stderr_output:
                    print(f"Standard Error:\n{stderr_output}")
                    log_file.write(f"STDERR: {stderr_output}\n")


        process.wait()  # Wait for the process to finish
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, stderr=stderr_output
            )

        print(f"Command completed. Return code: {process.returncode}")
        return process  # Return the process object

    except subprocess.CalledProcessError as e:
        print(f"Radiance command failed:\nCommand: {e.cmd}\nReturn Code: {e.returncode}\nError: {e.stderr}")
        raise
    except FileNotFoundError as e:
        print("Radiance command not found. Make sure Radiance is installed and in your system's PATH.")
        raise


def _load_gltf_or_glb(filepath: str) -> Tuple[Dict, bytes]:
    """Loads either a GLB or a GLTF file, handling external buffers."""
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
        raise ValueError("Unsupported file extension.  Must be .glb or .gltf")



def _extract_vertices(json_data: Dict, bin_data: bytes) -> np.ndarray:
    """Extract vertices from GLB data, handling indexed geometry."""
    all_vertices = []
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
            pos_stride = pos_buffer_view.get('byteStride', 12)  # Default for VEC3 of floats
            
            # Extract position data
            pos_data = bin_data[pos_offset:pos_offset + pos_length]
            positions = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 3)

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
                
                # Index into positions
                vertices = positions[indices]
                all_vertices.extend(vertices)
            else:
                # Non-indexed geometry
                all_vertices.extend(positions)

    return np.array(all_vertices)


def _load_ill_file(filepath: str) -> np.ndarray:
    """Loads irradiance values from a Radiance .ill file."""
    try:
        with open(filepath, "r") as f:
            values = []
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        irradiance = sum(float(x) for x in parts[-3:])  # Sum the last three values
                        values.append(irradiance)
                    except (ValueError, IndexError):
                        print(f"Warning: Skipping line due to format error: {line.strip()}")
            return np.array(values)
    except FileNotFoundError:
        print(f"Error: .ill file not found: {filepath}")
        raise  # Re-raise the exception to stop execution
    except Exception as e:
        print(f"Error reading .ill file: {e}")
        raise


def _generate_sky_from_epw(
    epw_data: EPW,
    hour: int,
    location: Location,
    output_dir: Path
) -> Path:
    """
    Generate a Radiance sky model using EPW data with validation and verification.
    
    Args:
        epw_data: EPW weather data
        hour: Hour of year (1-8760)
        location: Location data
        output_dir: Directory for output files
    
    Returns:
        Path to generated sky file
    """
    datetime = epw_data.dry_bulb_temperature.datetimes[hour - 1]
    direct_normal = epw_data.direct_normal_radiation.values[hour - 1]
    diffuse_horizontal = epw_data.diffuse_horizontal_radiation.values[hour - 1]
    
    # Validate inputs
    if not 1 <= hour <= 8760:
        raise ValueError(f"Hour must be between 1 and 8760, got {hour}")
    
    if not -90 <= location.latitude <= 90:
        raise ValueError(f"Invalid latitude: {location.latitude}")
    if not -180 <= location.longitude <= 180:
        raise ValueError(f"Invalid longitude: {location.longitude}")
    
    # Validate radiation values
    if direct_normal < 0 or diffuse_horizontal < 0:
        print(f"Warning: Negative radiation values detected:")
        print(f"  Direct normal: {direct_normal}")
        print(f"  Diffuse horizontal: {diffuse_horizontal}")
        # Set to 0 if negative
        direct_normal = max(0, direct_normal)
        diffuse_horizontal = max(0, diffuse_horizontal)
    
    # Create output files
    sky_file = output_dir / "sky.sky"
    sky_info_file = output_dir / "sky_info.txt"
    
    # Modified sky generation command with proper material definitions
    with open(sky_file, 'w') as f:
        # First write the gensky command as a comment
        f.write(f"# gensky {datetime.month} {datetime.day} "
                f"{datetime.hour + datetime.minute/60:.2f} "
                f"-a {location.latitude:.4f} -o {location.longitude:.4f} "
                f"-m {location.time_zone * 15:.1f} "  # Convert timezone to meridian degrees
                f"-B {direct_normal:.1f} "
                f"-g {diffuse_horizontal:.1f} "
                f"+s\n\n")  # Always use sunny sky with sun for UTCI
        
        # Define sky material and geometry
        f.write("void light solar\n0\n0\n3 1e6 1e6 1e6\n\n")  # Solar source
        f.write("solar source sun\n0\n0\n4 0 0 1 0.533\n\n")  # Sun geometry
        
        f.write("void light sky_mat\n0\n0\n3 1e1 1e1 1e1\n\n")  # Sky material
        f.write("sky_mat source sky\n0\n0\n4 0 0 1 180\n\n")  # Sky dome
        
        # Ground material and geometry (optional but recommended)
        f.write("void plastic ground_mat\n0\n0\n5 .2 .2 .2 0 0\n\n")
        f.write("ground_mat source ground\n0\n0\n4 0 0 -1 180\n")
    
    # Verify sky model
    if not sky_file.exists():
        print(f"Error: Failed to create sky file at {sky_file}")
        return None
        
    # Verify sky model validity
    sky_issues = []
    if 'void light sky' not in sky_file.read_text():
        sky_issues.append("No sky light source found")
    if 'void light solar' not in sky_file.read_text():
        sky_issues.append("No solar light source found")
    if direct_normal > 0 and 'solar source' not in sky_file.read_text().lower():
        sky_issues.append("Missing solar source despite non-zero direct radiation")
    
    if sky_issues:
        print("\nWarning: Sky model validation issues detected:")
        for issue in sky_issues:
            print(f"  - {issue}")
        print(f"\nSky model details saved to: {sky_info_file}")
        print("Please check the sky_info.txt file for more information.")
    else:
        print("\nSky model generated and validated successfully.")
        print(f"Sky model details saved to: {sky_info_file}")
    
    return sky_file


def example_usage():
    """Example usage with GLB file."""
    current_dir = Path(__file__).parent
    glb_file = current_dir / "data/rec_model_cleaned.glb"
    epw_file = current_dir / "data/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw"
    output_directory = current_dir / "output"
    hour = 12  # Changed from 2000 to 12 (noon) for daytime calculation

    # Ensure the file exists before proceeding
    if not glb_file.exists():
        print(f"Error: GLB file not found at {glb_file}")
        print("Please ensure the cleaned model is saved with .glb extension")
        return

    utci_values = calculate_utci_from_gltf_epw(str(glb_file), str(epw_file), str(output_directory), hour)

    if len(utci_values) > 0:
        print(f"Calculated UTCI values for {len(utci_values)} points.")
        print(f"UTCI values (first 10): {utci_values[:10]}")
        output_file = output_directory / "utci_values.txt"
        output_directory.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_file, utci_values)
        print(f"UTCI values saved to: {output_file}")
    else:
        print(f'No UTCI values to show for hour {hour}')



if __name__ == "__main__":
    example_usage()