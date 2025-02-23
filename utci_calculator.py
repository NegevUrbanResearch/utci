from ladybug.epw import EPW
from ladybug_comfort.utci import universal_thermal_climate_index
from ladybug.location import Location
from ladybug.sunpath import Sunpath
from typing import Dict, Tuple
import time
from tqdm import tqdm
import os
import subprocess
import struct
import json
from pathlib import Path
import numpy as np
import pandas as pd

def calculate_utci_from_gltf_epw(
    gltf_path: str,
    epw_path: str,
    output_dir: str,
    hour_of_year: int,
    ground_albedo: float = 0.2,
    solar_absorptance: float = 0.7
) -> np.ndarray:
    """Calculates UTCI from a GLTF/GLB model and EPW file, using Radiance."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    radiance_dir = output_dir / "radiance"
    radiance_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load EPW and get weather data ---
    epw = EPW(epw_path)
    location = Location(epw.location.city, epw.location.state, epw.location.country,
                         epw.location.latitude, epw.location.longitude, epw.location.time_zone,
                         epw.location.elevation)
    sunpath = Sunpath.from_location(location)
    sun = sunpath.calculate_sun_from_hoy(hour_of_year - 1)
    if not sun.is_during_day:
        return np.array([])

    air_temp = epw.dry_bulb_temperature.values[hour_of_year - 1]
    rel_humidity = epw.relative_humidity.values[hour_of_year - 1]
    wind_speed = epw.wind_speed.values[hour_of_year - 1]
    direct_normal_irradiance = epw.direct_normal_radiation.values[hour_of_year - 1]
    diffuse_horizontal_irradiance = epw.diffuse_horizontal_radiation.values[hour_of_year - 1]

    # --- 2. Load GLTF/GLB Model and create Radiance geometry ---
    try:
        json_data, bin_data = _load_gltf_or_glb(gltf_path)  # Use the new function
        vertices = _extract_vertices(json_data, bin_data)
        print(f"Successfully extracted {len(vertices):,} vertices")
        points_file = radiance_dir / "points.pts"
        with open(points_file, "w") as f:
            for vertex in vertices:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Create Radiance geometry from vertices (triangles)
        rad_file = radiance_dir / "model.rad"
        with open(rad_file, "w") as f:
            f.write("void plastic red\n0\n0\n5 1 0 0 0.2 0\n")  # Material
            # Assuming triangles (3 vertices per face)
            for i in range(0, len(vertices), 3):
                f.write("red polygon face_{}\n0\n0\n".format(i // 3))
                f.write("9\n")  # 3 vertices * 3 coordinates
                for j in range(3):
                    f.write(f"{vertices[i+j][0]} {vertices[i+j][1]} {vertices[i+j][2]} ")
                f.write("\n")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # --- 3. Prepare Radiance Files ---

    sky_file = radiance_dir / "sky.sky"
    _run_radiance_command(
        f"gensky {epw.dry_bulb_temperature.datetimes[hour_of_year - 1].month} {epw.dry_bulb_temperature.datetimes[hour_of_year - 1].day} {epw.dry_bulb_temperature.datetimes[hour_of_year - 1].hour + epw.dry_bulb_temperature.datetimes[hour_of_year - 1].minute/60}  -a {location.latitude} -o {location.longitude} -m {-15 * location.time_zone} +s > {sky_file}",
        radiance_dir
    )

    # Create Octree (including the sky file)
    oct_file = radiance_dir / "scene.oct"
    _run_radiance_command(f"oconv {rad_file} {sky_file} > {oct_file}", radiance_dir)

    # --- 4. Radiance Calculations ---
    # A. Calculate Direct Irradiance (using `rtrace`)
    direct_ill_file = radiance_dir / "direct.ill"
    _run_radiance_command(
        f"rtrace -h -ab 0 -ad 1 -lw 0.0001 -n {os.cpu_count()} -y {len(vertices)} {oct_file} < {points_file} | rcalc -e '$1=0;$2=0;$3=0;$4={direct_normal_irradiance};$5={diffuse_horizontal_irradiance};$6=$4+$5' > {direct_ill_file}",
        radiance_dir
    )

     # B. Calculate Indirect + Reflected Irradiance (using `rtrace`)
    indirect_ill_file = radiance_dir / "indirect.ill"

    _run_radiance_command(
        f"rtrace -h -ab 1 -ad 1000 -lw 0.0001 -n {os.cpu_count()} -y {len(vertices)} {oct_file} < {points_file} > {indirect_ill_file}",
        radiance_dir
    )
    # --- 5. Load Radiance Results and Calculate MRT ---
    direct_irradiance = _load_ill_file(direct_ill_file)
    indirect_irradiance = _load_ill_file(indirect_ill_file)
    min_len = min(len(direct_irradiance), len(indirect_irradiance), len(vertices))
    direct_irradiance = direct_irradiance[:min_len]
    indirect_irradiance = indirect_irradiance[:min_len]
    longwave_mrt = air_temp
    shortwave_mrt_delta = (solar_absorptance * (direct_irradiance + indirect_irradiance) / 0.95)
    mean_radiant_temperature = longwave_mrt + shortwave_mrt_delta

    # --- 6. Calculate UTCI ---
    utci_values = universal_thermal_climate_index(
        air_temp, mean_radiant_temperature, wind_speed, rel_humidity
    )
    return utci_values


def _run_radiance_command(command: str, cwd: Path):
    """Runs a Radiance command, handling errors."""

    if "RAYPATH" not in os.environ:
        print("WARNING: RAYPATH environment variable not set.  Radiance may not work correctly.")
        print("  You should set it to the directory containing the Radiance 'lib' folder.")
        print("  For example:  export RAYPATH=/usr/local/radiance/lib")

    try:
        print(f"Running Radiance command:\n{command}\n")
        log_file_path = cwd / "radiance_log.txt"

        # Split command if it contains file redirection
        if ">" in command:
            cmd_parts = command.split(">")
            base_command = cmd_parts[0].strip()
            output_file = Path(cwd / cmd_parts[1].strip())

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as out_file:
                process = subprocess.Popen(
                    base_command,
                    shell=True,
                    stdout=out_file,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd,
                )
        else:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
            )

        start_time = time.time()
        with tqdm(total=100, desc="Radiance Progress", unit="%") as pbar:
            stderr_output = process.stderr.read()
            if stderr_output:
                print(f"Standard Error:\n{stderr_output}")
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"Standard Error:\n{stderr_output}")

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, command, stderr=stderr_output
                )
            elapsed_time = time.time() - start_time
            pbar.update(100)
            print(f"Command completed. Elapsed Time: {elapsed_time:.2f} seconds")
            return process


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
        # GLB loading (same as before)
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

        # Find the binary buffer (usually a .bin file)
        if 'buffers' not in json_data or len(json_data['buffers']) == 0:
             raise ValueError("No buffers found in GLTF file.")

        buffer_uri = json_data['buffers'][0]['uri']

        # Check if it's a data URI or a file path
        if buffer_uri.startswith('data:'):
            # Handle data URI (base64 encoded)
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
    """Extract vertices from parsed GLTF/GLB data."""
    vertices = []
    for mesh in json_data.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            accessor = json_data['accessors'][primitive['attributes']['POSITION']]
            buffer_view = json_data['bufferViews'][accessor['bufferView']]
            start = buffer_view.get('byteOffset', 0)
            length = buffer_view['byteLength']
            stride = buffer_view.get('byteStride', 12) or 12  # Default stride
            vertex_data = bin_data[start: start + length]
            vertices.extend(np.frombuffer(vertex_data, dtype=np.float32).reshape(-1, 3))
    return np.array(vertices)


def _load_ill_file(filepath: str) -> np.ndarray:
    """Loads irradiance values from a Radiance .ill file."""
    try:
        with open(filepath, "r") as f:
            values = []
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        irradiance = sum(float(x) for x in parts[-3:])
                        values.append(irradiance)
                    except (ValueError, IndexError):
                        print(f"Warning: Skipping line due to format error: {line.strip()}")
            return np.array(values)
    except FileNotFoundError:
        print(f"Error: .ill file not found: {filepath}")
        raise
    except Exception as e:
        print(f"Error reading .ill file: {e}")
        raise

def example_usage():
    """Example usage."""
    current_dir = Path(__file__).parent
    gltf_file = current_dir / "test_main_data/rec_model.glb"   # Corrected path
    epw_file = current_dir / "data/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw"
    output_directory = current_dir / "utci_results"
    hour = 2000

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    radiance_dir = output_dir / "radiance"
    radiance_dir.mkdir(parents=True, exist_ok=True)

    utci_values = calculate_utci_from_gltf_epw(str(gltf_file), str(epw_file), str(output_directory), hour)

    if len(utci_values) > 0:
        print(f"Calculated UTCI values for {len(utci_values)} points.")
        print(f"UTCI values (first 10): {utci_values[:10]}")
        output_file = Path(output_directory) / "utci_values.txt"
        np.savetxt(output_file, utci_values)
        print(f"UTCI values saved to: {output_file}")
    else:
        print(f'No UTCI values to show for hour {hour}')


if __name__ == "__main__":
    example_usage()