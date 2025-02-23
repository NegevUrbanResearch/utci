# utci_calculator.py
import os
from pathlib import Path
import numpy as np
from ladybug.epw import EPW
from ladybug_comfort.map.utci import universal_thermal_climate_index_np
import pandas as pd
import matplotlib.pyplot as plt
import json
import struct
from typing import Dict, Tuple, List, Union
import time
from datetime import datetime, timedelta
import h5py  # Import h5py
from ladybug.location import Location
from ladybug.sunpath import Sunpath
from ladybug_comfort.map.mrt import shortwave_mrt_map  # Import shortwave_mrt_map
import subprocess  # Import subprocess


class UTCICalculator:
    def __init__(self, gltf_path: str, epw_path: str, output_dir: str = 'output', subsample: bool = True):
        self.gltf_path = Path(gltf_path)
        self.epw_path = Path(epw_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._validate_paths()
        self.max_plot_points = 100000  # Control subsampling for plotting
        self.max_json_points = 100000  # Control subsampling for JSON output
        self.subsample = subsample # Controls whether to subsample or not
        self.rad_dir = self.output_dir / "radiance"
        self.rad_dir.mkdir(parents=True, exist_ok=True)


    def _validate_paths(self):
        if not self.gltf_path.exists():
            raise FileNotFoundError(f"GLTF file not found: {self.gltf_path}")
        if not self.epw_path.exists():
            raise FileNotFoundError(f"EPW file not found: {self.epw_path}")
        if not self.gltf_path.suffix.lower() in ('.glb', '.gltf'):
            raise ValueError(f"Invalid GLTF file extension: {self.gltf_path.suffix}")

    def _parse_glb(self, filepath: Path) -> Tuple[Dict, bytes]:
        """Parse GLB file manually."""
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            if magic != b'glTF':
                raise ValueError("Not a valid GLB file")
            version, length = struct.unpack('<II', f.read(8))
            chunk_length, chunk_type = struct.unpack('<II', f.read(8))
            if chunk_type != 0x4E4F534A:  # JSON in little-endian
                raise ValueError("Expected JSON chunk")
            json_data = json.loads(f.read(chunk_length))
            chunk_length, chunk_type = struct.unpack('<II', f.read(8))
            if chunk_type != 0x004E4942:  # BIN in little-endian
                raise ValueError("Expected BIN chunk")
            bin_data = f.read(chunk_length)
            return json_data, bin_data

    def _extract_vertices(self, json_data: Dict, bin_data: bytes) -> np.ndarray:
        """Extract vertices from parsed GLB data."""
        vertices = []
        for mesh in json_data.get('meshes', []):
            for primitive in mesh.get('primitives', []):
                accessor = json_data['accessors'][primitive['attributes']['POSITION']]
                buffer_view = json_data['bufferViews'][accessor['bufferView']]
                start = buffer_view.get('byteOffset', 0)
                length = buffer_view['byteLength']
                stride = buffer_view.get('byteStride', 12) or 12
                vertex_data = bin_data[start:start + length]
                count = length // 4
                vertices_array = np.frombuffer(vertex_data, dtype=np.float32, count=count).reshape(-1, 3)
                vertices.extend(vertices_array)  # Corrected line
        return np.array(vertices)

    def load_model(self) -> np.ndarray:
        try:
            json_data, bin_data = self._parse_glb(self.gltf_path)
            vertices = self._extract_vertices(json_data, bin_data)
            print(f"Successfully extracted {len(vertices):,} vertices")
            return vertices
        except Exception as e:
            print(f"Error in load_model: {e}")
            raise

    def load_weather_data(self) -> pd.DataFrame:
        try:
            epw = EPW(self.epw_path.as_posix())
            data = {
                'air_temp': epw.dry_bulb_temperature.values,
                'rel_humidity': epw.relative_humidity.values,
                'wind_speed': epw.wind_speed.values,
                'mean_rad_temp': epw.dry_bulb_temperature.values  # Placeholder, will be overwritten
            }
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error loading weather data: {e}")
            raise

    def _run_radiance_simulation(self, hour_index: int):
        """Runs Radiance simulations to generate .ill files."""

        if not self.check_radiance_dependencies():
            raise EnvironmentError("Radiance is not installed or not in the system PATH.")

        # 1. Convert GLTF to Radiance format (using obj2rad as a placeholder)
        rad_file = self.rad_dir / "model.rad"
        # In a real implementation, you'd use a proper GLTF to Radiance converter.
        # This is a SIMPLIFIED example.
        with open(rad_file, "w") as f:
            f.write("void plastic red\n0\n0\n5 1 0 0 0.2 0\n")  # Simple material
            f.write(f"red sphere model\n0\n0\n4 0 0 0 1\n") # Simple sphere

        # 2. Create a Radiance scene description
        oct_file = self.rad_dir / "scene.oct"
        with open(self.rad_dir / "scene.rad", "w") as f:
            f.write(f"!oconv {rad_file} > {oct_file}\n")  # Very basic oconv command.

        # 3.  Generate Sun Matrix
        epw = EPW(self.epw_path.as_posix())
        location = epw.location
        sun_vector = location.sun_vector(epw.analysis_period.datetimes[hour_index])
        smx_file = self.rad_dir / "sun.smx"
        with open(smx_file, 'w') as f:
             f.write(f'!genskyvec -c {sun_vector.x} {sun_vector.y} {sun_vector.z}  -m 1 | rcolmap -t 3 -h -ff > {smx_file} \n')

        # 4. Prepare vertices for rfluxmtx
        vertices = self.load_model()
        points_file = self.rad_dir / "points.pts"
        with open(points_file, "w") as f:
            for vertex in vertices:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Create sunuphours file (required by shortwave_mrt_map)
        sun_up_hours_file = self.rad_dir / "sun_up_hours.txt"
        with open(sun_up_hours_file, "w") as f:
            f.write(f"{hour_index}\n")  # Write the current hour.

        # 5. Run rfluxmtx for direct, indirect, and reflected
        direct_ill = self.rad_dir / "direct.ill"
        indirect_ill = self.rad_dir / "indirect.ill"
        reflected_ill = self.rad_dir / "reflected.ill"
       
        # Placeholder commands.  Real commands would need sky definitions, etc.
        commands = [
            f"rfluxmtx -ab 0 -ad 1 -n {os.cpu_count()} -y {len(vertices)} -I+ -lw 0.0001 {smx_file} < {points_file} > {direct_ill}",
            f"rfluxmtx -ab 1 -ad 1000 -n {os.cpu_count()} -y {len(vertices)} -I+ -lw 0.0001 {smx_file} < {points_file} > {indirect_ill}",
            f"rfluxmtx -ab 1 -ad 1000 -n {os.cpu_count()} -y {len(vertices)} -I+ -lw 0.0001 {smx_file} < {points_file} > {reflected_ill}", # Assuming ground is part of the scene.
            f"oconv {self.rad_dir / 'scene.rad'} > {oct_file}" # must execute before rfluxmtx
        ]

        for command in commands:
            try:
                subprocess.run(command, shell=True, check=True, cwd=self.rad_dir, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Radiance command failed:\nCommand: {e.cmd}\nError: {e.stderr.decode()}")
                raise

        return str(direct_ill), str(indirect_ill), str(reflected_ill), str(sun_up_hours_file)



    def calculate_utci(self, points: np.ndarray, weather_data: pd.DataFrame, hour_index: int = 12) -> np.ndarray:
        """Calculates UTCI, utilizing shortwave_mrt_map."""

        if not 0 <= hour_index < len(weather_data):
            raise ValueError(f"Invalid hour_index: {hour_index}")

        weather = weather_data.iloc[hour_index]
        epw = EPW(self.epw_path.as_posix())
        location = epw.location

        # Run Radiance simulations to generate .ill files
        direct_ill, indirect_ill, reflected_ill, sun_up_hours_file = self._run_radiance_simulation(hour_index)


        # Prepare longwave MRT data (simplified: equal to air temperature)
        longwave_mrt_data = [pd.DataFrame({'mean_radiant_temperature': [weather['air_temp']]}) for _ in range(len(points))]

        # Calculate shortwave-adjusted MRT using ladybug_comfort
        mrt_data = shortwave_mrt_map(
            location=location,
            longwave_data=longwave_mrt_data,
            sun_up_hours=sun_up_hours_file,
            indirect_ill=indirect_ill,
            direct_ill=direct_ill,
            ref_ill=reflected_ill
        )

        # Extract MRT values from the result (should be a list of DataCollections)
        mrt_values = np.array([data.values[0] for data in mrt_data])

        # Now calculate UTCI using the *combined* MRT
        total_points = len(points)
        if total_points == 0:
            return np.array([])
        utci_values = np.zeros(total_points)
        ta = np.full(total_points, weather['air_temp'])
        tr = mrt_values # Use calculated MRT
        vel = np.full(total_points, weather['wind_speed'])
        rh = np.full(total_points, weather['rel_humidity'])
        vel = np.maximum(vel, 0.5)
        assert np.all(vel >= 0.5)
        utci_values[:] = universal_thermal_climate_index_np(ta, tr, vel, rh) # Simplified calculation

        return utci_values
    def _print_progress(self, start_time: float, current_time: float, points_processed: int, total_points: int):
        elapsed_time = current_time - start_time
        progress = points_processed / total_points
        estimated_total_time = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total_time - elapsed_time
        print(f"Progress: {progress*100:.1f}% ({points_processed:,}/{total_points:,} points)")
        print(f"Elapsed: {timedelta(seconds=int(elapsed_time))}")
        print(f"Estimated remaining: {timedelta(seconds=int(remaining_time))}")
        print(f"Processing speed: {points_processed/elapsed_time:,.0f} points/second")
        print()

    def create_visualization(self, points: np.ndarray, utci_values: np.ndarray, hour_index: int):
      print("Creating visualizations...")
      if len(points) == 0 or len(utci_values) == 0:
          print("Warning: No data to visualize.")
          return
      if len(points) != len(utci_values):
          raise ValueError("Mismatch between points and UTCI values length.")

      # Subsampling for plotting
      if self.subsample and len(points) > self.max_plot_points :
          indices = np.random.choice(len(points), self.max_plot_points, replace=False)
          plot_points = points[indices]
          plot_utci_values = utci_values[indices]
          print(f"Subsampling to {self.max_plot_points:,} points for visualization.")
      else:
          plot_points = points
          plot_utci_values = utci_values

      categories = [
          (-40, 'Extreme Cold Stress', 'darkblue'),
          (-27, 'Very Strong Cold Stress', 'blue'),
          (-13, 'Strong Cold Stress', 'lightblue'),
          (0, 'Moderate Cold Stress', 'green'),
          (9, 'No Thermal Stress', 'yellow'),
          (26, 'Moderate Heat Stress', 'orange'),
          (32, 'Strong Heat Stress', 'red'),
          (38, 'Very Strong Heat Stress', 'darkred'),
          (46, 'Extreme Heat Stress', 'purple')
      ]

      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [3, 1]})

      try:
          scatter = ax1.scatter(plot_points[:, 0], plot_points[:, 1], c=plot_utci_values, cmap='RdYlBu_r', s=20)
          ax1.set_title(f'UTCI Distribution (Hour {hour_index})')
          ax1.set_xlabel('X coordinate (m)')
          ax1.set_ylabel('Y coordinate (m)')
          plt.colorbar(scatter, ax=ax1, label='UTCI (°C)')
          ax2.set_title('UTCI Thermal Stress Categories')
          for i, (temp, label, color) in enumerate(categories):
              ax2.barh(i, 1, color=color)
              ax2.text(1.1, i, f'{label}\n(>{temp}°C)', va='center')
          ax2.set_ylim(-0.5, len(categories) - 0.5)
          ax2.set_xlim(0, 1)
          ax2.axis('off')
          plt.tight_layout()
          filepath = self.output_dir / f'utci_visualization_hour_{hour_index}.png'
          plt.savefig(filepath, dpi=300, bbox_inches='tight')
          plt.close()
          print(f"Visualization saved to: {filepath}")
      except Exception as e:
          print(f"Error creating visualization: {e}")
          plt.close()
          raise

      self._save_results(points, utci_values, hour_index)

    def _save_results(self, points: np.ndarray, utci_values: np.ndarray, hour_index: int):
        """Saves results to JSON (subsampled) or HDF5 (full data)."""

        if self.subsample:
            # Subsampling for JSON
            if len(points) > self.max_json_points:
                indices = np.random.choice(len(points), self.max_json_points, replace=False)
                json_points = points[indices]
                json_utci_values = utci_values[indices]
                print(f"Subsampling to {self.max_json_points:,} points for JSON output.")
            else:
                json_points = points
                json_utci_values = utci_values


            results_dict = {
                'metadata': {
                    'hour_index': hour_index,
                    'num_points_calculated': len(points),
                    'num_points_saved': len(json_points),
                    'min_utci': float(np.min(utci_values)),
                    'max_utci': float(np.max(utci_values)),
                    'mean_utci': float(np.mean(utci_values))
                },
                'point_data': [{
                    'coordinates': point.tolist(),
                    'utci': float(utci)
                } for point, utci in zip(json_points, json_utci_values)]
            }

            filepath = self.output_dir / f'utci_results_hour_{hour_index}.json'
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"Results saved to: {filepath}")

        else:
            # Save all data to HDF5 (example)
            filepath = self.output_dir / f'utci_results_hour_{hour_index}.h5'
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('points', data=points)
                f.create_dataset('utci_values', data=utci_values)
            print(f"Full results saved to HDF5: {filepath}")



    def run_analysis(self, hour_index: int = 12) -> Dict:
        """Main analysis function."""
        try:
            points = self.load_model()
            weather_data = self.load_weather_data()
            utci_values = self.calculate_utci(points, weather_data, hour_index)
            self.create_visualization(points, utci_values, hour_index)
            print("\nUTCI Analysis Complete!")
            print(f"Number of points analyzed: {len(points):,}")
            print(f"UTCI Range: {np.min(utci_values):.1f}°C to {np.max(utci_values):.1f}°C")
            print(f"Mean UTCI: {np.mean(utci_values):.1f}°C")
            return {
                'points': points,
                'utci_values': utci_values,
                'summary': {
                    'min_utci': float(np.min(utci_values)),
                    'max_utci': float(np.max(utci_values)),
                    'mean_utci': float(np.mean(utci_values))
                }
            }
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise

    def check_radiance_dependencies(self) -> bool:
        """Checks if Radiance is installed and accessible in the system's PATH."""
        try:
            # Check for rfluxmtx as a proxy for Radiance installation
            subprocess.run(["rfluxmtx", "-version"], check=True, capture_output=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calculate UTCI from GLTF and EPW files.")
    parser.add_argument("gltf_path", type=str, help="Path to the GLTF file.")
    parser.add_argument("epw_path", type=str, help="Path to the EPW file.")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Output directory.")
    parser.add_argument("-H", "--hour", type=int, default=12, help="Hour for UTCI calculation (0-23).")
    parser.add_argument("-p", "--plot_points", type=int, default=100000, help="Max points for plotting (only if subsampling).")
    parser.add_argument("-j", "--json_points", type=int, default=100000, help="Max points for JSON output (only if subsampling).")
    parser.add_argument("--no-subsample", action="store_false", dest="subsample", help="Disable subsampling and save all data (use with caution!).")

    args = parser.parse_args()

    calculator = UTCICalculator(args.gltf_path, args.epw_path, args.output_dir, subsample=args.subsample)
    calculator.max_plot_points = args.plot_points
    calculator.max_json_points = args.json_points
    try:
        calculator.run_analysis(hour_index=args.hour)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


    # Example of how to run from the command line:
    #
    # 1. Default behavior (subsampling for both plotting and JSON):
    #    python utci_calculator.py data/rec_model.gltf /Users/noamgal/DSProjects/utci/data/ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw
    # 
    # 2. Specify output directory and hour:
    #    python utci_calculator.py path/to/model.gltf path/to/weather.epw -o my_output -H 14
    #
    # 3. Change the number of points for plotting and JSON:
    #    python utci_calculator.py path/to/model.gltf path/to/weather.epw -p 50000 -j 25000
    #
    # 4. Disable subsampling (save all data to HDF5):
    #    python utci_calculator.py path/to/model.gltf path/to/weather.epw --no-subsample
    #    WARNING: This can create very large files and may require significant memory.
    #

if __name__ == "__main__":
    main()