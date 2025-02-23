import os
from pathlib import Path
import numpy as np
from ladybug.epw import EPW
from ladybug_comfort.map.utci import universal_thermal_climate_index_np
import pandas as pd
import matplotlib.pyplot as plt
import json
import struct
from typing import Dict, Tuple, List, Union  # Added type hints
import time
from datetime import datetime, timedelta

class UTCICalculator:
    def __init__(self, gltf_path: str, epw_path: str, output_dir: str = 'output'):
        self.gltf_path = Path(gltf_path)
        self.epw_path = Path(epw_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Create parent dirs if needed
        self._validate_paths()  # Private method for validation

    def _validate_paths(self):
        if not self.gltf_path.exists():
            raise FileNotFoundError(f"GLTF file not found: {self.gltf_path}")
        if not self.epw_path.exists():
            raise FileNotFoundError(f"EPW file not found: {self.epw_path}")
        if not self.gltf_path.suffix.lower() in ('.glb', '.gltf'):
            raise ValueError(f"Invalid GLTF file extension: {self.gltf_path.suffix}")

    def _parse_glb(self, filepath: Path) -> Tuple[Dict, bytes]:
        """Parse GLB file manually to extract vertex data."""
        print(f"Parsing file as GLB: {filepath}")
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            if magic != b'glTF':
                raise ValueError("Not a valid GLB file")

            version = struct.unpack('<I', f.read(4))[0]
            length = struct.unpack('<I', f.read(4))[0]

            print(f"GLB version: {version}")
            print(f"File length: {length} bytes")

            # Read JSON chunk
            chunk_length = struct.unpack('<I', f.read(4))[0]
            chunk_type = f.read(4)
            if chunk_type != b'JSON':
                raise ValueError("Expected JSON chunk")

            json_data = json.loads(f.read(chunk_length))

            # Read BIN chunk
            chunk_length = struct.unpack('<I', f.read(4))[0]
            chunk_type = f.read(4)
            if chunk_type != b'BIN\x00':
                raise ValueError("Expected BIN chunk")

            bin_data = f.read(chunk_length)

            return json_data, bin_data

    def _extract_vertices(self, json_data: Dict, bin_data: bytes) -> np.ndarray:
        """Extract vertices from parsed GLB data."""
        print("Extracting vertices from GLB data...")
        vertices = []

        for mesh in json_data.get('meshes', []):
            for primitive in mesh.get('primitives', []):
                position_accessor = json_data['accessors'][primitive['attributes']['POSITION']]
                buffer_view = json_data['bufferViews'][position_accessor['bufferView']]

                start = buffer_view.get('byteOffset', 0)
                length = buffer_view['byteLength']
                stride = buffer_view.get('byteStride', 12) or 12  # Default stride if missing
                
                vertex_data = bin_data[start:start + length]
                
                vertices_array = np.frombuffer(
                    vertex_data,
                    dtype=np.float32,
                    count=length // 4  # Correct count calculation
                ).reshape(-1, 3)

                vertices.extend(vertices_array)

        return np.array(vertices)


    def load_model(self) -> np.ndarray:
        """Load and process the 3D model."""
        try:
            json_data, bin_data = self._parse_glb(self.gltf_path)
            vertices = self._extract_vertices(json_data, bin_data)
            print(f"Successfully extracted {len(vertices):,} vertices")
            return vertices
        except Exception as e:
            print(f"Error in load_model: {e}")  # More concise error message
            raise


    def load_weather_data(self) -> pd.DataFrame:
        """Load and process EPW weather data, handling potential errors."""
        try:
            epw = EPW(self.epw_path.as_posix())  # Use as_posix() for cross-platform compatibility
            data = {
                'air_temp': epw.dry_bulb_temperature.values,
                'rel_humidity': epw.relative_humidity.values,
                'wind_speed': epw.wind_speed.values,
                'mean_rad_temp': epw.dry_bulb_temperature.values  # Simplification, consider a better MRT proxy
            }
            return pd.DataFrame(data)

        except Exception as e:
            print(f"Error loading weather data: {e}")
            raise

    def calculate_utci(self, points: np.ndarray, weather_data: pd.DataFrame, hour_index: int = 12) -> np.ndarray:
        """Calculate UTCI, handling edge cases and providing progress updates."""
        print(f"\nCalculating UTCI for hour {hour_index}...")
        if not 0 <= hour_index < len(weather_data):
            raise ValueError(f"Invalid hour_index: {hour_index}. Must be within 0-{len(weather_data) -1}")

        weather = weather_data.iloc[hour_index]
        total_points = len(points)

        if total_points == 0:  # Handle empty points array
            print("Warning: No points provided for UTCI calculation.")
            return np.array([])

        utci_values = np.zeros(total_points)
        batch_size = 100000
        num_batches = (total_points + batch_size - 1) // batch_size

        start_time = time.time()
        last_update = start_time
        points_processed = 0

        print(f"Processing {total_points:,} points...")

        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, total_points)
            current_batch_size = batch_end - batch_start

            ta = np.full(current_batch_size, weather['air_temp'])
            tr = np.full(current_batch_size, weather['mean_rad_temp'])
            vel = np.full(current_batch_size, weather['wind_speed'])
            rh = np.full(current_batch_size, weather['rel_humidity'])


            # Handle edge case of zero wind speed
            vel = np.maximum(vel, 0.5)  # Replace 0 with 0.5 m/s
            
            utci_values[batch_start:batch_end] = universal_thermal_climate_index_np(ta, tr, vel, rh)

            points_processed = batch_end
            current_time = time.time()

            if current_time - last_update >= 5:
                self._print_progress(start_time, current_time, points_processed, total_points)
                last_update = current_time

        total_time = time.time() - start_time
        print(f"\nUTCI calculation complete for hour {hour_index}")
        print(f"Total processing time: {timedelta(seconds=int(total_time))}")
        print(f"Average processing speed: {total_points/total_time:,.0f} points/second")

        return utci_values

    def _print_progress(self, start_time: float, current_time: float, points_processed: int, total_points: int):
        """Prints the progress of the UTCI calculation."""
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
        """Create visualization, handling potential errors."""
        print("Creating visualizations...")

        if len(points) == 0 or len(utci_values) == 0:
            print("Warning: No data to visualize.")
            return

        if len(points) != len(utci_values):
            raise ValueError("Mismatch between points and UTCI values length.")

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
            scatter = ax1.scatter(points[:, 0], points[:, 1], c=utci_values, cmap='RdYlBu_r', s=20)
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
            plt.close()  # Close the figure in case of an error
            raise

        self._save_results(points, utci_values, hour_index)


    def _save_results(self, points: np.ndarray, utci_values: np.ndarray, hour_index: int):
        """Saves the results to a JSON file."""
        results_dict = {
            'metadata': {
                'hour_index': hour_index,
                'num_points': len(points),
                'min_utci': float(np.min(utci_values)),
                'max_utci': float(np.max(utci_values)),
                'mean_utci': float(np.mean(utci_values))
            },
            'point_data': [{
                'coordinates': point.tolist(),
                'utci': float(utci)
            } for point, utci in zip(points, utci_values)]
        }

        filepath = self.output_dir / f'utci_results_hour_{hour_index}.json'
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to: {filepath}")


    def run_analysis(self, hour_index: int = 12) -> Dict:
        """Run complete UTCI analysis."""
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

def main():
    # Use argparse for better command-line argument handling
    import argparse
    parser = argparse.ArgumentParser(description="Calculate UTCI from GLTF and EPW files.")
    parser.add_argument("gltf_path", type=str, help="Path to the GLTF file.")
    parser.add_argument("epw_path", type=str, help="Path to the EPW file.")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Output directory (default: output).")
    parser.add_argument("-H", "--hour", type=int, default=12, help="Hour of the day for UTCI calculation (0-23, default: 12).")
    args = parser.parse_args()
    
    calculator = UTCICalculator(args.gltf_path, args.epw_path, args.output_dir)
    calculator.run_analysis(hour_index=args.hour)

if __name__ == "__main__":
    main()