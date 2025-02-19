import os
from pathlib import Path
import numpy as np
from ladybug.epw import EPW
from ladybug_comfort.utci import universal_thermal_climate_index
import pandas as pd
import matplotlib.pyplot as plt
import json
import struct

class UTCICalculator:
    def __init__(self, gltf_path, epw_path, output_dir='output'):
        self.gltf_path = Path(gltf_path)
        self.epw_path = Path(epw_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.validate_paths()
        
    def validate_paths(self):
        if not self.gltf_path.exists():
            raise FileNotFoundError(f"GLTF file not found: {self.gltf_path}")
        if not self.epw_path.exists():
            raise FileNotFoundError(f"EPW file not found: {self.epw_path}")

    def parse_glb(self, filepath):
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

    def extract_vertices(self, json_data, bin_data):
        """Extract vertices from parsed GLB data."""
        print("Extracting vertices from GLB data...")
        vertices = []
        
        for mesh in json_data.get('meshes', []):
            for primitive in mesh.get('primitives', []):
                position_accessor = json_data['accessors'][primitive['attributes']['POSITION']]
                buffer_view = json_data['bufferViews'][position_accessor['bufferView']]
                
                start = buffer_view.get('byteOffset', 0)
                length = buffer_view['byteLength']
                stride = buffer_view.get('byteStride', 12)
                
                vertex_data = bin_data[start:start + length]
                vertex_count = length // stride
                
                vertices_array = np.frombuffer(
                    vertex_data, 
                    dtype=np.float32,
                    count=vertex_count * 3
                ).reshape(-1, 3)
                
                vertices.extend(vertices_array)
                
        return np.array(vertices)

    def load_model(self):
        """Load and process the 3D model."""
        try:
            json_data, bin_data = self.parse_glb(self.gltf_path)
            vertices = self.extract_vertices(json_data, bin_data)
            print(f"Successfully extracted {len(vertices)} vertices")
            return vertices
        except Exception as e:
            print(f"Error in load_model: {str(e)}")
            raise

    def load_weather_data(self):
        """Load and process EPW weather data."""
        try:
            epw = EPW(str(self.epw_path))
            
            # Create hourly data DataFrame
            data = {
                'air_temp': epw.dry_bulb_temperature,  # These are now directly float arrays
                'rel_humidity': epw.relative_humidity,
                'wind_speed': epw.wind_speed,
                'mean_rad_temp': epw.dry_bulb_temperature  # Using air temp as MRT for simplification
            }
            
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error loading weather data: {str(e)}")
            raise

    def calculate_utci(self, points, weather_data, hour_index=12):
        """Calculate UTCI for given points at specified hour."""
        import time
        from datetime import datetime, timedelta
        
        print(f"\nCalculating UTCI for hour {hour_index}...")
        weather = weather_data.iloc[hour_index]
        total_points = len(points)
        
        # Initialize array for results
        utci_values = np.zeros(total_points)
        
        # Process in batches to show progress
        batch_size = 100000  # Adjust based on your needs
        num_batches = (total_points + batch_size - 1) // batch_size
        
        start_time = time.time()
        last_update = start_time
        points_processed = 0
        
        print(f"Processing {total_points:,} points...")
        
        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = min((batch + 1) * batch_size, total_points)
            
            # Calculate UTCI for this batch
            utci_values[batch_start:batch_end] = np.array([
                universal_thermal_climate_index(
                    float(weather['air_temp']),
                    float(weather['mean_rad_temp']),
                    float(weather['rel_humidity']),
                    float(weather['wind_speed'])
                )
                for _ in range(batch_start, batch_end)
            ])
            
            points_processed = batch_end
            current_time = time.time()
            
            # Update progress every 5 seconds
            if current_time - last_update >= 5:
                elapsed_time = current_time - start_time
                progress = points_processed / total_points
                estimated_total_time = elapsed_time / progress if progress > 0 else 0
                remaining_time = estimated_total_time - elapsed_time
                
                print(f"Progress: {progress*100:.1f}% ({points_processed:,}/{total_points:,} points)")
                print(f"Elapsed: {timedelta(seconds=int(elapsed_time))}")
                print(f"Estimated remaining: {timedelta(seconds=int(remaining_time))}")
                print(f"Processing speed: {points_processed/elapsed_time:,.0f} points/second")
                print()
                
                last_update = current_time
        
        total_time = time.time() - start_time
        print(f"\nUTCI calculation complete for hour {hour_index}")
        print(f"Total processing time: {timedelta(seconds=int(total_time))}")
        print(f"Average processing speed: {total_points/total_time:,.0f} points/second")
        
        return utci_values

    def create_visualization(self, points, utci_values, hour_index):
        """Create visualization of UTCI results."""
        print("Creating visualizations...")
        
        # UTCI thermal stress categories
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
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), 
                                      gridspec_kw={'width_ratios': [3, 1]})
        
        # Scatter plot
        scatter = ax1.scatter(points[:, 0], points[:, 1], c=utci_values,
                            cmap='RdYlBu_r', s=20)
        ax1.set_title(f'UTCI Distribution (Hour {hour_index})')
        ax1.set_xlabel('X coordinate (m)')
        ax1.set_ylabel('Y coordinate (m)')
        plt.colorbar(scatter, ax=ax1, label='UTCI (°C)')
        
        # Legend/categories plot
        ax2.set_title('UTCI Thermal Stress Categories')
        for i, (temp, label, color) in enumerate(categories):
            ax2.barh(i, 1, color=color)
            ax2.text(1.1, i, f'{label}\n(>{temp}°C)', va='center')
        
        ax2.set_ylim(-0.5, len(categories)-0.5)
        ax2.set_xlim(0, 1)
        ax2.axis('off')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.output_dir / f'utci_visualization_hour_{hour_index}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
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
        
        with open(self.output_dir / f'utci_results_hour_{hour_index}.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to output directory for hour {hour_index}")

    def run_analysis(self, hour_index=12):
        """Run complete UTCI analysis."""
        try:
            # Load model and weather data
            points = self.load_model()
            weather_data = self.load_weather_data()
            
            # Calculate UTCI
            utci_values = self.calculate_utci(points, weather_data, hour_index)
            
            # Create visualizations
            self.create_visualization(points, utci_values, hour_index)
            
            # Print summary
            print("\nUTCI Analysis Complete!")
            print(f"Number of points analyzed: {len(points)}")
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
            print(f"Error during analysis: {str(e)}")
            raise

def main():
    # File paths
    current_dir = Path.cwd()
    gltf_path = current_dir / "data" / "rec_model.gltf"  # Using .gltf extension
    epw_path = current_dir / "data" / "ISR_D_Beer.Sheva.401900_TMYx" / "ISR_D_Beer.Sheva.401900_TMYx.epw"
    output_dir = current_dir / "output"
    
    print("\nChecking file paths...")
    print(f"GLTF file path: {gltf_path}")
    print(f"EPW file path: {epw_path}")
    print(f"Output directory: {output_dir}")
    
    # Create calculator and run analysis
    calculator = UTCICalculator(gltf_path, epw_path, output_dir)
    
    # Analyze for multiple hours (e.g., daytime hours)
    for hour in range(8, 18):  # 8 AM to 5 PM
        print(f"\nAnalyzing hour {hour}:00")
        results = calculator.run_analysis(hour_index=hour)

if __name__ == "__main__":
    main()