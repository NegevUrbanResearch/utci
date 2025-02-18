import os
from pathlib import Path
import numpy as np
from ladybug.epw import EPW
from ladybug_comfort.utci import universal_thermal_climate_index
from pygltflib import GLTF2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
        print("Parsing GLB file manually...")
        with open(filepath, 'rb') as f:
            # Read GLB header
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
                
                # Extract vertex data from binary chunk
                start = buffer_view.get('byteOffset', 0)
                length = buffer_view['byteLength']
                stride = buffer_view.get('byteStride', 12)  # 3 floats * 4 bytes
                
                vertex_data = bin_data[start:start + length]
                vertex_count = length // stride
                
                # Convert to numpy array
                vertices_array = np.frombuffer(
                    vertex_data, 
                    dtype=np.float32,
                    count=vertex_count * 3
                ).reshape(-1, 3)
                
                vertices.extend(vertices_array)
                
        return np.array(vertices)

    def load_model(self):
        """Load and process the 3D model."""
        print("Loading 3D model...")
        try:
            # Parse GLB file
            json_data, bin_data = self.parse_glb(self.gltf_path)
            
            # Extract vertices
            vertices = self.extract_vertices(json_data, bin_data)
            
            print(f"Successfully extracted {len(vertices)} vertices")
            return vertices
            
        except Exception as e:
            print(f"Error in load_model: {str(e)}")
            raise

    def load_weather_data(self):
        """Load and process EPW weather data."""
        print("Loading EPW data...")
        print(f"Reading from: {self.epw_path}")
        
        # Check first few lines of the EPW file
        with open(self.epw_path, 'r') as f:
            print("First line of EPW file:")
            print(f.readline().strip())
            
        try:
            epw = EPW(str(self.epw_path))
            data = {
                'air_temp': [h.dry_bulb_temperature for h in epw.dry_bulb_temperature],
                'rel_humidity': [h.relative_humidity for h in epw.relative_humidity],
                'wind_speed': [h.wind_speed for h in epw.wind_speed],
                'mean_rad_temp': [h.dry_bulb_temperature for h in epw.dry_bulb_temperature]
            }
            print("Weather data loaded successfully")
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error loading weather data: {str(e)}")
            raise

    def calculate_utci(self, points, weather_data, hour_index=12):
        """Calculate UTCI for given points at specified hour."""
        print(f"Calculating UTCI for hour {hour_index}...")
        weather = weather_data.iloc[hour_index]
        
        utci_values = []
        for point in points:
            utci = universal_thermal_climate_index(
                weather['air_temp'],
                weather['mean_rad_temp'],
                weather['rel_humidity'],
                weather['wind_speed']
            )
            utci_values.append(utci)
        
        return np.array(utci_values)

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
        plt.savefig(self.output_dir / 'utci_visualization.jpg', dpi=300, bbox_inches='tight')
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
        
        with open(self.output_dir / 'utci_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print("Visualizations saved to output directory")

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
            print(f"\nResults saved to: {self.output_dir}")
            
            return {
                'points': points,
                'utci_values': utci_values,
                'summary': {
                    'min_utci': np.min(utci_values),
                    'max_utci': np.max(utci_values),
                    'mean_utci': np.mean(utci_values)
                }
            }
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

def main():
    # File paths and debug info
    print("\nChecking file paths...")
    
    current_dir = Path.cwd()
    gltf_path = current_dir / "data" / "rec_model.gltf"
    epw_path = current_dir / "data" / "ISR_D_Beer.Sheva.401900_TMYx/ISR_D_Beer.Sheva.401900_TMYx.epw"
    output_dir = current_dir / "output"
    
    print(f"Looking for EPW file at: {epw_path}")
    print(f"File exists: {epw_path.exists()}")
    print(f"File size: {epw_path.stat().st_size if epw_path.exists() else 'N/A'} bytes\n")
    
    # Create calculator and run analysis
    calculator = UTCICalculator(gltf_path, epw_path, output_dir)
    results = calculator.run_analysis(hour_index=12)  # Analyze for noon

if __name__ == "__main__":
    main()