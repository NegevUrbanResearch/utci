import json
from pathlib import Path
import numpy as np
import struct
import base64
from typing import Dict, Tuple

def examine_gltf(filepath: Path):
    """Main function to examine a GLTF/GLB file."""
    print(f"Examining file: {filepath}")
    
    try:
        json_data, bin_data = _load_gltf_or_glb(filepath)
        
        # Basic file structure analysis
        print("\n=== File Structure ===")
        for key, value in json_data.items():
            if isinstance(value, list):
                print(f"{key}: {len(value)} items")
            else:
                print(f"{key}: {value}")
        
        # Analyze meshes
        if "meshes" in json_data:
            print("\n=== Mesh Analysis ===")
            for mesh_idx, mesh in enumerate(json_data['meshes']):
                print(f"\nMesh {mesh_idx}:")
                print(f"Number of primitives: {len(mesh['primitives'])}")
                
                for prim_idx, primitive in enumerate(mesh['primitives']):
                    print(f"\nPrimitive {prim_idx}:")
                    print(f"Mode: {primitive.get('mode', 4)}")
                    print("Attributes:", primitive['attributes'])
                    if 'indices' in primitive:
                        print(f"Has indices: Yes")
                    
                    # Analyze vertex data
                    if 'POSITION' in primitive['attributes']:
                        pos_accessor = json_data['accessors'][primitive['attributes']['POSITION']]
                        print(f"Vertex count: {pos_accessor['count']}")
                        if 'min' in pos_accessor:
                            print(f"Bounds min: {pos_accessor['min']}")
                            print(f"Bounds max: {pos_accessor['max']}")
        
        # Run geometry analysis
        analyze_geometry_type(json_data, bin_data)
        
        # If multiple meshes exist, analyze the line mesh
        if len(json_data['meshes']) > 1:
            analyze_line_mesh(json_data, bin_data, 1)
            
        # If multiple meshes exist, clean and save the GLB
        if len(json_data['meshes']) > 1:
            # Force .glb extension if input is GLB format
            with open(filepath, 'rb') as f:
                magic = f.read(4)
                f.seek(0)
                is_glb = magic == b'glTF'
            
            if is_glb:
                output_path = filepath.parent / f"{filepath.stem}_cleaned.glb"
            else:
                output_path = filepath.parent / f"{filepath.stem}_cleaned{filepath.suffix}"
            
            clean_and_save_glb(filepath, output_path)
            
    except Exception as e:
        print(f"\nDetailed error information:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise

def _load_gltf_or_glb(filepath: Path) -> Tuple[Dict, bytes]:
    """Loads either a GLB or a GLTF file, detecting format by content."""
    
    try:
        # First try to read as binary
        with open(filepath, 'rb') as f:
            # Peek at the first 4 bytes to detect GLB
            magic = f.read(4)
            f.seek(0)  # Reset to start of file
            if magic == b'glTF':
                print("Detected GLB format by content")
                # Process as GLB
                header_data = f.read(12)
                magic, version, length = struct.unpack('<4sII', header_data)
                print(f"GLB version: {version}, total length: {length} bytes")
                
                # Continue with GLB parsing...
                # Read JSON chunk header
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    raise ValueError("Incomplete JSON chunk header")
                
                chunk_length, chunk_type = struct.unpack('<II', chunk_header)
                if chunk_type != 0x4E4F534A:  # JSON
                    raise ValueError(f"Expected JSON chunk, got: 0x{chunk_type:08X}")
                print(f"JSON chunk length: {chunk_length} bytes")
                
                # Read JSON data
                json_data = f.read(chunk_length)
                if len(json_data) < chunk_length:
                    raise ValueError(f"Incomplete JSON chunk: expected {chunk_length}, got {len(json_data)}")
                
                try:
                    json_data = json.loads(json_data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON data: {e}")
                
                # Read binary chunk
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    raise ValueError("Missing or incomplete BIN chunk header")
                    
                chunk_length, chunk_type = struct.unpack('<II', chunk_header)
                if chunk_type != 0x004E4942:  # BIN
                    raise ValueError(f"Expected BIN chunk, got: 0x{chunk_type:08X}")
                print(f"BIN chunk length: {chunk_length} bytes")
                
                bin_data = f.read(chunk_length)
                if len(bin_data) < chunk_length:
                    raise ValueError(f"Incomplete BIN chunk: expected {chunk_length}, got {len(bin_data)}")
                
                return json_data, bin_data
            
            else:
                # Try to process as GLTF
                print("Attempting to read as GLTF format")
                f.seek(0)
                try:
                    # Try to decode as UTF-8 text
                    content = f.read().decode('utf-8')
                    json_data = json.loads(content)
                    
                    if 'buffers' not in json_data or len(json_data['buffers']) == 0:
                        raise ValueError("No buffers found in GLTF file")

                    buffer_uri = json_data['buffers'][0]['uri']
                    print(f"Buffer URI: {buffer_uri}")

                    if buffer_uri.startswith('data:'):
                        data_header, data_encoded = buffer_uri.split(',', 1)
                        bin_data = base64.b64decode(data_encoded)
                    else:
                        bin_filepath = filepath.parent / buffer_uri
                        if not bin_filepath.exists():
                            raise FileNotFoundError(f"Binary file not found: {bin_filepath}")
                        with open(bin_filepath, 'rb') as bf:
                            bin_data = bf.read()
                    return json_data, bin_data
                    
                except UnicodeDecodeError:
                    # If we get here, it's neither a valid GLB nor GLTF
                    print("\nFile appears to be corrupted:")
                    # Dump first 32 bytes for inspection
                    f.seek(0)
                    header = f.read(32)
                    print("First 32 bytes (hex):", " ".join(f"{b:02X}" for b in header))
                    print("First 32 bytes (ASCII):", "".join(chr(b) if 32 <= b <= 126 else '.' for b in header))
                    raise ValueError("File is neither a valid GLB nor GLTF format")

    except Exception as e:
        print(f"\nDetailed error information:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise

def validate_primitive_data(json_data: Dict, bin_data: bytes, mesh_index: int, primitive_index: int):
    """Validates and prints details about a specific primitive's point cloud data."""
    primitive = json_data['meshes'][mesh_index]['primitives'][primitive_index]
    
    # Check primitive mode (0 = POINTS, 1 = LINES, 4 = TRIANGLES)
    mode = primitive.get('mode', 4)
    mode_names = {0: 'POINTS', 1: 'LINES', 4: 'TRIANGLES'}
    print(f"\nPrimitive {primitive_index} mode: {mode_names.get(mode, 'UNKNOWN')}")
    
    if mode != 0:
        print("Warning: Expected POINTS mode (0) for point cloud data")
    
    # Get position accessor
    pos_accessor = json_data['accessors'][primitive['attributes']['POSITION']]
    pos_buffer_view = json_data['bufferViews'][pos_accessor['bufferView']]
    
    # Calculate point positions
    pos_start = pos_buffer_view['byteOffset'] if 'byteOffset' in pos_buffer_view else 0
    pos_stride = pos_buffer_view.get('byteStride', 12)  # 3 floats * 4 bytes
    pos_count = pos_accessor['count']
    
    # Print point cloud statistics
    print(f"\nPoint cloud statistics:")
    print(f"Total points: {pos_count}")
    print(f"Stride: {pos_stride} bytes")
    if 'min' in pos_accessor and 'max' in pos_accessor:
        print(f"Bounding box:")
        print(f"  Min: {pos_accessor['min']}")
        print(f"  Max: {pos_accessor['max']}")
    
    # Print sample of points
    print(f"\nSample points (first 5):")
    points = []
    for i in range(min(5, pos_count)):
        offset = pos_start + i * pos_stride
        point = struct.unpack_from('<fff', bin_data, offset)
        points.append(point)
        print(f"Point {i}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")
    
    # Check for additional attributes (color, intensity, etc.)
    print("\nAdditional attributes:")
    for attr, accessor_idx in primitive['attributes'].items():
        if attr != 'POSITION':
            accessor = json_data['accessors'][accessor_idx]
            print(f"  {attr}: count={accessor['count']}, type={accessor.get('type', 'unknown')}")
    
    return points

def analyze_geometry_type(json_data: Dict, bin_data: bytes):
    """Analyzes the geometry to determine if it's likely a point cloud, mesh, or line structure."""
    
    print("\n=== Geometry Analysis ===")
    
    for mesh_idx, mesh in enumerate(json_data['meshes']):
        print(f"\nMesh {mesh_idx}:")
        print(f"Number of primitives: {len(mesh['primitives'])}")
        
        primitive_modes = {}
        vertex_counts = set()
        has_indices = set()
        has_normals = False
        has_uvs = False
        
        for prim_idx, primitive in enumerate(mesh['primitives']):
            # Track primitive modes
            mode = primitive.get('mode', 4)
            primitive_modes[mode] = primitive_modes.get(mode, 0) + 1
            
            # Get position data info
            pos_accessor = json_data['accessors'][primitive['attributes']['POSITION']]
            vertex_counts.add(pos_accessor['count'])
            
            # Check for indices
            has_indices.add('indices' in primitive)
            
            # Check for other attributes
            has_normals = has_normals or 'NORMAL' in primitive['attributes']
            has_uvs = has_uvs or 'TEXCOORD_0' in primitive['attributes']
        
        # Print analysis
        print("\nGeometry characteristics:")
        mode_names = {0: 'POINTS', 1: 'LINES', 4: 'TRIANGLES'}
        for mode, count in primitive_modes.items():
            print(f"- {mode_names.get(mode, f'Mode {mode}')} primitives: {count}")
        
        print(f"Vertex counts per primitive: {list(vertex_counts)}")
        print(f"Has indices: {all(has_indices)}")
        print(f"Has normals: {has_normals}")
        print(f"Has UVs: {has_uvs}")
        
        # Make an educated guess about the geometry type
        print("\nGeometry type assessment:")
        if 0 in primitive_modes:  # POINTS mode
            print("✓ This appears to be a point cloud")
        elif 1 in primitive_modes and not has_normals and not has_uvs:  # LINES mode without typical mesh attributes
            print("✓ This appears to be a line/wireframe structure")
        elif 4 in primitive_modes and has_normals:  # TRIANGLES mode with normals
            print("✓ This appears to be a regular mesh")
        else:
            print("? Unable to definitively determine geometry type")
            
        # Additional warnings
        if len(vertex_counts) == 1 and len(mesh['primitives']) > 1:
            print("\nWarning: Multiple primitives using identical vertex counts - possible data duplication")
        
        if not has_normals and 4 in primitive_modes:
            print("\nWarning: Mesh has triangles but no normals - might be incomplete/corrupted")

def clean_and_save_glb(input_path: Path, output_path: Path):
    """Removes the line mesh and saves a new GLB with only the triangulated mesh."""
    print(f"\nCleaning GLB file...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    with open(input_path, 'rb') as f:
        json_data, bin_data = _load_gltf_or_glb(input_path)
    
    # Store original mesh count
    original_mesh_count = len(json_data['meshes'])
    
    # Keep only the first mesh (the triangulated one)
    json_data['meshes'] = [json_data['meshes'][0]]
    
    # Update any nodes that referenced the second mesh
    if 'nodes' in json_data:
        for node in json_data['nodes']:
            if 'mesh' in node and node['mesh'] > 0:
                # Remove mesh reference or update to 0
                node.pop('mesh', None)
    
    # Create new GLB file
    json_bytes = json.dumps(json_data).encode('utf-8')
    # Pad to 4-byte boundary
    json_length = len(json_bytes)
    padding = (4 - (json_length % 4)) % 4
    json_bytes += b' ' * padding
    
    # Create GLB header
    header = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(json_bytes) + 8 + len(bin_data))
    json_header = struct.pack('<II', len(json_bytes), 0x4E4F534A)  # JSON chunk
    bin_header = struct.pack('<II', len(bin_data), 0x004E4942)     # BIN chunk
    
    # Write new GLB file
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(json_header)
        f.write(json_bytes)
        f.write(bin_header)
        f.write(bin_data)
    
    print(f"\nCleaning complete:")
    print(f"- Original mesh count: {original_mesh_count}")
    print(f"- New mesh count: 1")
    print(f"- Removed all line geometry")
    print(f"- Cleaned file saved to: {output_path}")

def analyze_line_mesh(json_data: Dict, bin_data: bytes, mesh_index: int):
    """Analyzes the line mesh to determine if it's a base map or reference drawing."""
    
    print(f"\n=== Analyzing Potential Base Map (Mesh {mesh_index}) ===")
    mesh = json_data['meshes'][mesh_index]
    
    # Get the first primitive to analyze vertex data
    primitive = mesh['primitives'][0]
    pos_accessor = json_data['accessors'][primitive['attributes']['POSITION']]
    pos_buffer_view = json_data['bufferViews'][pos_accessor['bufferView']]
    
    # Calculate bounds
    min_coords = pos_accessor['min']
    max_coords = pos_accessor['max']
    print(f"Bounding Box:")
    print(f"  X: {min_coords[0]:.3f} to {max_coords[0]:.3f}")
    print(f"  Y: {min_coords[1]:.3f} to {max_coords[1]:.3f}")
    print(f"  Z: {min_coords[2]:.3f} to {max_coords[2]:.3f}")
    
    # Check if it's planar (all Z values same)
    is_planar = abs(min_coords[2] - max_coords[2]) < 0.0001
    print(f"\nIs Planar: {is_planar}")
    if is_planar:
        print(f"Z-plane position: {min_coords[2]:.3f}")
    
    # Sample some vertices to check for patterns
    pos_start = pos_buffer_view['byteOffset'] if 'byteOffset' in pos_buffer_view else 0
    pos_stride = pos_buffer_view.get('byteStride', 12)
    
    print("\nSampling line segments:")
    for i in range(min(3, len(mesh['primitives']))):
        prim = mesh['primitives'][i]
        idx_accessor = json_data['accessors'][prim['indices']]
        idx_buffer_view = json_data['bufferViews'][idx_accessor['bufferView']]
        idx_start = idx_buffer_view['byteOffset'] if 'byteOffset' in idx_buffer_view else 0
        
        # Get first two indices to show a line segment
        indices = struct.unpack_from('<HH', bin_data, idx_start)
        points = []
        for idx in indices:
            offset = pos_start + idx * pos_stride
            point = struct.unpack_from('<fff', bin_data, offset)
            points.append(point)
        
        print(f"\nLine {i}:")
        print(f"  Start: ({points[0][0]:.3f}, {points[0][1]:.3f}, {points[0][2]:.3f})")
        print(f"  End:   ({points[1][0]:.3f}, {points[1][1]:.3f}, {points[1][2]:.3f})")
    
    # Check if coordinates align with main mesh
    if len(json_data['meshes']) > 1:
        main_mesh = json_data['meshes'][0]
        main_pos = json_data['accessors'][main_mesh['primitives'][0]['attributes']['POSITION']]
        print("\nAlignment with main mesh:")
        print(f"Main mesh bounds: {main_pos['min']} to {main_pos['max']}")
        
        # Check if bounds overlap in XY plane
        x_overlap = (min_coords[0] <= main_pos['max'][0] and max_coords[0] >= main_pos['min'][0])
        y_overlap = (min_coords[1] <= main_pos['max'][1] and max_coords[1] >= main_pos['min'][1])
        print(f"Overlaps with main mesh in XY plane: {x_overlap and y_overlap}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        filepath = Path("data/rec_model.gltf")
    
    try:
        examine_gltf(filepath)
    except Exception as e:
        print(f"Error loading GLTF/GLB: {str(e)}")