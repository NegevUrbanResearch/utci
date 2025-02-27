import json
from pathlib import Path
import numpy as np
import struct

def analyze_gltf_or_glb(file_path):
    """Analyze a GLTF or GLB file and print its structure."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return
    
    print(f"Analyzing file: {file_path}")
    
    # First try to detect if it's actually a GLB file regardless of extension
    try:
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            if magic == b'glTF':
                print("\nDetected binary GLB format despite the file extension")
                # Parse GLB structure
                f.seek(0)
                magic = f.read(4)
                version, length = struct.unpack('<II', f.read(8))
                print(f"GLB version: {version}")
                print(f"Total file size: {length} bytes")
                
                # Read JSON chunk
                chunk_length, chunk_type = struct.unpack('<II', f.read(8))
                if chunk_type == 0x4E4F534A:  # 'JSON' in ASCII
                    print(f"JSON chunk size: {chunk_length} bytes")
                    json_data = json.loads(f.read(chunk_length))
                    print_gltf_structure(json_data)
                    
                    # Check for binary chunk
                    if f.tell() < length:
                        try:
                            chunk_length, chunk_type = struct.unpack('<II', f.read(8))
                            if chunk_type == 0x004E4942:  # 'BIN\0' in ASCII
                                print(f"\nBinary (BIN) chunk size: {chunk_length} bytes")
                        except:
                            print("\nNo valid binary chunk found")
                else:
                    print("Error: First chunk is not a JSON chunk")
                return
    except Exception as e:
        print(f"Error checking for GLB format: {str(e)}")
    
    # If we're here, try parsing as JSON GLTF
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("\nDetected JSON GLTF format")
            print_gltf_structure(data)
    except Exception as e:
        print(f"Error analyzing as GLTF file: {str(e)}")
        print("\nThis appears to be neither a valid GLTF nor a valid GLB file.")

def print_gltf_structure(data):
    """Print the structure of a GLTF JSON object."""
    print(f"Asset version: {data.get('asset', {}).get('version', 'Not specified')}")
    
    # Scene info
    scenes = data.get('scenes', [])
    print(f"Scenes: {len(scenes)}")
    
    # Node info
    nodes = data.get('nodes', [])
    print(f"Nodes: {len(nodes)}")
    
    # Mesh info
    meshes = data.get('meshes', [])
    print(f"Meshes: {len(meshes)}")
    
    for i, mesh in enumerate(meshes):
        print(f"\nMesh {i}:")
        primitives = mesh.get('primitives', [])
        print(f"  Primitives: {len(primitives)}")
        
        for j, prim in enumerate(primitives[:3]):  # Show first 3 primitives
            print(f"  Primitive {j}:")
            print(f"    Mode: {prim.get('mode', 4)}")  # 4 is TRIANGLES
            print(f"    Attributes: {prim.get('attributes', {})}")
            print(f"    Has indices: {'indices' in prim}")
            
        if len(primitives) > 3:
            print(f"  ... {len(primitives) - 3} more primitives ...")
    
    # Material info
    materials = data.get('materials', [])
    print(f"\nMaterials: {len(materials)}")
    
    # Accessor info
    accessors = data.get('accessors', [])
    print(f"\nAccessors: {len(accessors)}")
    
    # Buffer views
    buffer_views = data.get('bufferViews', [])
    print(f"BufferViews: {len(buffer_views)}")
    
    # Buffers
    buffers = data.get('buffers', [])
    print(f"Buffers: {len(buffers)}")
    for i, buffer in enumerate(buffers):
        print(f"  Buffer {i}:")
        print(f"    Size: {buffer.get('byteLength', 0)} bytes")
        if 'uri' in buffer:
            print(f"    URI: {buffer['uri']}")
        else:
            print(f"    Embedded in binary chunk")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_gltf_or_glb(sys.argv[1])
    else:
        print("Usage: python gltf_analyzer.py path/to/file.gltf_or_glb") 