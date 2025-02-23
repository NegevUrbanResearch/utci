import json
from pathlib import Path
import numpy as np
import struct
import base64
from typing import Dict, Tuple

def examine_gltf(filepath: str):
    """Loads and examines a GLTF or GLB file, printing key information."""

    filepath = Path(filepath)
    print(f"Examining file: {filepath}")

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return

    try:
        json_data, bin_data = _load_gltf_or_glb(filepath) #use the parsing method
    except Exception as e:
        print(f"Error loading GLTF/GLB: {e}")
        return

    print("\n--- GLTF Structure ---")
    for key, value in json_data.items():
        if key != "buffers":  # Print buffers last (potentially large)
            print(f"{key}: {type(value)}")
            if isinstance(value, list):
                print(f"  Number of items: {len(value)}")
                if len(value) > 0 and isinstance(value[0], dict):
                    print(f"    First item keys: {list(value[0].keys())}")
                else:
                    print(f"    First 5 values: {value[:5]}")  # Limit output
            elif isinstance(value, dict):
                print(f"    Keys: {list(value.keys())}")
            else:
                print(f"    Value: {value}")


    if bin_data:
        print("\n--- Binary Data ---")
        print(f"  Length: {len(bin_data)} bytes")



    # Accessors examination (show componentType mapping)
    if "accessors" in json_data:
        print("\n--- Accessors ---")
        component_types = {
            5120: "BYTE",
            5121: "UNSIGNED_BYTE",
            5122: "SHORT",
            5123: "UNSIGNED_SHORT",
            5125: "UNSIGNED_INT",
            5126: "FLOAT",
        }
        for i, accessor in enumerate(json_data['accessors']):
            print(f"Accessor {i}:")
            for key, value in accessor.items():
                if key == "componentType":
                    print(f"  {key}: {value} ({component_types.get(value, 'UNKNOWN')})")
                else:
                    print(f"  {key}: {value}")


def _load_gltf_or_glb(filepath: Path) -> Tuple[Dict, bytes]:
    """Loads either a GLB or a GLTF file, handling external buffers."""

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



if __name__ == '__main__':
    gltf_file_path = "test_main_data/rec_model.glb"  # Now with .glb extension
    examine_gltf(gltf_file_path)