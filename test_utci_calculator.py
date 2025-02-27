# run with:
# pytest test_utci_calculator.py -v
import pytest
from pathlib import Path
import numpy as np
import utci_calculator  # Import the module
import os
import tempfile
import shutil
import struct
import json
import subprocess  # Add import for subprocess

# Helper function to create a temporary directory
@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_epw():
    """Uses actual EPW file for testing. You can switch to the below fixture to create a sample EPW file for testing."""
    epw_path = Path(__file__).parent / "data" / "ISR_D_Beer.Sheva.401900_TMYx" / "ISR_D_Beer.Sheva.401900_TMYx.epw"
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW file not found at: {epw_path}")
    return str(epw_path)
'''
@pytest.fixture
def sample_epw(temp_dir):
    # Create a sample (minimal) EPW file with 8760 lines
    epw_path = os.path.join(temp_dir, "test.epw")
    with open(epw_path, "w") as f:
        f.write("LOCATION,Unknown Location,,,,,,,0,0,0,0\n") #Added dummy location
        f.write("DESIGN CONDITIONS,0\n")
        f.write("TYPICAL/EXTREME PERIODS,0\n")
        f.write("GROUND TEMPERATURES,0\n")
        f.write("HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n")
        f.write("COMMENTS 1\n")
        f.write("COMMENTS 2\n")
        f.write("DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31\n")
        # Add 8760 lines of dummy data, setting direct and diffuse radiation
        for i in range(8760):
            if 7 < i%24 < 19: # Simulate daytime hours for testing, avoiding night hours.

                f.write(f"2024,1,1,{i%24 + 1},0,10.0,90.0,0,0,100,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n") #Example line for hour with sun.
            else:
                f.write(f"2024,1,1,{i%24 + 1},0,10.0,90.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n")

    return epw_path

@pytest.fixture
def sample_glb(temp_dir):
    # Correct, minimal GLB content.
    glb_path = os.path.join(temp_dir, "test.glb")
    # Create a minimal GLB. This GLB contains a single triangle.
    gltf_data = {
      "asset": {"version": "2.0"},
      "scene": 0,
      "scenes": [{"nodes": [0]}],
      "nodes": [{"mesh": 0}],
      "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 4}]}],
      "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "max":[1.0, 1.0, 0.0], "min": [-1.0, -1.0, 0.0]}],
      "bufferViews": [{"buffer": 0, "byteLength": 36, "target": 34962}],
      "buffers": [{"byteLength": 36}]
    }
    binary_data = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [-1.0,  1.0, 0.0],
    ], dtype=np.float32).tobytes()

    json_string = json.dumps(gltf_data)
    json_bytes = json_string.encode('utf-8')
    # Pad JSON chunk with spaces
    json_padding = b' ' * (4 - (len(json_bytes) % 4)) if len(json_bytes) % 4 != 0 else b''
    # BIN chunk must be padded with 0x00
    bin_padding = b'\x00' * (4 - (len(binary_data) % 4)) if len(binary_data) % 4 != 0 else b''

    glb_content = b'glTF'  # Magic number
    glb_content += struct.pack('<I', 2)  # Version 2
    glb_content += struct.pack('<I', 28 + len(json_bytes) + len(json_padding) + len(binary_data) + len(bin_padding))  # Total length
    glb_content += struct.pack('<I', len(json_bytes) + len(json_padding))  # JSON chunk length
    glb_content += struct.pack('<I', 0x4E4F534A)  # JSON chunk type (JSON)
    glb_content += json_bytes + json_padding
    glb_content += struct.pack('<I', len(binary_data) + len(bin_padding)) # BIN chunk length
    glb_content += struct.pack('<I', 0x004E4942) # BIN chunk type
    glb_content += binary_data + bin_padding

    with open(glb_path, "wb") as f:
        f.write(glb_content)

    return glb_path'''

@pytest.fixture
def sample_glb(temp_dir):
    # Correct, minimal GLB content (indexed geometry).
    glb_path = os.path.join(temp_dir, "test.glb")
    # Create a minimal GLB.  This GLB contains a single triangle, defined *with indices*.
    gltf_data = {
      "asset": {"version": "2.0"},
      "scene": 0,
      "scenes": [{"nodes": [0]}],
      "nodes": [{"mesh": 0}],
      "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],  # Added indices
      "accessors": [
          {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "max":[1.0, 1.0, 0.0], "min": [-1.0, -1.0, 0.0]},  # 4 vertices
          {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR", "max": [2], "min": [0]},  # 3 indices for one triangle
      ],
      "bufferViews": [
          {"buffer": 0, "byteLength": 48, "target": 34962},  # vertices
          {"buffer": 0, "byteOffset": 48, "byteLength": 6, "target": 34963}  # indices
      ],
      "buffers": [{"byteLength": 54}] # Combined buffer length
    }
    # Vertex positions (4 vertices forming two triangles, but sharing 2 vertices)
    binary_data = np.array([
        [-1.0, -1.0, 0.0],  # Vertex 0
        [ 1.0, -1.0, 0.0],  # Vertex 1
        [-1.0,  1.0, 0.0],  # Vertex 2
        [ 1.0,  1.0, 0.0]   # Vertex 3
    ], dtype=np.float32).tobytes()

    # Indices for a single triangle (using the first 3 vertices)
    indices_data = np.array([0, 1, 2], dtype=np.uint16).tobytes()  # Use uint16 for UNSIGNED_SHORT

    json_string = json.dumps(gltf_data)
    json_bytes = json_string.encode('utf-8')
    # Pad JSON chunk with spaces
    json_padding = b' ' * (4 - (len(json_bytes) % 4)) if len(json_bytes) % 4 != 0 else b''
    # BIN chunk must be padded with 0x00
    bin_padding = b'\x00' * (4 - (len(binary_data + indices_data) % 4)) if (len(binary_data + indices_data) % 4) != 0 else b''

    glb_content = b'glTF'  # Magic number
    glb_content += struct.pack('<I', 2)  # Version 2
    glb_content += struct.pack('<I', 28 + len(json_bytes) + len(json_padding) + len(binary_data) + len(indices_data) + len(bin_padding))  # Total length
    glb_content += struct.pack('<I', len(json_bytes) + len(json_padding))  # JSON chunk length
    glb_content += struct.pack('<I', 0x4E4F534A)  # JSON chunk type (JSON)
    glb_content += json_bytes + json_padding
    glb_content += struct.pack('<I', len(binary_data) + len(indices_data) + len(bin_padding)) # BIN chunk length
    glb_content += struct.pack('<I', 0x004E4942) # BIN chunk type
    glb_content += binary_data + indices_data + bin_padding

    with open(glb_path, "wb") as f:
        f.write(glb_content)

    return glb_path

@pytest.fixture
def cleaned_glb(temp_dir):
    """Create a fixture for the cleaned GLB file."""
    cleaned_glb_path = os.path.join(temp_dir, "rec_model_cleaned.glb")  # Updated extension
    # Update path to look for .glb instead of .gltf
    original_path = Path(__file__).parent / "data" / "rec_model_cleaned.glb"
    if not original_path.exists():
        raise FileNotFoundError(f"Cleaned GLB file not found at: {original_path}")
    shutil.copy(str(original_path), cleaned_glb_path)
    return cleaned_glb_path

@pytest.fixture
def real_model_glb():
    """Create a fixture for the real model GLB file."""
    glb_path = Path(__file__).parent / "data" / "rec_model_no_curve.glb"
    if not glb_path.exists():
        raise FileNotFoundError(f"Real model GLB file not found at: {glb_path}")
    return str(glb_path)

def test_calculate_utci_daytime(cleaned_glb, sample_epw, temp_dir):
    """Test UTCI calculation for a daytime hour with cleaned GLB."""
    hour_of_year = 12  # Changed to noon (hour 12) which is definitely during daytime
    utci_values = utci_calculator.calculate_utci_from_gltf_epw(
        cleaned_glb, sample_epw, temp_dir, hour_of_year
    )
    assert isinstance(utci_values, np.ndarray)
    assert len(utci_values) > 0

def test_calculate_utci_nighttime(cleaned_glb, sample_epw, temp_dir):
    """Test UTCI calculation for a nighttime hour with cleaned GLB."""
    hour_of_year = 1  # Use an hour that *should not* have sun
    utci_values = utci_calculator.calculate_utci_from_gltf_epw(
        cleaned_glb, sample_epw, temp_dir, hour_of_year
    )
    assert isinstance(utci_values, np.ndarray)
    assert len(utci_values) == 0

def test_parse_glb_valid(cleaned_glb):
    """Test parsing the cleaned GLB file."""
    json_data, bin_data = utci_calculator._load_gltf_or_glb(cleaned_glb)
    assert isinstance(json_data, dict)
    assert isinstance(bin_data, bytes)
    assert "asset" in json_data
    assert json_data["asset"]["version"] == "2.0"
    assert len(json_data['meshes']) == 1  # Should only have one mesh now
    
def test_extract_vertices(cleaned_glb):
    """Test vertex extraction from cleaned GLB."""
    json_data, bin_data = utci_calculator._load_gltf_or_glb(cleaned_glb)
    vertices = utci_calculator._extract_vertices(json_data, bin_data)
    assert isinstance(vertices, np.ndarray)
    assert len(vertices) > 0  # Should have vertices
    assert vertices.shape[1] == 3  # Each vertex should have x,y,z coordinates

def test_load_ill_file(temp_dir):
    # Create a dummy .ill file
    ill_path = os.path.join(temp_dir, "test.ill")
    with open(ill_path, "w") as f:
        f.write("0.1 0.2 0.3\n")
        f.write("0.4 0.5 0.6\n")
        f.write("invalid line\n") #test invalid lines

    irradiance_values = utci_calculator._load_ill_file(ill_path)
    assert isinstance(irradiance_values, np.ndarray)
    assert np.allclose(irradiance_values, np.array([0.6, 1.5]))

def test_load_ill_file_not_found(temp_dir):
    ill_path = os.path.join(temp_dir, "nonexistent.ill")
    with pytest.raises(FileNotFoundError):
        utci_calculator._load_ill_file(ill_path)

def test_run_radiance_command_success(temp_dir):
    # Test a command that should succeed
    result = utci_calculator._run_radiance_command("echo Hello", Path(temp_dir))
    assert result.returncode == 0

def test_run_radiance_command_failure(temp_dir):
    # Test with a command that will fail
    with pytest.raises(subprocess.CalledProcessError):
        utci_calculator._run_radiance_command("nonexistent_command", Path(temp_dir))

def test_parse_gltf_valid(temp_dir):
    """Test parsing a GLTF file."""
    # Create a minimal GLTF file
    gltf_path = os.path.join(temp_dir, "test.gltf")
    
    # Create a minimal GLTF content
    gltf_data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "max":[1.0, 1.0, 0.0], "min": [-1.0, -1.0, 0.0]},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR", "max": [2], "min": [0]},
        ],
        "bufferViews": [
            {"buffer": 0, "byteLength": 48, "target": 34962},
            {"buffer": 0, "byteOffset": 48, "byteLength": 6, "target": 34963}
        ],
        "buffers": [{"byteLength": 54, "uri": "test.bin"}]
    }
    
    # Create the binary file
    binary_data = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 1.0,  1.0, 0.0]
    ], dtype=np.float32).tobytes()
    
    indices_data = np.array([0, 1, 2], dtype=np.uint16).tobytes()
    
    bin_path = os.path.join(temp_dir, "test.bin")
    with open(bin_path, "wb") as f:
        f.write(binary_data + indices_data)
    
    with open(gltf_path, "w") as f:
        json.dump(gltf_data, f)
    
    # Test parsing
    json_data, bin_data = utci_calculator._load_gltf_or_glb(gltf_path)
    assert isinstance(json_data, dict)
    assert isinstance(bin_data, bytes)
    assert "asset" in json_data
    assert json_data["asset"]["version"] == "2.0"

def test_calculate_utci_with_real_model(real_model_glb, sample_epw, temp_dir):
    """Test UTCI calculation with the real model GLB file."""
    hour_of_year = 12  # Noon for daytime calculation
    utci_values = utci_calculator.calculate_utci_from_gltf_epw(
        real_model_glb, sample_epw, temp_dir, hour_of_year
    )
    assert isinstance(utci_values, np.ndarray)
    assert len(utci_values) > 0
    
    # Print some statistics about the UTCI values for visual inspection
    print(f"\nUTCI statistics for real model:")
    print(f"  Count: {len(utci_values):,}")
    print(f"  Min: {np.min(utci_values):.2f}°C")
    print(f"  Max: {np.max(utci_values):.2f}°C")
    print(f"  Mean: {np.mean(utci_values):.2f}°C")
    print(f"  Median: {np.median(utci_values):.2f}°C")

def test_extract_vertices_from_real_model(real_model_glb):
    """Test vertex extraction from the real model GLB file."""
    json_data, bin_data = utci_calculator._load_gltf_or_glb(real_model_glb)
    assert isinstance(json_data, dict)
    assert "meshes" in json_data
    
    # Verify mesh count
    assert len(json_data["meshes"]) > 0
    
    # Extract and verify vertices
    vertices = utci_calculator._extract_vertices(json_data, bin_data)
    assert isinstance(vertices, np.ndarray)
    assert len(vertices) > 0
    assert vertices.shape[1] == 3  # Each vertex has x,y,z coordinates
    
    # Print vertex statistics for visual inspection
    print(f"\nVertex statistics for real model:")
    print(f"  Total vertices: {len(vertices):,}")
    
    # Calculate bounding box
    min_x, min_y, min_z = np.min(vertices, axis=0)
    max_x, max_y, max_z = np.max(vertices, axis=0)
    print(f"  Bounding box:")
    print(f"    X: {min_x:.2f} to {max_x:.2f} (width: {max_x-min_x:.2f})")
    print(f"    Y: {min_y:.2f} to {max_y:.2f} (depth: {max_y-min_y:.2f})")
    print(f"    Z: {min_z:.2f} to {max_z:.2f} (height: {max_z-min_z:.2f})")