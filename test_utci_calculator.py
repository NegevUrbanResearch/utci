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
def sample_epw(temp_dir):
    """Create a sample EPW file with proper formatting."""
    epw_path = os.path.join(temp_dir, "test.epw")
    with open(epw_path, "w") as f:
        # Write EPW header lines
        f.write("LOCATION,Unknown Location,,,,,,,0,0,0,0\n")
        f.write("DESIGN CONDITIONS,0\n")
        f.write("TYPICAL/EXTREME PERIODS,0\n")
        f.write("GROUND TEMPERATURES,0\n")
        f.write("HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n")
        f.write("COMMENTS 1\n")
        f.write("COMMENTS 2\n")
        f.write("DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31\n")
        
        # Add 8760 lines of properly formatted data
        # Format: Year,Month,Day,Hour,Minute,Data Source,Dry Bulb Temp,Dew Point Temp,Rel Humidity,Atm Pressure,
        #         Extraterrestrial Horizontal Radiation,Extraterrestrial Direct Normal Radiation,
        #         Horizontal Infrared Radiation Intensity,Global Horizontal Radiation,
        #         Direct Normal Radiation,Diffuse Horizontal Radiation,Global Horizontal Illuminance,
        #         Direct Normal Illuminance,Diffuse Horizontal Illuminance,Zenith Luminance,
        #         Wind Direction,Wind Speed,Total Sky Cover,Opaque Sky Cover,Visibility,
        #         Ceiling Height,Present Weather Observation,Present Weather Codes,
        #         Precipitable Water,Aerosol Optical Depth,Snow Depth,Days Since Last Snowfall,
        #         Albedo,Liquid Precipitation Depth,Liquid Precipitation Quantity
        for i in range(8760):
            hour = i % 24
            day = (i // 24) % 31 + 1
            month = (i // (24 * 31)) % 12 + 1
            
            # Set radiation values based on hour
            if 8 <= hour <= 18:  # Daytime
                dir_normal_rad = "100"
                diff_horiz_rad = "20"
            else:  # Nighttime
                dir_normal_rad = "0"
                diff_horiz_rad = "0"
            
            # Create a full data line with all 35 required fields
            data_line = f"2024,{month},{day},{hour},0,9999,10.0,9.0,90.0,101300,0,0,315,0,{dir_normal_rad},{diff_horiz_rad}," + \
                       f"0,0,0,0,180,1.0,0,0,20000,77777,9,999999999,0,0.0,0,88,0.0,0,0\n"
            f.write(data_line)

    return epw_path

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