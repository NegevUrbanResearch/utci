import pytest
from pathlib import Path
import numpy as np
import utci_calculator
import os
import tempfile
import shutil
import struct
import json
import subprocess

# Helper function to create a temporary directory
@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


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

                f.write(f"2024,1,1,{i%24 + 1},0,10.0,90.0,0,0,100,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n") #Example line for hour with sun.
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

    return glb_path


def test_calculate_utci_daytime(sample_glb, sample_epw, temp_dir):
    """Test UTCI calculation for a daytime hour."""
    hour_of_year = 20
    utci_values = utci_calculator.calculate_utci_from_gltf_epw(
        sample_glb, sample_epw, temp_dir, hour_of_year
    )
    assert isinstance(utci_values, np.ndarray)
    assert len(utci_values) == 3


def test_calculate_utci_nighttime(sample_glb, sample_epw, temp_dir):
    """Test UTCI calculation for a nighttime hour."""
    hour_of_year = 1
    utci_values = utci_calculator.calculate_utci_from_gltf_epw(
        sample_glb, sample_epw, temp_dir, hour_of_year
    )
    assert isinstance(utci_values, np.ndarray)
    assert len(utci_values) == 0

def test_parse_glb_valid(sample_glb):
    json_data, bin_data = utci_calculator._load_gltf_or_glb(sample_glb)
    assert isinstance(json_data, dict)
    assert isinstance(bin_data, bytes)
    assert "asset" in json_data
    assert json_data["asset"]["version"] == "2.0"
    assert len(bin_data) == 36


def test_parse_glb_invalid(temp_dir):
    # Create an invalid .glb file
    glb_path = os.path.join(temp_dir, "invalid.glb")
    with open(glb_path, "w") as f:
        f.write("Invalid GLB content")

    with pytest.raises(ValueError):
        utci_calculator._load_gltf_or_glb(glb_path)

def test_extract_vertices(sample_glb):
    json_data, bin_data = utci_calculator._load_gltf_or_glb(sample_glb)
    vertices = utci_calculator._extract_vertices(json_data, bin_data)
    assert isinstance(vertices, np.ndarray)
    assert vertices.shape == (3, 3)
    expected_vertices = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [-1.0,  1.0, 0.0],
    ])
    assert np.allclose(vertices, expected_vertices)

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