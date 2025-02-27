"""
Tests for the Simplified Honeybee UTCI calculator

Run with:
pytest test_simplified_honeybee_utci_calculator.py -v
"""

import pytest
import os
import time
import tempfile
import shutil
import json
import struct
import numpy as np
from pathlib import Path
import subprocess

# Import the module to test
import utci_calculator

# Import Honeybee modules for testing
from honeybee.model import Model
from honeybee_radiance.sensorgrid import SensorGrid


# Helper function to create a temporary directory
@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_epw():
    """Uses actual EPW file for testing."""
    epw_path = Path(__file__).parent / "data" / "ISR_D_Beer.Sheva.401900_TMYx" / "ISR_D_Beer.Sheva.401900_TMYx.epw"
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW file not found at: {epw_path}")
    return str(epw_path)


@pytest.fixture
def real_model_glb():
    """Create a fixture for the real model GLB file."""
    glb_path = Path(__file__).parent / "data" / "rec_model_no_curve.glb"
    if not glb_path.exists():
        raise FileNotFoundError(f"Real model GLB file not found at: {glb_path}")
    return str(glb_path)


@pytest.fixture
def sample_glb(temp_dir):
    """Create a simple GLB file for testing."""
    # Create a minimal GLB with a single triangle
    glb_path = os.path.join(temp_dir, "test.glb")
    
    # Simple geometry data - just one triangle
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
      "buffers": [{"byteLength": 54}]
    }
    
    # Binary data for vertices and indices
    binary_data = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 1.0,  1.0, 0.0]
    ], dtype=np.float32).tobytes()
    
    indices_data = np.array([0, 1, 2], dtype=np.uint16).tobytes()
    
    # Create GLB file
    json_string = json.dumps(gltf_data)
    json_bytes = json_string.encode('utf-8')
    json_padding = b' ' * (4 - (len(json_bytes) % 4)) if len(json_bytes) % 4 != 0 else b''
    bin_padding = b'\x00' * (4 - (len(binary_data + indices_data) % 4)) if (len(binary_data + indices_data) % 4) != 0 else b''

    glb_content = b'glTF'  # Magic number
    glb_content += struct.pack('<I', 2)  # Version 2
    glb_content += struct.pack('<I', 28 + len(json_bytes) + len(json_padding) + len(binary_data + indices_data) + len(bin_padding))  # Total length
    glb_content += struct.pack('<I', len(json_bytes) + len(json_padding))  # JSON chunk length
    glb_content += struct.pack('<I', 0x4E4F534A)  # JSON chunk type (JSON)
    glb_content += json_bytes + json_padding
    glb_content += struct.pack('<I', len(binary_data + indices_data) + len(bin_padding))  # BIN chunk length
    glb_content += struct.pack('<I', 0x004E4942)  # BIN chunk type (BIN)
    glb_content += binary_data + indices_data + bin_padding

    with open(glb_path, "wb") as f:
        f.write(glb_content)

    return glb_path


def test_load_gltf_or_glb(sample_glb):
    """Test loading of GLB file."""
    json_data, bin_data = utci_calculator._load_gltf_or_glb(sample_glb)
    assert isinstance(json_data, dict)
    assert isinstance(bin_data, bytes)
    assert "asset" in json_data
    assert json_data["asset"]["version"] == "2.0"


def test_gltf_to_honeybee_model(sample_glb):
    """Test conversion of GLB to Honeybee Model."""
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    assert isinstance(hb_model, Model)
    assert len(hb_model.shades) > 0  # Should have created shades


def test_create_sensor_grid(sample_glb):
    """Test creation of sensor grid from Honeybee Model."""
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    sensor_grid = utci_calculator.create_sensor_grid(hb_model, grid_size=0.5, offset=0.1)
    assert isinstance(sensor_grid, SensorGrid)
    assert len(sensor_grid.sensors) > 0  # Should have sensors


def test_run_radiance_command(temp_dir):
    """Test running a simple Radiance command."""
    # Create a simple test file
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test content")
    
    # Run a simple command (using cat or type depending on OS)
    if os.name == 'nt':  # Windows
        command = f"type {test_file}"
    else:  # Unix-like
        command = f"cat {test_file}"
        
    try:
        result = utci_calculator._run_radiance_command(command, temp_dir)
        assert result.returncode == 0
        assert "test content" in result.stdout
    except subprocess.CalledProcessError:
        pytest.skip("Skipping test_run_radiance_command as subprocess execution failed")


def test_generate_sky_file(temp_dir, sample_epw):
    """Test generating a Radiance sky file."""
    from ladybug.epw import EPW
    from ladybug.location import Location
    
    # Load EPW file for a test location
    epw = EPW(sample_epw)
    location = Location(
        epw.location.city, 
        epw.location.state, 
        epw.location.country,
        epw.location.latitude, 
        epw.location.longitude, 
        epw.location.time_zone,
        epw.location.elevation
    )
    
    # Generate sky file
    sky_file = utci_calculator._generate_sky_file(
        temp_dir, location, 6, 21, 12, 800, 120
    )
    
    # Check if file was created
    assert os.path.exists(sky_file)
    
    # Check content
    with open(sky_file, "r") as f:
        content = f.read()
    
    assert "gensky" in content
    assert "solar source sun" in content
    assert "sky_mat source sky" in content


def test_create_radiance_model(sample_glb, temp_dir):
    """Test creating a Radiance model file from a Honeybee model."""
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    rad_file = utci_calculator._create_radiance_model(hb_model, temp_dir)
    
    assert os.path.exists(rad_file)
    
    with open(rad_file, "r") as f:
        content = f.read()
    
    assert "plastic red" in content
    assert "polygon shade_" in content


def test_create_sensor_file(sample_glb, temp_dir):
    """Test creating a sensor points file from a sensor grid."""
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    sensor_grid = utci_calculator.create_sensor_grid(hb_model, grid_size=0.5, offset=0.1)
    sensor_file = utci_calculator._create_sensor_file(sensor_grid, temp_dir)
    
    assert os.path.exists(sensor_file)
    
    # Check if file contains sensor data
    with open(sensor_file, "r") as f:
        lines = f.readlines()
    
    assert len(lines) == len(sensor_grid.sensors)
    

@pytest.mark.parametrize("hour_of_year", [12])  # Noon, should be daytime
def test_calculate_utci_from_honeybee_model(sample_glb, sample_epw, temp_dir, hour_of_year):
    """Test UTCI calculation for a specific hour using a Honeybee model."""
    # Skip if Radiance is not installed or not working properly
    try:
        if os.name == 'nt':  # Windows
            result = os.system("rtrace -version")
        else:  # Unix-like
            result = os.system("rtrace -version > /dev/null 2>&1")
        if result != 0:
            pytest.skip("Radiance not properly installed or configured. Skipping test.")
    except:
        pytest.skip("Error checking Radiance installation. Skipping test.")
    
    # Create Honeybee model from sample GLB
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    
    # Calculate UTCI
    utci_values = utci_calculator.calculate_utci_from_honeybee_model(
        hb_model, sample_epw, temp_dir, hour_of_year
    )
    
    # Verify results
    assert isinstance(utci_values, np.ndarray)
    
    # If we got results, check they're within reasonable range
    if len(utci_values) > 0:
        assert np.min(utci_values) > -50
        assert np.max(utci_values) < 50


def test_calculate_utci_nighttime(sample_glb, sample_epw, temp_dir):
    """Test UTCI calculation for a nighttime hour."""
    hour_of_year = 1  # Early morning, likely to be night
    
    # Try to calculate UTCI for nighttime
    utci_values = utci_calculator.calculate_utci_from_gltf_epw(
        sample_glb, sample_epw, temp_dir, hour_of_year
    )
    
    # For nighttime, should return empty array
    assert isinstance(utci_values, np.ndarray)
    assert len(utci_values) == 0


def test_large_model_vertex_count_match(real_model_glb, sample_epw, temp_dir):
    """Test that sensor count matches values in UTCI calculation for large models."""
    # Skip if running in CI or if the real model is too large for quick tests
    if os.environ.get("CI", "false").lower() == "true":
        pytest.skip("Skipping large model test in CI environment")
    
    # Create Honeybee model
    hb_model = utci_calculator.gltf_to_honeybee_model(real_model_glb)
    
    # Create sensor grid (using smaller grid size for faster testing)
    sensor_grid = utci_calculator.create_sensor_grid(hb_model, grid_size=1.0, offset=0.1)
    sensor_count = len(sensor_grid.sensors)
    
    # Run calculation for a daytime hour
    hour_of_year = 12  # Noon
    utci_values = utci_calculator.calculate_utci_from_honeybee_model(
        hb_model, sample_epw, temp_dir, hour_of_year,
        grid_size=1.0  # Same grid size as above
    )
    
    print(f"\nTest validation for large model:")
    print(f"  UTCI values: {len(utci_values)}")
    print(f"  Total sensors: {sensor_count}")
    print(f"  Model shade count: {len(hb_model.shades)}")
    
    # Check if UTCI count matches sensor count (this should now be consistent)
    assert len(utci_values) == sensor_count, \
        f"UTCI values count ({len(utci_values)}) should match sensor count ({sensor_count})"
    
    # Check UTCI values are in reasonable range
    if len(utci_values) > 0:
        assert np.min(utci_values) > -50, f"Minimum UTCI too low: {np.min(utci_values):.2f}°C"
        assert np.max(utci_values) < 50, f"Maximum UTCI too high: {np.max(utci_values):.2f}°C"