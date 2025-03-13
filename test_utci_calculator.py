"""
Tests for the Optimized Ground-Level UTCI Calculator
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
import logging
import multiprocessing

# Import the module to test
import utci_calculator

# Import Honeybee modules for testing
from honeybee.model import Model
from honeybee_radiance.sensorgrid import SensorGrid
from ladybug.epw import EPW
from ladybug.location import Location


# Fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_epw():
    """Get path to sample EPW file."""
    epw_path = Path(__file__).parent / "data" / "ISR_D_Beer.Sheva.401900_TMYx" / "ISR_D_Beer.Sheva.401900_TMYx.epw"
    if not epw_path.exists():
        pytest.skip(f"EPW file not found at: {epw_path}")
    return str(epw_path)


@pytest.fixture
def real_model_glb():
    """Get path to real model GLB file."""
    glb_path = Path(__file__).parent / "data" / "rec_model_no_curve.glb"
    if not glb_path.exists():
        pytest.skip(f"Real model GLB file not found at: {glb_path}")
    return str(glb_path)


@pytest.fixture
def sample_glb(temp_dir):
    """Create a simple GLB file with one triangle for testing."""
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


# Tests
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
    assert len(hb_model.shades) > 0


def test_create_sensor_grid_with_ground_focus(sample_glb):
    """Test creation of sensor grid with ground-level focus."""
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    
    # Test with ground-level focus
    sensor_grid = utci_calculator.create_sensor_grid(
        hb_model, grid_size=0.5, offset=0.1, use_centroids=True, max_sensors=10,
        focus_ground_level=True, height_threshold=1.5
    )
    
    assert isinstance(sensor_grid, SensorGrid)
    assert len(sensor_grid.sensors) > 0


def test_create_sensor_grid_parallel(real_model_glb):
    """Test parallel creation of sensor grid."""
    # Skip if running in CI or only 1 CPU core
    if os.environ.get("CI", "false").lower() == "true" or multiprocessing.cpu_count() < 2:
        pytest.skip("Skipping parallel test (CI environment or insufficient cores)")
    
    hb_model = utci_calculator.gltf_to_honeybee_model(real_model_glb)
    
    # This should trigger parallel processing with a large model
    sensor_grid = utci_calculator.create_sensor_grid(
        hb_model, grid_size=2.0, offset=0.1, use_centroids=True, max_sensors=1000,
        focus_ground_level=True, height_threshold=1.5
    )
    
    assert isinstance(sensor_grid, SensorGrid)
    assert len(sensor_grid.sensors) > 0


def test_generate_sky_file(temp_dir, sample_epw):
    """Test generating a Radiance sky file."""
    epw = EPW(sample_epw)
    location = Location(
        epw.location.city, epw.location.state, epw.location.country,
        epw.location.latitude, epw.location.longitude, epw.location.time_zone,
        epw.location.elevation
    )
    
    sky_file = utci_calculator._generate_sky_file(
        temp_dir, location, 6, 21, 12, 800, 120
    )
    
    assert os.path.exists(sky_file)
    with open(sky_file, "r") as f:
        content = f.read()
        assert "solar source sun" in content
        assert "sky_mat source sky" in content


def test_create_radiance_model(sample_glb, temp_dir):
    """Test creating a Radiance model file."""
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    rad_file = utci_calculator._create_radiance_model(hb_model, temp_dir)
    
    assert os.path.exists(rad_file)
    with open(rad_file, "r") as f:
        content = f.read()
        assert "plastic red" in content
        assert "polygon shade_" in content


def test_read_rtrace_results(temp_dir):
    """Test reading rtrace results from text format."""
    # Create a test text format file
    text_file = os.path.join(temp_dir, "text_results.dat")
    with open(text_file, "w") as f:
        f.write("0.5 0.5 0.5\n")
        f.write("0.8 0.8 0.8\n")
        f.write("0.2 0.2 0.2\n")

    results = utci_calculator._read_rtrace_results(text_file)
    assert len(results) == 3
    assert abs(results[0] - 0.5) < 1e-10
    assert abs(results[1] - 0.8) < 1e-10
    assert abs(results[2] - 0.2) < 1e-10


def test_process_utci_batch():
    """Test batch processing of UTCI calculations."""
    # Prepare test data
    air_temp = 25.0
    mrt_batch = [25.0, 26.0, 27.0, 28.0, 29.0]
    wind_speed = 1.0
    rel_humidity = 50.0
    
    results = utci_calculator._process_utci_batch((air_temp, mrt_batch, wind_speed, rel_humidity))
    
    assert len(results) == len(mrt_batch)
    # Values should increase with MRT
    assert results[0] < results[-1]
    # Values should be in a reasonable range
    assert all(15 < utci < 40 for utci in results)


@pytest.mark.parametrize("hour_of_year", [12])  # Noon
def test_calculate_utci_with_ground_focus(sample_glb, sample_epw, temp_dir, hour_of_year):
    """Test UTCI calculation with ground-level focus."""
    # Skip if Radiance not available
    try:
        cmd = "rtrace -version > /dev/null 2>&1" if os.name != 'nt' else "rtrace -version"
        if os.system(cmd) != 0:
            pytest.skip("Radiance not properly installed")
    except:
        pytest.skip("Error checking Radiance installation")
    
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    
    # Test with ground-level focus
    utci_values = utci_calculator.calculate_utci_from_honeybee_model(
        hb_model, sample_epw, temp_dir, hour_of_year,
        focus_ground_level=True, height_threshold=1.5
    )
    
    assert isinstance(utci_values, np.ndarray)
    if len(utci_values) > 0:
        assert -50 < np.min(utci_values) < 50


def test_error_handling_for_missing_files(temp_dir):
    """Test error handling for missing files."""
    with pytest.raises(Exception):
        utci_calculator.gltf_to_honeybee_model("/nonexistent/path.glb")
    
    hb_model = Model("test_model")
    with pytest.raises(Exception):
        utci_calculator.calculate_utci_from_honeybee_model(
            hb_model, "/nonexistent/path.epw", temp_dir, 12
        )


def test_end_to_end_with_ground_focus(sample_glb, sample_epw, temp_dir):
    """Test end-to-end UTCI calculation with ground-level focus."""
    # Skip if Radiance not available
    try:
        cmd = "rtrace -version > /dev/null 2>&1" if os.name != 'nt' else "rtrace -version"
        if os.system(cmd) != 0:
            pytest.skip("Radiance not properly installed")
    except:
        pytest.skip("Error checking Radiance installation")
    
    utci_values = utci_calculator.calculate_utci_from_gltf_epw(
        sample_glb, sample_epw, temp_dir, 12,
        grid_size=0.5, max_sensors=100,
        focus_ground_level=True, height_threshold=1.5
    )
    
    assert isinstance(utci_values, np.ndarray)
    assert len(utci_values) > 0
    assert -50 < np.min(utci_values) < 50


def test_logging_level_control():
    """Test log level setting."""
    utci_calculator.set_log_level("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    
    utci_calculator.set_log_level("WARNING")
    assert logging.getLogger().level == logging.WARNING
    
    # Reset to INFO
    utci_calculator.set_log_level("INFO")
    
    with pytest.raises(ValueError):
        utci_calculator.set_log_level("INVALID_LEVEL")


def test_fallback_to_sequential(sample_glb, monkeypatch):
    """Test fallback to sequential processing."""
    # Mock ProcessPoolExecutor.map to raise an exception
    def mock_map(*args, **kwargs):
        raise RuntimeError("Simulated parallel processing failure")
    
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor.map", mock_map)
    
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    sensor_grid = utci_calculator.create_sensor_grid(hb_model, grid_size=0.5, offset=0.1)
    
    assert isinstance(sensor_grid, SensorGrid)
    assert len(sensor_grid.sensors) > 0


def test_check_gpu_availability():
    """Test GPU availability check."""
    gpu_available = utci_calculator.check_gpu_availability()
    assert isinstance(gpu_available, bool)


if __name__ == "__main__":
    pytest.main(["-v"])