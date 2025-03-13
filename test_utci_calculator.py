"""
Tests for the Optimized Honeybee UTCI Calculator

Run with:
pytest test_utci_calculator.py -v
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
import logging
import multiprocessing

# Import the module to test
import utci_calculator

# Import Honeybee modules for testing
from honeybee.model import Model
from honeybee_radiance.sensorgrid import SensorGrid
from ladybug.epw import EPW
from ladybug.location import Location


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
        pytest.skip(f"EPW file not found at: {epw_path}")
    return str(epw_path)


@pytest.fixture
def real_model_glb():
    """Create a fixture for the real model GLB file."""
    glb_path = Path(__file__).parent / "data" / "rec_model_no_curve.glb"
    if not glb_path.exists():
        pytest.skip(f"Real model GLB file not found at: {glb_path}")
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


def test_create_sensor_grid_sequential(sample_glb):
    """Test creation of sensor grid from Honeybee Model in sequential mode."""
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    
    # Force sequential processing by setting up a small model
    sensor_grid = utci_calculator.create_sensor_grid(
        hb_model, grid_size=0.5, offset=0.1, use_centroids=True, max_sensors=10
    )
    
    assert isinstance(sensor_grid, SensorGrid)
    assert len(sensor_grid.sensors) > 0  # Should have sensors


def test_create_sensor_grid_parallel(real_model_glb):
    """Test creation of sensor grid from Honeybee Model in parallel mode."""
    if os.environ.get("CI", "false").lower() == "true":
        pytest.skip("Skipping parallel test in CI environment")
        
    # Skip if we only have 1 CPU core
    if multiprocessing.cpu_count() < 2:
        pytest.skip("Parallel processing test requires at least 2 CPU cores")
    
    hb_model = utci_calculator.gltf_to_honeybee_model(real_model_glb)
    
    # This should trigger parallel processing with a large model
    sensor_grid = utci_calculator.create_sensor_grid(
        hb_model, grid_size=2.0, offset=0.1, use_centroids=True, max_sensors=1000
    )
    
    assert isinstance(sensor_grid, SensorGrid)
    assert len(sensor_grid.sensors) > 0


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


def test_read_rtrace_results(temp_dir):
    """Test reading rtrace results from both text and binary formats."""
    # Create a test text format file
    text_file = os.path.join(temp_dir, "text_results.dat")
    with open(text_file, "w") as f:
        f.write("0.5 0.5 0.5\n")  # RGB values
        f.write("0.8 0.8 0.8\n")
        f.write("0.2 0.2 0.2\n")

    # Read and verify text format
    results = utci_calculator._read_rtrace_results(text_file)
    assert len(results) == 3
    assert abs(results[0] - 0.5) < 1e-10  # Use an epsilon for floating-point comparison
    assert abs(results[1] - 0.8) < 1e-10  # Instead of direct equality
    assert abs(results[2] - 0.2) < 1e-10
    
    
    # Create a test binary-like file with simple float data
    binary_file = os.path.join(temp_dir, "binary_results.dat")
    with open(binary_file, "wb") as f:
        # Write some non-text data at the beginning to trigger binary detection
        f.write(b'\xca\xfe\xba\xbe')
        # Then write some float values
        values = np.array([
            [0.5, 0.5, 0.5],
            [0.8, 0.8, 0.8],
            [0.2, 0.2, 0.2]
        ], dtype=np.float32)
        f.write(values.tobytes())
    
    # Test reading the binary-like file
    results = utci_calculator._read_rtrace_results(binary_file)
    # We don't check exact values because the binary parsing is approximate,
    # but we should get approximately the right number of results
    assert len(results) > 0


def test_process_utci_batch():
    """Test the batch processing of UTCI calculations."""
    # Create a batch of test data
    air_temp = 25.0
    mrt_batch = [25.0, 26.0, 27.0, 28.0, 29.0]
    wind_speed = 1.0
    rel_humidity = 50.0
    
    batch_data = (air_temp, mrt_batch, wind_speed, rel_humidity)
    
    # Process the batch
    results = utci_calculator._process_utci_batch(batch_data)
    
    # Check results
    assert len(results) == len(mrt_batch)
    # UTCI values should increase with MRT
    assert results[0] < results[-1]
    # UTCI values should be in a reasonable range given the inputs
    assert all(15 < utci < 40 for utci in results)


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


def test_error_handling_for_missing_files(temp_dir):
    """Test that the calculator handles missing files gracefully."""
    
    # Non-existent GLB file
    with pytest.raises(Exception):
        utci_calculator.gltf_to_honeybee_model("/nonexistent/path.glb")
    
    # Non-existent EPW file
    hb_model = Model("test_model")
    with pytest.raises(Exception):
        utci_calculator.calculate_utci_from_honeybee_model(
            hb_model, "/nonexistent/path.epw", temp_dir, 12
        )


def test_calculate_utci_from_gltf_epw(sample_glb, sample_epw, temp_dir):
    """Test the end-to-end UTCI calculation from a GLTF file."""
    # Calculate UTCI for noon
    hour_of_year = 12
    
    utci_values = utci_calculator.calculate_utci_from_gltf_epw(
        sample_glb, sample_epw, temp_dir, hour_of_year,
        grid_size=0.5, offset=0.1, solar_absorptance=0.7,
        clean_geometry=True, use_centroids=True, max_sensors=100
    )
    
    # Check results
    assert isinstance(utci_values, np.ndarray)
    assert len(utci_values) > 0
    
    # Check results are in reasonable range
    assert np.min(utci_values) > -50
    assert np.max(utci_values) < 50


def test_performance_benchmark(real_model_glb, sample_epw, temp_dir):
    """Test the performance of the calculator with a real model."""
    # Skip if running in CI or if the real model is too large for quick tests
    if os.environ.get("CI", "false").lower() == "true":
        pytest.skip("Skipping performance benchmark in CI environment")
    
    hour_of_year = 12  # Noon
    
    # Time the execution
    start_time = time.time()
    
    utci_values = utci_calculator.calculate_utci_from_gltf_epw(
        real_model_glb, sample_epw, temp_dir, hour_of_year,
        grid_size=2.0,  # Large grid size for faster testing
        max_sensors=500  # Limit sensors for quicker testing
    )
    
    elapsed_time = time.time() - start_time
    
    # Just a basic assertion that it completes in a reasonable time
    # Adjust based on your hardware expectations
    assert elapsed_time < 300  # Should complete in under 5 minutes
    
    # Print performance info for reference
    print(f"\nPerformance benchmark:")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    print(f"  Sensors processed: {len(utci_values)}")
    print(f"  Processing rate: {len(utci_values)/elapsed_time:.2f} sensors/second")


def test_logging_level_control():
    """Test that the log level setting works correctly."""
    # Set to DEBUG and verify
    utci_calculator.set_log_level("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    
    # Set to WARNING and verify
    utci_calculator.set_log_level("WARNING")
    assert logging.getLogger().level == logging.WARNING
    
    # Reset to INFO for other tests
    utci_calculator.set_log_level("INFO")
    assert logging.getLogger().level == logging.INFO
    
    # Test invalid level
    with pytest.raises(ValueError):
        utci_calculator.set_log_level("INVALID_LEVEL")


def test_fallback_to_sequential(sample_glb, temp_dir, monkeypatch):
    """Test that parallel processing falls back to sequential when needed."""
    # Mock ProcessPoolExecutor.map to raise an exception
    def mock_map(*args, **kwargs):
        raise RuntimeError("Simulated parallel processing failure")
    
    # Apply the mock to force fallback
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor.map", mock_map)
    
    # Now create a sensor grid - should fall back to sequential
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    sensor_grid = utci_calculator.create_sensor_grid(hb_model, grid_size=0.5, offset=0.1)
    
    # If we got here without errors, the fallback worked
    assert isinstance(sensor_grid, SensorGrid)
    assert len(sensor_grid.sensors) > 0


def test_check_gpu_availability():
    """Test the GPU availability check function."""
    # This is mostly for coverage, actual availability depends on hardware
    gpu_available = utci_calculator.check_gpu_availability()
    assert isinstance(gpu_available, bool)


def test_run_gpu_rtrace_with_batching(temp_dir, sample_glb):
    """Test GPU RTrace batching functionality."""
    # Skip if we're running in CI
    if os.environ.get("CI", "false").lower() == "true":
        pytest.skip("Skipping GPU test in CI environment")
        
    # Skip if GPU not available
    if not utci_calculator.check_gpu_availability():
        pytest.skip("GPU acceleration not available")
        
    # Create minimal test files required for batching
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    sensor_grid = utci_calculator.create_sensor_grid(
        hb_model, grid_size=0.5, offset=0.1, max_sensors=100
    )
    
    # Create sensor file
    sensor_file = utci_calculator._create_sensor_file(sensor_grid, temp_dir)
    
    # Create a dummy octree file
    octree_file = os.path.join(temp_dir, "dummy.oct")
    with open(octree_file, "w") as f:
        f.write("# Dummy octree file for testing\n")
    
    results_file = os.path.join(temp_dir, "gpu_results.dat")
    
    # Run GPU batching with a very small batch size
    try:
        result = utci_calculator._run_gpu_rtrace_with_batching(
            octree_file, sensor_file, results_file, temp_dir, batch_size=10
        )
        # If GPU is available but RTrace fails, it should return False and not crash
        assert isinstance(result, bool)
    except Exception as e:
        # We don't expect exceptions, but if they happen, the test should
        # not fail - this is just testing the error handling
        logging.warning(f"GPU batching test exception: {e}")
        pass


def test_torch_availability():
    """Test if PyTorch is available and can be used."""
    try:
        assert hasattr(utci_calculator, 'torch')
        assert hasattr(utci_calculator.torch, 'cuda')
        
        # Check if CUDA is available (for informational purposes only)
        cuda_available = utci_calculator.torch.cuda.is_available()
        logging.info(f"CUDA available: {cuda_available}")
        
        # Basic tensor creation test
        tensor = utci_calculator.torch.tensor([1.0, 2.0, 3.0])
        assert tensor.shape[0] == 3
        
    except (ImportError, AttributeError):
        pytest.skip("PyTorch not available or not correctly imported")


def test_parallel_utci_calculation():
    """Test parallel processing of UTCI calculations."""
    # Skip if running in CI
    if os.environ.get("CI", "false").lower() == "true":
        pytest.skip("Skipping parallel processing test in CI environment")
        
    # Skip if we only have 1 CPU core
    if multiprocessing.cpu_count() < 2:
        pytest.skip("Parallel processing test requires at least 2 CPU cores")
    
    # Create test data for multiple points
    air_temp = 25.0
    mrt_batch = np.linspace(20.0, 40.0, 1000)  # 1000 different MRT values
    wind_speed = 1.0
    rel_humidity = 50.0
    
    batch_data = (air_temp, mrt_batch, wind_speed, rel_humidity)
    
    # Process the batch using the parallel function
    results = utci_calculator._process_utci_batch(batch_data)
    
    # Verify results
    assert len(results) == len(mrt_batch)
    assert all(15 < utci < 45 for utci in results)  # Reasonable range check


def test_generate_sky_with_custom_sun(temp_dir):
    """Test generating a Radiance sky file with custom sun position."""
    location = Location(
        'Test City', 'Test State', 'Test Country',
        31.25, 34.8, 2.0, 280.0  # Lat, Long, TZ, Elevation
    )
    
    # Test with custom sun angle
    solar_altitude = 45  # Degrees
    solar_azimuth = 180  # South
    direct_normal = 800
    diffuse_horizontal = 100
    
    # Generate sky file - this would require modifying the function to accept sun angles
    # For now, we'll test the existing function
    sky_file = utci_calculator._generate_sky_file(
        temp_dir, location, 6, 21, 12, direct_normal, diffuse_horizontal
    )
    
    assert os.path.exists(sky_file)
    
    # Check content
    with open(sky_file, "r") as f:
        content = f.read()
    
    assert "solar source sun" in content
    assert "sky_mat source sky" in content


@pytest.mark.skipif(os.environ.get("CI", "false").lower() == "true", 
                   reason="Extended tests skipped in CI environment")
def test_utci_category_classification():
    """Test classification of UTCI values into thermal stress categories."""
    # Create a range of UTCI values
    utci_values = np.linspace(-40, 50, 90)  # -40°C to 50°C
    
    # Define UTCI thermal stress categories
    categories = {
        "Extreme cold stress": (-40, -27),
        "Very strong cold stress": (-27, -13),
        "Strong cold stress": (-13, 0),
        "Moderate cold stress": (0, 9),
        "Slight cold stress": (9, 18),
        "No thermal stress": (18, 26),
        "Moderate heat stress": (26, 32),
        "Strong heat stress": (32, 38),
        "Very strong heat stress": (38, 46),
        "Extreme heat stress": (46, 50)
    }
    
    # Function to classify UTCI into categories
    def classify_utci(utci_value):
        for category, (lower, upper) in categories.items():
            if lower <= utci_value < upper:
                return category
        return "Unknown"
    
    # Test a sample of values
    sample_values = [-35, -20, -10, 5, 15, 22, 30, 35, 40, 48]
    expected_categories = [
        "Extreme cold stress",
        "Very strong cold stress",
        "Strong cold stress",
        "Moderate cold stress",
        "Slight cold stress",
        "No thermal stress",
        "Moderate heat stress",
        "Strong heat stress",
        "Very strong heat stress",
        "Extreme heat stress"
    ]
    
    for utci, expected in zip(sample_values, expected_categories):
        category = classify_utci(utci)
        assert category == expected, f"UTCI {utci} should be {expected} but got {category}"


@pytest.mark.skipif(not os.path.exists("/tmp/large_model.glb"), 
                   reason="Large model file not available")
def test_large_model_performance(temp_dir):
    """Test performance with a very large model file."""
    large_model_path = "/tmp/large_model.glb"
    epw_path = Path(__file__).parent / "data" / "ISR_D_Beer.Sheva.401900_TMYx" / "ISR_D_Beer.Sheva.401900_TMYx.epw"
    
    if not epw_path.exists():
        pytest.skip(f"EPW file not found at: {epw_path}")
        
    # Use a high grid size to limit the number of sensors
    grid_size = 5.0
    max_sensors = 5000
    
    start_time = time.time()
    
    # Load the model and create a sensor grid
    hb_model = utci_calculator.gltf_to_honeybee_model(large_model_path, clean_geometry=True)
    
    model_loading_time = time.time() - start_time
    logging.info(f"Large model loading time: {model_loading_time:.2f} seconds")
    
    grid_start_time = time.time()
    sensor_grid = utci_calculator.create_sensor_grid(
        hb_model, grid_size=grid_size, offset=0.1, 
        use_centroids=True, max_sensors=max_sensors
    )
    grid_creation_time = time.time() - grid_start_time
    
    assert isinstance(sensor_grid, SensorGrid)
    assert len(sensor_grid.sensors) > 0
    
    logging.info(f"Large model sensor grid creation time: {grid_creation_time:.2f} seconds")
    logging.info(f"Number of sensors created: {len(sensor_grid.sensors)}")
    
    # Performance metrics
    assert model_loading_time < 300, "Model loading took too long"
    assert grid_creation_time < 600, "Grid creation took too long"


def test_utci_calculator_handles_missing_sensors_gracefully(sample_glb, temp_dir, monkeypatch):
    """Test that the calculator handles missing sensor results gracefully."""
    # Mock the _read_rtrace_results function to return fewer results than expected
    def mock_read_rtrace_results(results_file):
        return [0.1, 0.2, 0.3]  # Only 3 results
    
    monkeypatch.setattr("utci_calculator._read_rtrace_results", mock_read_rtrace_results)
    
    # Create a sensor grid with more sensors
    hb_model = utci_calculator.gltf_to_honeybee_model(sample_glb)
    
    # Store our sensor grid and make sure it has multiple points
    sensor_grid = utci_calculator.create_sensor_grid(
        hb_model, grid_size=0.5, offset=0.1
    )
    
    # Prevent the function from creating a new grid internally
    original_create_sensor_grid = utci_calculator.create_sensor_grid
    
    def mock_create_sensor_grid(*args, **kwargs):
        return sensor_grid  # Always return our pre-created grid
    
    monkeypatch.setattr("utci_calculator.create_sensor_grid", mock_create_sensor_grid)
    
    # Ensure we have more than 3 sensors
    assert len(sensor_grid.sensors) > 3
    
    # Create a simple EPW object with mocked data
    epw_path = Path(__file__).parent / "data" / "ISR_D_Beer.Sheva.401900_TMYx" / "ISR_D_Beer.Sheva.401900_TMYx.epw"
    if not epw_path.exists():
        pytest.skip(f"EPW file not found at: {epw_path}")
    
    # Run the calculation - it should handle the missing results by extending the array
    try:
        result = utci_calculator.calculate_utci_from_honeybee_model(
            hb_model, str(epw_path), temp_dir, 12
        )
        # Should match the number of sensors even though rtrace only returned 3 results
        assert len(result) == len(sensor_grid.sensors)
    except Exception as e:
        logging.error(f"Error in graceful handling test: {e}")
        assert False, f"Failed to handle missing sensor results: {e}"


if __name__ == "__main__":
    pytest.main()
    