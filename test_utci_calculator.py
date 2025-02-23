# test_utci_calculator.py

# run tests with:
# python -m pytest test_utci_calculator.py -v

from utci_calculator import UTCICalculator, main  # Assuming your class is in utci_calculator.py
import numpy as np
import pytest
from pathlib import Path
import shutil
import json
import pandas as pd
import os
import mock
import sys

# Update paths to match your production data
PROJECT_ROOT = Path(__file__).parent  # Changed to match your directory structure
GLTF_PATH = PROJECT_ROOT / "data" / "rec_model.gltf"
EPW_PATH = PROJECT_ROOT / "data" / "ISR_D_Beer.Sheva.401900_TMYx" / "ISR_D_Beer.Sheva.401900_TMYx.epw"

# Test class organization
class TestUTCICalculator:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        # Setup
        self.test_output = Path("test_output")
        self.calculator = UTCICalculator(
            gltf_path=Path("test_data/simple_cube.glb"),
            epw_path=Path("test_data/test.epw"),
            output_dir=self.test_output
        )
        
        yield  # Test runs here
        
        # Teardown
        if self.test_output.exists():
            shutil.rmtree(self.test_output)

    def test_validate_paths(self):
        """Test path validation with various invalid inputs"""
        with pytest.raises(FileNotFoundError):
            UTCICalculator("bad_path.glb", "test.epw", self.test_output)
            
        with pytest.raises(IsADirectoryError):
            UTCICalculator(Path(__file__).parent, "test.epw", self.test_output)

    def test_vertex_extraction(self):
        """Test GLB model parsing and vertex extraction"""
        vertices = self.calculator.load_model()
        
        # Basic shape validation
        assert vertices.shape[1] == 3, "Should have XYZ coordinates"
        assert len(vertices) > 0, "No vertices extracted"
        
        # Data type and value checks
        assert isinstance(vertices, np.ndarray), "Should return numpy array"
        assert np.issubdtype(vertices.dtype, np.float32), "Should contain float values"
        
        # Coordinate range validation (assuming metric units)
        assert vertices[:, 0].max() < 100, "Unrealistic X coordinate"
        assert vertices[:, 1].max() < 100, "Unrealistic Y coordinate"
        assert vertices[:, 2].max() < 100, "Unrealistic Z coordinate"

    # Add new test cases
    def test_invalid_glb_file(self):
        """Test error handling for corrupted GLB files"""
        corrupted_calculator = UTCICalculator(
            gltf_path=Path("test_data/corrupted.glb"),
            epw_path=Path("test_data/test.epw"),
            output_dir=self.test_output
        )
        
        with pytest.raises(ValueError) as exc_info:
            corrupted_calculator.load_model()
            
        assert "Invalid GLB format" in str(exc_info.value)

class TestUTCICalculatorRealData:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        # Validate real data exists first
        if not GLTF_PATH.exists():
            pytest.skip(f"Missing production GLTF file: {GLTF_PATH}")
        if not EPW_PATH.exists():
            pytest.skip(f"Missing production EPW file: {EPW_PATH}")
            
        self.test_output = PROJECT_ROOT / "test_output"
        self.test_output.mkdir(exist_ok=True)
        
        self.calculator = UTCICalculator(
            gltf_path=GLTF_PATH,
            epw_path=EPW_PATH,
            output_dir=self.test_output
        )
        
        yield  # Test runs here
        
        # Cleanup
        shutil.rmtree(self.test_output, ignore_errors=True)

    @pytest.mark.parametrize("hour", [8, 12, 17])  # Test morning, noon, evening
    def test_hourly_analysis(self, hour):
        """Test analysis workflow for different hours"""
        results = self.calculator.run_analysis(hour_index=hour)
        
        # Verify outputs
        output_png = self.test_output / f"utci_visualization_hour_{hour}.png"
        output_json = self.test_output / f"utci_results_hour_{hour}.json"
        
        assert output_png.exists(), "Missing visualization output"
        assert output_json.exists(), "Missing results JSON"
        
        # Validate JSON structure
        with open(output_json) as f:
            data = json.load(f)
            assert 'metadata' in data
            assert 'point_data' in data
            assert len(data['point_data']) == len(results['points'])

    def test_data_characteristics(self):
        """Validate properties of real data through the calculator"""
        vertices = self.calculator.load_model()
        weather = self.calculator.load_weather_data()
        
        # Model validation
        assert vertices.shape[1] == 3, "3D coordinates expected"
        assert len(vertices) > 0, "Model should have vertices"
        
        # Weather validation
        assert len(weather) == 8760, "EPW should have full year data"
        assert weather['air_temp'].between(-40, 50).all(), "Unrealistic temperatures"
        assert weather['wind_speed'].between(0, 25).all(), "Invalid wind speeds"

    def test_utci_calculation(self):
        """Test UTCI calculation logic with known values"""
        # Create test weather data
        test_weather = pd.DataFrame({
            'air_temp': [20.0],
            'rel_humidity': [50.0],
            'wind_speed': [1.0],
            'mean_rad_temp': [20.0]
        })
        
        # Create test points
        test_points = np.array([[0, 0, 0], [1, 1, 1]])
        
        # Calculate UTCI
        utci_values = self.calculator.calculate_utci(test_points, test_weather, 0)
        
        # Validate against reference value (20°C, 50% RH, 1m/s wind)
        expected_utci = 20.3  # From UTCI reference table
        assert np.allclose(utci_values, expected_utci, atol=0.5), \
            f"Expected UTCI ~{expected_utci}°C, got {utci_values[0]:.1f}°C"

class TestMainFunctionality:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.original_dir = Path.cwd()
        self.test_output = Path("test_main_output")
        yield
        # Cleanup and reset working directory
        if self.test_output.exists():
            shutil.rmtree(self.test_output)
        os.chdir(self.original_dir)

    def test_main_outputs(self):
        """Test the main function's output generation"""
        # Create dummy data structure
        test_data_dir = Path("test_main_data")
        test_data_dir.mkdir(exist_ok=True)
        (test_data_dir / "rec_model.gltf").touch()
        (test_data_dir / "ISR_D_Beer.Sheva.401900_TMYx").mkdir(exist_ok=True)
        (test_data_dir / "ISR_D_Beer.Sheva.401900_TMYx" / "ISR_D_Beer.Sheva.401900_TMYx.epw").touch()
        
        # Mock command line arguments
        with mock.patch.object(sys, 'argv', [
            'utci_calculator.py',
            str(test_data_dir / "rec_model.gltf"),
            str(test_data_dir / "ISR_D_Beer.Sheva.401900_TMYx" / "ISR_D_Beer.Sheva.401900_TMYx.epw"),
            str(self.test_output)
        ]):
            main()
        
        # Verify outputs
        for hour in range(8, 18):
            assert (self.test_output / f"utci_visualization_hour_{hour}.png").exists()
            assert (self.test_output / f"utci_results_hour_{hour}.json").exists()