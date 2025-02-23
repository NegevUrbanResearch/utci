# test_utci_calculator.py
import pytest
from pathlib import Path
import shutil
import json
import pandas as pd
import numpy as np
import os
import mock
from utci_calculator import UTCICalculator, main

# --- Constants & Setup ---
PROJECT_ROOT = Path(__file__).parent
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)
if not (TEST_DATA_DIR / "simple_cube.glb").exists():
    (TEST_DATA_DIR / "simple_cube.glb").write_bytes(
        b'glTF\x02\x00\x00\x00\x80\x00\x00\x00,\x00\x00\x00JSON{"accessors":[{"bufferView":0,"byteOffset":0,"componentType":5126,"count":1,"max":[0.0,0.0,0.0],"min":[0.0,0.0,0.0],"type":"VEC3"}],"bufferViews":[{"buffer":0,"byteLength":12,"byteOffset":0,"target":34962}],"buffers":[{"byteLength":12}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],"nodes":[{"mesh":0}],"scenes":[{"nodes":[0]}],"scene":0}\x1c\x00\x00\x00BIN\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    )
if not (TEST_DATA_DIR / "test.epw").exists():
    with open(TEST_DATA_DIR / "test.epw", "w") as f:
        f.write("LOCATION,Test Location,,,0,0,0,0,0,0\n")
        f.write("DATA PERIODS,1,1,Data,,1/1,12/31\n")
        f.write("DESIGN CONDITIONS,0\n")
        f.write("TYPICAL/EXTREME PERIODS,0\n")
        f.write("GROUND TEMPERATURES,0\n")
        f.write("HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n")
        f.write("COMMENTS 1\n")
        f.write("COMMENTS 2\n")
        f.write("DATA,1,1,1,1,20,9999,9999,9999,9999,50,1,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999\n")

GLTF_PATH = PROJECT_ROOT / "data" / "rec_model.gltf"
EPW_PATH = PROJECT_ROOT / "data" / "ISR_D_Beer.Sheva.401900_TMYx" / "ISR_D_Beer.Sheva.401900_TMYx.epw"

class TestUTCICalculator:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.test_output = Path("test_output")
        self.calculator = UTCICalculator(
            gltf_path=TEST_DATA_DIR / "simple_cube.glb",
            epw_path=TEST_DATA_DIR / "test.epw",
            output_dir=self.test_output
        )
        yield
        if self.test_output.exists():
            shutil.rmtree(self.test_output)

    def test_validate_paths(self):
        with pytest.raises(FileNotFoundError):
            UTCICalculator("bad_path.glb", "test.epw", self.test_output)
        with pytest.raises(ValueError, match="Invalid GLTF file extension"):
             UTCICalculator(TEST_DATA_DIR / "test.epw", "test.epw", self.test_output)

    def test_vertex_extraction(self):
        vertices = self.calculator.load_model()
        assert isinstance(vertices, np.ndarray)
        assert vertices.shape == (1, 3)
        assert np.array_equal(vertices, [[0.0, 0.0, 0.0]])

    def test_load_weather_data(self):
        weather_data = self.calculator.load_weather_data()
        assert isinstance(weather_data, pd.DataFrame)
        assert len(weather_data) == 1
        assert 'air_temp' in weather_data.columns

    def test_calculate_utci_edge_cases(self):
        test_weather = pd.DataFrame({
            'air_temp': [20.0], 'rel_humidity': [50.0],
            'wind_speed': [0.0], 'mean_rad_temp': [20.0]
        })
        test_points = np.array([[0, 0, 0]])
        utci_value = self.calculator.calculate_utci(test_points, test_weather, 0)[0]
        assert utci_value > 0, "UTCI should be calculated even with zero wind"
        empty_points = np.array([])
        utci_empty = self.calculator.calculate_utci(empty_points, test_weather, 0)
        assert utci_empty.size == 0, "UTCI with empty points should return empty array"

    def test_invalid_hour_index(self):
        test_weather = pd.DataFrame({'air_temp': [20], 'rel_humidity': [50], 'wind_speed': [1], 'mean_rad_temp': [20]})
        test_points = np.array([[0, 0, 0]])
        with pytest.raises(ValueError, match="Invalid hour_index"):
            self.calculator.calculate_utci(test_points, test_weather, 24)

    def test_create_visualization(self):
        test_points = np.array([[0, 0, 0], [1, 1, 1]])
        test_utci_values = np.array([20.0, 25.0])
        self.calculator.create_visualization(test_points, test_utci_values, 12)
        assert (self.test_output / "utci_visualization_hour_12.png").exists()
        assert (self.test_output / "utci_results_hour_12.json").exists()
        with pytest.raises(ValueError):
             self.calculator.create_visualization(test_points, np.array([20, 25, 30]), hour_index=1)

    def test_visualization_empty_data(self):
        self.calculator.create_visualization(np.array([]), np.array([]), hour_index=1)
        assert not any(self.test_output.iterdir()), "No files should be created."

    def test_subsampling(self):
        """Test that subsampling is working correctly."""
        test_points = np.random.rand(2000, 3)  # More points than max_plot_points
        test_utci_values = np.random.rand(2000)
        self.calculator.max_plot_points = 100  # Set a small limit
        self.calculator.max_json_points = 150

        self.calculator.create_visualization(test_points, test_utci_values, 1)

        # Check plot file
        plot_file = self.test_output / "utci_visualization_hour_1.png"
        assert plot_file.exists()

        # Check JSON file and subsampling
        json_file = self.test_output / "utci_results_hour_1.json"
        assert json_file.exists()
        with open(json_file, 'r') as f:
            data = json.load(f)
        assert data['metadata']['num_points_calculated'] == 2000
        assert data['metadata']['num_points_saved'] <= self.calculator.max_json_points
        assert len(data['point_data']) <= self.calculator.max_json_points



class TestUTCICalculatorRealData:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        if not GLTF_PATH.exists():
            pytest.skip(f"Missing production GLTF file: {GLTF_PATH}")
        if not EPW_PATH.exists():
            pytest.skip(f"Missing production EPW file: {EPW_PATH}")

        self.test_output = PROJECT_ROOT / "test_output"
        self.test_output.mkdir(exist_ok=True)
        self.calculator = UTCICalculator(gltf_path=GLTF_PATH, epw_path=EPW_PATH, output_dir=self.test_output)

        #Temporarily limit points, restore original
        self.original_max_plot_points = getattr(self.calculator, 'max_plot_points', None)
        self.original_max_json_points = getattr(self.calculator, 'max_json_points', None)
        self.calculator.max_plot_points = 1000
        self.calculator.max_json_points = 1000

        yield

        if self.original_max_plot_points is not None:
                self.calculator.max_plot_points = self.original_max_plot_points
        if self.original_max_json_points is not None:
                self.calculator.max_json_points = self.original_max_json_points
        shutil.rmtree(self.test_output, ignore_errors=True)

    def test_hourly_analysis(self):
        hour = 12
        results = self.calculator.run_analysis(hour_index=hour)
        output_png = self.test_output / f"utci_visualization_hour_{hour}.png"
        output_json = self.test_output / f"utci_results_hour_{hour}.json"
        assert output_png.exists()
        assert output_json.exists()
        with open(output_json) as f:
            data = json.load(f)
            assert 'metadata' in data
            assert 'point_data' in data
            # Don't check exact length; we're subsampling now.
            assert len(data['point_data']) <= self.calculator.max_json_points
            assert data['metadata']['num_points_saved'] <= self.calculator.max_json_points
            assert data['metadata']['num_points_calculated'] > 0


    def test_utci_calculation(self):
        test_weather = pd.DataFrame({
            'air_temp': [20.0],
            'rel_humidity': [50.0],
            'wind_speed': [1.0],
            'mean_rad_temp': [20.0]
        })
        test_points = np.array([[0, 0, 0], [1, 1, 1]])
        utci_values = self.calculator.calculate_utci(test_points, test_weather, 0)
        expected_utci = 20.3
        assert np.allclose(utci_values, expected_utci, atol=0.5)

class TestMainFunctionality:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.original_dir = Path.cwd()
        self.test_output = Path("test_main_output")
        yield
        if self.test_output.exists():
            shutil.rmtree(self.test_output)
        os.chdir(self.original_dir)

    def test_main_outputs(self):
        test_data_dir = Path("test_main_data")
        test_data_dir.mkdir(exist_ok=True)
        (test_data_dir / "rec_model.gltf").touch()
        epw_dir = test_data_dir / "ISR_D_Beer.Sheva.401900_TMYx"
        epw_dir.mkdir(exist_ok=True)
        (epw_dir / "ISR_D_Beer.Sheva.401900_TMYx.epw").touch()

        with mock.patch('sys.argv', [
            'utci_calculator.py',
            str(test_data_dir / "rec_model.gltf"),
            str(epw_dir / "ISR_D_Beer.Sheva.401900_TMYx.epw"),
            "--output_dir", str(self.test_output),
            "--hour", "10",
            "--plot_points", "500",  # Test command-line args
            "--json_points", "200"
        ]):
            main()

        assert (self.test_output / "utci_visualization_hour_10.png").exists()
        assert (self.test_output / "utci_results_hour_10.json").exists()

        # Check subsampling via command-line args
        with open(self.test_output / "utci_results_hour_10.json") as f:
            data = json.load(f)
        assert len(data['point_data']) <= 200, "JSON subsampling should respect --json_points"