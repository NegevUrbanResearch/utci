import subprocess
from pathlib import Path

# 1. Create the directory (using pathlib)
debug_dir = Path("radiance_debug")
debug_dir.mkdir(parents=True, exist_ok=True)
print(f"Directory created: {debug_dir.resolve()}")  # Print absolute path

# 2. Define the output file (using pathlib)
sky_file = debug_dir / "sky.sky"
print(f"Sky file path: {sky_file.resolve()}")

# 3. Hardcoded gensky command (with NO variables, just constants)
command = f"gensky 3 25 7 -a 31.2515 -o 34.7995 -m -30 +s > {sky_file.resolve()}"
print(f"Command to run: {command}")

# 4. Run the command using subprocess.run (with explicit error checking)
try:
    process = subprocess.run(
        command,
        shell=True,
        capture_output=True,  # Capture stdout and stderr
        text=True,           # Decode as text
        check=True,          # Raise exception on error
        cwd=str(debug_dir)   # Run from the debug directory
    )
    print("Command succeeded!")
    print("Standard Output:\n", process.stdout)

except subprocess.CalledProcessError as e:
    print(f"Command FAILED with return code: {e.returncode}")
    print("Standard Output:\n", e.stdout)
    print("Standard Error:\n", e.stderr)

except FileNotFoundError:
    print("ERROR: gensky command not found. Check Radiance installation and PATH.")