#!/usr/bin/env python3
"""Debug Quarto functionality step by step."""

import os
import subprocess
import sys
from pathlib import Path
import json

print("=== Debug Quarto Setup ===")

# 1. Check if Quarto is available
print("1. Checking Quarto installation...")
try:
    result = subprocess.run(['quarto', '--version'], capture_output=True, text=True)
    print(f"   Quarto version: {result.stdout.strip()}")
except FileNotFoundError:
    print("   ERROR: Quarto not found in PATH")
    sys.exit(1)

# 2. Create simple test files
print("2. Creating simple test files...")
outputs_dir = Path("outputs/reports/debug_test")
outputs_dir.mkdir(parents=True, exist_ok=True)

# Create simple JSON data file
test_data = {
    "message": "Hello from Python!",
    "numbers": [1, 2, 3, 4, 5]
}

data_file = outputs_dir / "test_data.json"
with open(data_file, 'w') as f:
    json.dump(test_data, f, indent=2)
print(f"   Created: {data_file}")

# Create simple QMD file
qmd_content = """---
title: "Test Report"
format: html
jupyter: python3
---

# Simple Test

```{python}
import json
from pathlib import Path

# Load test data
with open('test_data.json', 'r') as f:
    data = json.load(f)

print(f"Message: {data['message']}")
print(f"Numbers: {data['numbers']}")
```

This is a test report.
"""

qmd_file = outputs_dir / "simple_test.qmd"
with open(qmd_file, 'w') as f:
    f.write(qmd_content)
print(f"   Created: {qmd_file}")

# 3. Test Quarto rendering
print("3. Testing Quarto rendering...")
try:
    env = os.environ.copy()
    env['QUARTO_PYTHON'] = sys.executable
    
    # Try with absolute path and forward slashes
    qmd_file_posix = str(qmd_file.absolute()).replace('\\', '/')
    print(f"   Using file path: {qmd_file_posix}")
    
    result = subprocess.run([
        'quarto', 'render', qmd_file_posix,
        '--to', 'html'
    ], 
    capture_output=True, 
    text=True, 
    cwd=str(outputs_dir.absolute()),
    env=env
    )
    
    print(f"   Return code: {result.returncode}")
    if result.stdout:
        print(f"   Stdout: {result.stdout}")
    if result.stderr:
        print(f"   Stderr: {result.stderr}")
    
    # Check if HTML was created
    html_file = outputs_dir / "simple_test.html"
    if html_file.exists():
        print(f"   SUCCESS: HTML created at {html_file}")
        print(f"   File size: {html_file.stat().st_size} bytes")
    else:
        print(f"   ERROR: HTML file not created")
        
except Exception as e:
    print(f"   ERROR: {e}")

print("=== Debug Complete ===")