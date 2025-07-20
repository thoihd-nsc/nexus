#!/bin/bash
python3 -m venv .venv
if [ $? -eq 0 ]; then
    echo "✓ Virtual environment created successfully"
else
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install other dependencies
echo "Installing additional dependencies..."
pip install -r requirements.txt

# Create Jupyter kernel
python -m ipykernel install --user --name=venv --display-name="venv"

echo "Setup completed successfully!"
