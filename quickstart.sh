#!/bin/bash
set -e

# Quickstart script for the JuFo2026 project.
# This script sets up a virtual environment, installs dependencies,
# and runs the full data processing and analysis pipeline.

echo "--- JuFo2026 Project Quickstart ---"

# 1. Check for prerequisites
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 is not installed. Please install Python 3 and try again."
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found."
    exit 1
fi

DATASET_DIR="15571083"
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory '$DATASET_DIR/' not found in the project root."
    echo "The required PDMX dataset is available for download at: https://zenodo.org/records/13763756"
    echo "Please download and extract it to the project root before running."
    exit 1
fi

echo "Prerequisites met."

# 2. Set up virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 3. Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "Dependencies installed."

# 4. Run the analysis pipeline
echo "--- Starting Analysis Pipeline ---"

echo "[1/8] Running Corpus Curation..."
python3 src/corpus_curation.py --min-rating 0

echo "[2/8] Running Score Parser..."
python3 src/score_parser.py --output data/parsed/summaries.json

echo "[3/8] Extracting Harmonic Features..."
python3 src/harmonic_features.py --output-csv data/features/harmonic_features.csv

echo "[4/8] Extracting Melodic Features..."
python3 src/melodic_features.py --output-csv data/features/melodic_features.csv

echo "[5/8] Extracting Rhythmic Features..."
python3 src/rhythmic_features.py --output-csv data/features/rhythmic_features.csv

echo "[6/8] Running Significance Tests..."
python3 src/significance_tests.py \
  --anova-output data/stats/anova_summary.csv \
  --tukey-output data/stats/tukey_hsd.csv

echo "[7/8] Generating Significance Visualizations..."
python3 src/significance_visualizations.py --top-n 15

echo "[8/8] Aggregating Final Metrics..."
python3 src/aggregate_metrics.py

echo "--- Pipeline Complete ---"
echo "Results are available in the 'data/' and 'figures/' directories."
