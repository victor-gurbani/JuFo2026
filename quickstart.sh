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

# 1a. Check for dataset and offer to download
DATASET_DIR="15571083"
if [ ! -d "$DATASET_DIR" ]; then
    echo "Dataset directory '$DATASET_DIR/' not found."
    read -p "Would you like to download and extract it automatically? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating dataset directory..."
        mkdir -p "$DATASET_DIR"

        echo "Fetching file list from Zenodo..."
        ZENODO_API_URL="https://zenodo.org/api/records/15571083/files"
        FILE_URLS=$(curl -s "$ZENODO_API_URL" | grep -o 'https://[^\"]*content' | sed 's/\"//g')

        if [ -z "$FILE_URLS" ]; then
            echo "Error: Could not retrieve file list from Zenodo."
            exit 1
        fi

        echo "Downloading dataset files..."
        cd "$DATASET_DIR"
        for url in $FILE_URLS; do
            filename=$(echo "$url" | sed -e 's|/content$||' -e 's|.*/||')
            echo "Downloading $filename..."
            wget -c -q --show-progress -O "$filename" "$url"
        done

        echo "Extracting archives..."
        for archive in *.tar.gz; do
            echo "Extracting $archive..."
            tar -xzf "$archive"
            rm "$archive"
        done

        cd ..
        echo "Dataset downloaded and extracted successfully."
    else
        echo "Please download and extract the dataset manually to the '$DATASET_DIR' directory."
        exit 1
    fi
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
