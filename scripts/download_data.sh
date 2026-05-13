#!/bin/bash
# Download the CFPB Consumer Complaints dataset
# Source: https://www.consumerfinance.gov/data-research/consumer-complaints/

set -euo pipefail

DATA_DIR="data/raw"
URL="https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
OUTPUT="$DATA_DIR/complaints.csv"

if [ -f "$OUTPUT" ]; then
    echo "Dataset already exists at $OUTPUT"
    exit 0
fi

mkdir -p "$DATA_DIR"

echo "Downloading CFPB Consumer Complaints dataset..."
curl -L -o "$DATA_DIR/complaints.csv.zip" "$URL"

echo "Extracting..."
cd "$DATA_DIR"
unzip -o complaints.csv.zip
rm complaints.csv.zip
cd -

echo "Dataset ready at $OUTPUT"
echo "Size: $(du -h $OUTPUT | cut -f1)"
echo "Lines: $(wc -l < $OUTPUT)"
