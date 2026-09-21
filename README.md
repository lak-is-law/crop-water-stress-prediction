# Crop Water Stress Prediction

This repository contains a Kernel-Based Classification model (Non-linear SVM with RBF Kernel) to predict whether a crop plot is under water stress based on features such as:
- Soil moisture
- Temperature
- Humidity
- Vegetation index

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

This project uses a flattened structure where datasets and scripts live in the root directory.

### Database Connectivity & Analysis
Run the analysis script. This script will:
- Connect to an SQLite database (`crop_data.db`) to demonstrate DB connectivity.
- Load the `dataset.csv` into a SQL table.
- Query the database to fetch the data.
- Train the non-linear SVM (RBF kernel) on the queried data.

```bash
python analyze.py
```
