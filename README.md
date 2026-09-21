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

2. Make sure you have your Kaggle credentials set up if you intend to download datasets directly from Kaggle. You can place your `kaggle.json` inside `~/.kaggle/`.

## Usage

This project has been split into two scripts to separate the dataset download process from the database connectivity and analysis.

### 1. Download Dataset
Run this script to download the dataset from Kaggle into a local `data/` folder.
```bash
python download_dataset.py --dataset your-kaggle-username/your-dataset-name
```
*(By default, it will download a sample plant health dataset.)*

### 2. Database Connectivity & Analysis
Once downloaded, run the analysis script. This script will:
- Connect to an SQLite database (`crop_data.db`) to demonstrate DB connectivity.
- Load the downloaded CSV data into a SQL table.
- Query the database to fetch the data.
- Train the non-linear SVM (RBF kernel) on the queried data.

```bash
python analyse.py
```
