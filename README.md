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

Run the script by passing a Kaggle dataset handle:
```bash
python train.py --dataset your-kaggle-username/your-dataset-name
```

By default, it will attempt to download a generic plant health dataset. Please modify the target and feature column names inside `train.py` to match the exact schema of your dataset.
