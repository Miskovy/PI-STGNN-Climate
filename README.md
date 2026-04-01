# Physics-Informed Spatiotemporal Graph Neural Network (PI-STGNN) for Climate Attribution

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the official PyTorch implementation of the **PI-STGNN**, a novel deep learning architecture designed to predict and attribute the drivers of extreme precipitation events in the U.S. Midwest. 

By modeling the atmosphere as a spherical graph and embedding the Clausius-Clapeyron thermodynamic relationship directly into the loss landscape, this model overcomes the spatial distortion and physical inconsistencies of traditional 2D Convolutional Neural Networks (CNNs).

## 🌍 Overview
Traditional climate attribution models often process atmospheric data as independent, temporally agnostic 2D snapshots. The PI-STGNN improves upon baseline models by:
1. **Spherical Graph Mapping:** Reprojecting latitude-longitude grids onto a continuous Fibonacci sphere to preserve true geographical distances.
2. **Spatiotemporal Tracking:** Utilizing Graph Convolutional Networks (GCN) combined with Long Short-Term Memory (LSTM) layers to track the 5-day antecedent trajectory of synoptic-scale systems.
3. **Physics-Informed Regularization:** Penalizing the network for predicting extreme precipitation in environments lacking the requisite vertically integrated moisture, ensuring predictions obey atmospheric thermodynamics.

## 📊 Performance Metrics (40-Year Test Set)
Evaluated on an unseen 40-year historical dataset (ERA5/PRISM), the PI-STGNN demonstrates superior capability in catching extreme events in highly imbalanced datasets (top 5% threshold):
* **Accuracy:** 91.65%
* **Recall:** 85.52%
* **Precision:** 35.73%
* **F1-Score:** 0.5041

## ⚙️ Repository Structure
* `src/data_pipeline.py`: Automated batch downloaders and preprocessors for 40 years of ERA5 (NetCDF) and PRISM (GeoTIFF) data.
* `src/model.py`: The PyTorch definition of the `PI_STGNN` architecture and the weighted Physics-Informed loss function.
* `src/train.py`: The production training loop featuring dynamic class weighting and Z-score normalization.
* `src/explainability.py`: Gradient-based feature attribution to generate saliency heatmaps, decoupling dynamic (Z500) and thermodynamic (Humidity) drivers.

## 🚀 Quick Start
### 1. Installation
Ensure you have PyTorch and PyTorch Geometric installed. 
```bash
pip install -r requirements.txt
```
### 2. Copernicus API Setup
You will need a CDS API key to download the historical ERA5 dataset. Save your credentials to ~/.cdsapirc:
```bash
url: [https://cds.climate.copernicus.eu/api](https://cds.climate.copernicus.eu/api)
key: YOUR_PERSONAL_ACCESS_TOKEN
```
### 3. Execution
Run the end-to-end pipeline from the src directory:
```bash
cd src
python data_pipeline.py  # Downloads and chunks data into .pt tensors
python train.py          # Trains the STGNN and outputs evaluation metrics
python explainability.py # Generates the feature attribution heatmap
```
## 📝 Citation
If you utilize this codebase or architecture in your research, please cite our corresponding paper:

```text
@article{YourName2026,
  title={Unraveling the Spatiotemporal Dynamics of Extreme Precipitation: A Physics-Informed Graph Neural Network Approach},
  author={Mazen Khairy},
  journal={TBD},
  year={2026}
}
```
