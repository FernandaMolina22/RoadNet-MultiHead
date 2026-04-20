# Topology-Aware Road Extraction from Sentinel-2 Imagery

This repository contains the implementation developed for my Master's thesis focused on improving road extraction from satellite imagery by incorporating **topological supervision** into deep learning models.

The objective of this work is to move beyond pixel-wise segmentation and improve the **structural quality and connectivity** of predicted road networks.

---

## Project Overview

This project compares two deep learning approaches:

### UNetSingle (Baseline)
- Standard U-Net architecture
- Predicts a **binary road segmentation mask**
- Focuses on pixel-wise accuracy

### UNetMultiHead (Proposed)
- Multi-head U-Net with shared encoder-decoder
- Predicts:
  - Road mask
  - Centerline heatmap
  - Junction heatmap
- Uses **multi-task learning** to incorporate structural information

The goal is to improve **road connectivity and topology**, not just segmentation accuracy.

---

## Dataset

This project uses the **Amazon Wild Roads dataset**.

Download dataset here:  
https://polimi365-my.sharepoint.com/:f:/g/personal/10947496_polimi_it/IgCZxN7u9CAHTqI62gw0oQTtAeUqHeuqQBQ9kvUGVxSFfFQ?e=N7Nypj 

##  Environment Setup

This project was developed using Python and PyTorch.

Main dependencies include:
- Python 3.10+
- PyTorch
- torchvision
- NumPy
- Pillow

You can install them manually or using your preferred environment manager (e.g., pip or conda).

---
## Repository Structure
```text
├── make_topology_labels_phase1.py
├──  dataset_awr_seg_patch_split.py
├── dataset_awr_seg_mosaic_split.py
├── model_unet_single.py
├── model_unet_multihead.py
├── train_unet_single.py
├── train_unet_multihead.py
├── evaluate_models.py
├── evaluation_utils.py
├── connectivity_utils.py
├── skeleton_utils.py
```
