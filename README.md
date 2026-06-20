# MVB-ML-MSCNN
Official PyTorch implementation of **"MVB Compound Fault Diagnosis Based on Time-Frequency Analysis and Multi-Label Multi-Scale Convolutional Neural Networks"**.

## Overview
This work proposes a multi-label multi-scale CNN (ML-MSCNN) framework for MVB compound fault diagnosis. It converts 1D MVB signals into 2D STFT time-frequency maps, extracts multi-scale distortion features via parallel convolutional branches, enhances key fault regions with CBAM attention, and achieves parallel multi-fault diagnosis with a physically-constrained Sigmoid decision strategy.

## Environment Requirements
- Python 3.9
- PyTorch 2.0.1
- CUDA 11.7 (optional, for GPU acceleration)
- OS: Windows / Linux / macOS

## Installation
1. Clone this repository:
```bash
git clone https://github.com/[xiaoxuu13]/MVB-ML-MSCNN.git
cd MVB-ML-MSCNN
