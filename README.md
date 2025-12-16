# KGTCN: Kernel-Guided Transformer-CNN Network for Low-Cost Pathological Whole Slide Imaging


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of the paper **"Kernel-Guided Transformer-CNN Network for Low-Cost Pathological Whole Slide Imaging: Hardware-Algorithm Synergy"**

## 🌟 Overview
This project proposes a hardware-algorithm co-design solution for low-cost pathological whole slide imaging (WSI). By replacing expensive Z-axis components with intelligent algorithms, we achieve diagnostic-grade WSI quality at a cost below $10k. The core innovations include:
- **KGTCN Model**: Kernel-Guided Transformer-CNN for asymmetric defocus restoration
- **GORA Algorithm**: Global Optimization Registration for seamless image stitching
- **Three-step Pipeline**: Defocus classification → Blur kernel estimation → Targeted restoration

## 📋 Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
  - [Defocus Classification Network](#1-defocus-classification-network)
  - [Blur Kernel Extraction Network](#2-blur-kernel-extraction-network)
  - [Image Restoration Network (KGTCN)](#3-image-restoration-network-kgtcn)
- [Model Evaluation](#model-evaluation)
- [Code Structure](#code-structure)
- [Citation](#citation)
- [Contact](#contact)

## 🛠️ Environment Setup
### Prerequisites
- Python 3.7.6
- CUDA 11.7 (recommended)
- PyTorch 1.13.0

### Installation
1. Create a virtual environment (recommended):
   ```bash
   conda create -n kgtcn python=3.7.6
   conda activate kgtcn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## 📊 Dataset Preparation
We use the **ProPathoFocus-32** dataset, which contains 32 histologically confirmed prostate biopsy specimens with controlled defocus levels. The dataset structure should follow:

### Dataset Structure
```
Dataset/
├── Train/
│   ├── LR-0_5/       # Mild negative defocus (-5, 0)μm
│   ├── LR-5_10/      # Severe negative defocus [-10, -5]μm
│   ├── LR+0_5/       # Mild positive defocus (0, 5)μm
│   ├── LR+5_10/      # Severe positive defocus [5, 10]μm
│   └── HR/           # In-focus images (ground truth)
└── Test/
    ├── LR-0_5/
    ├── LR-5_10/
    ├── LR+0_5/
    ├── LR+5_10/
    └── HR/
```

### Data Preprocessing
- Images are resized to 256×256 (classification) / 224×224 (restoration)
- Normalization: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
- Blank region filtration and contrast optimization

## 🚀 Model Training
### 1. Defocus Classification Network
Classifies defocused images into 5 categories (LR-5_10, LR-0_5, HR, LR+0_5, LR+5_10)

#### Training Script
```bash
cd blur_classification
python train_my_classify.py
```

#### Key Parameters (modify in `train_my_classify.py`)
- `data_dir`: Path to dataset root
- `batch_size`: 64 (default)
- `num_epochs`: 150 (default)
- `lr`: 0.0001 (default)

#### Network Structure
- 12-layer CNN with CBAM attention modules
- 5×5 convolutions for larger receptive field
- Total parameters: ~7.5M (lightweight for edge deployment)

### 2. Blur Kernel Extraction Network
Estimates category-specific blur kernels based on physical light propagation models

#### Training Scripts
```bash
cd blur_kernel_extract
# Train for specific defocus level (example: LR+0_5)
python train_LR+0_5.py
```

#### Key Parameters (modify in `train_LR-{level}.py`)
- `epochs`: 300 (default)
- `lr`: 0.0001 (default)
- `batch_size`: 64 (default)
- `dataset_root`: Path to dataset
- `model_save_path`: Path to save trained models

#### Loss Function
- CharbonnierLoss for stable kernel estimation
- L1 loss between synthetic defocused images and real defocused images

### 3. Image Restoration Network (KGTCN)
Core model integrating kernel guidance and transformer-CNN hybrid architecture

#### Training Scripts
```bash
# Train for specific defocus level (example: LR+0_5)
python train_deblur_LR+0_5.py
```

#### Key Parameters (modify in `train_deblur_LR-{level}.py`)
- `g_lr`: Generator learning rate (0.0001 default)
- `d_lr`: Discriminator learning rate (0.00001 default)
- `num_epochs`: 300 (default)
- `batch_size`: 16 (default)
- `middle_block_num`: Number of SHSA-B modules (12 default)
- `base_channel`: 32 (default)
- `gan`: Enable GAN loss (False default)
- `L`: Adversarial loss weight (0.1 default)
- `p`: Perceptual loss weight (0.1 default)

## 📈 Model Evaluation
### Evaluation Metrics
- **Classification**: Precision, Recall, F1-score, AUC
- **Restoration**: PSNR, SSIM, LPIPS, MAE
- **Stitching**: Alignment error, processing time, WSI size

### Evaluation Scripts
```bash
# Evaluate defocus classification
cd blur_classification
python compute_test.py

# Evaluate blur kernel estimation
cd blur_kernel_extract
python test.py

# Evaluate image restoration (example for LR-5_10)
python compute_test_LR-5_10.py

# Calculate inference time
python compute_time.py
```

### Expected Performance
| Model | PSNR (LR+0_5) | SSIM (LR+0_5) | FPS | Model Size |
|-------|---------------|---------------|-----|------------|
| KGTCN | 38.76 dB      | 0.968         | 43.45 | 51.2M |

## 📁 Code Structure
```
KGTCN-WSI/
├── backbones/                # Basic network components
│   ├── resnet.py             # ResNet backbone
│   ├── unet_parts.py         # U-Net components
│   └── arch_util.py          # Architecture utilities
├── blur_classification/      # Defocus classification
│   ├── net.py                # Network definition
│   └── train_my_classify.py  # Training script
├── blur_kernel_extract/      # Blur kernel estimation
│   ├── blur_kernel_estimator_net.py  # Kernel network
│   ├── blur_kernel_loss.py   # Loss functions
│   ├── train_LR-{level}.py   # Training scripts
│   └── test.py               # Evaluation script
├── cal_psnr_ssim.py          # PSNR/SSIM calculation
├── compute_test.py           # General evaluation
├── compute_test_LR-{level}.py # Level-specific evaluation
├── compute_time.py           # Inference time measurement
├── data_loader.py            # Data loading utilities
├── draw_loss.py              # Loss curve visualization
├── draw_psnr.py              # PSNR curve visualization
├── loss.py                   # Restoration loss functions
├── model.py                  # KGTCN model definition
├── process_dataset.py        # Dataset preprocessing
├── requirement.txt           # Dependencies
├── train_deblur_LR-{level}.py # Restoration training scripts
├── utils.py                  # Utility functions
└── val_data_loader.py        # Validation data loader
```

## 📝 Citation
If you use this code or dataset in your research, please cite our paper:
```
@article{yang2024kgtcn,
  title={Kernel-Guided Transformer-CNN Network for Low-Cost Pathological Whole Slide Imaging: Hardware-Algorithm Synergy},
  author={Yang, Haichao and Luo, Tao and Zhang, Yuqing and Yang, Jiyun and Yan, Changbao and Zhang, Chenggui and Li, Yi and Yang, Runbiao},
  journal={},
  year={2024},
  publisher={}
}
```

## 📧 Contact
For questions or issues, please contact:
- Haichao Yang: dlyhc@163.com


## Data and Code Availability
- The ProPathoFocus-32 dataset is available upon reasonable request
- Implementation codes, training/testing scripts, and configuration files are publicly accessible at: [https://github.com/dlyhc/KGTCN-WSI](https://github.com/dlyhc/KGTCN-WSI)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
