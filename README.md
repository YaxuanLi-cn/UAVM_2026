<h1 align="center">Last-Meter Precision Navigation for UAVs</h1>


<p align="center">
  <!-- Dataset -->
  <a href="https://huggingface.co/datasets/YaxuanLi/UAVM_2026_test">
    <img src="https://img.shields.io/badge/Dataset-HF%20Data-d8b04c?style=for-the-badge" alt="Dataset">
  </a>

  <!-- Paper -->
  <a href="https://your-paper-link-here">
    <img src="https://img.shields.io/badge/Paper-arXiv-d46a5a?style=for-the-badge" alt="Paper">
  </a>

  <!-- Email -->
  <a href="yaxuanli.cn@gmail.com">
    <img src="https://img.shields.io/badge/Email-YaxuanLi-6b6b6b?style=for-the-badge" alt="Email">
  </a>
</p>

<hr />

## Project Structure

```
UAVM_2026/
├── models/   
├── pairUAV/
│   ├── data_process.sh 
│   └── University-Release.zip
├── baseline/                  
│   ├── SuperGlue/   
│   ├── train.py      
└── └── run.sh             
```

---

## 1. Environment Setup

Create a unified conda environment for our baseline:

```bash
conda create -n uavm python=3.9
conda activate uavm

# Install PyTorch with CUDA support
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Install all dependencies
pip install -r requirements.txt

# Download pretrained models
huggingface-cli download Ramos-Ramos/dino-resnet-50 --local-dir models/dino_resnet
huggingface-cli download Boese0601/MagicDance control_sd15_ini.ckpt --local-dir models/controlnet
```

---

## 2. Data Preparation

### 2.1 Download University-1652 Dataset

Download [University-1652](https://github.com/layumi/University1652-Baseline) upon request (Usually I will reply you in 5 minutes). You may use the [request template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

### 2.2 Download and Process PairUAV Dataset

Download and process the PairUAV dataset:

```bash
cd pairUAV/
bash data_process.sh
cd ..
```

This script downloads the dataset from HuggingFace and extracts train/test/tours data to the `pairUAV/` directory.

## 3. Baseline

### 3.1 Run SuperGlue Feature Matching

First, perform feature matching on image pairs:

```bash
cd baseline/SuperGlue

# 运行总共需要6h，你也可以通过以下方式直接下载我们运行好的结果
bash download_results.sh

# Or Run feature matching
bash run_train.sh

python gen_test_pairs.py
bash run_test.sh
python reorganize_matches.py

rm -rf origin_test_matches_data
cd ..

```

This generates matching results in `train_matches_data/` and `test_matches_data/`.

### 3.2 Train Model

```bash
cd step1
bash run.sh
cd ..
```

---
