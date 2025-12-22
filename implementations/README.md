# 🛠️ Implementations

This directory contains links to official and third-party implementations of deepfake detection methods.

## Official Implementations

| Method | Year | Framework | Link | Notes |
|--------|------|-----------|------|-------|
| FaceForensics++ Benchmark | 2019 | PyTorch | [GitHub](https://github.com/ondyari/FaceForensics) | Official benchmark code |
| Face X-Ray | 2020 | PyTorch | [GitHub](https://github.com/neverUseThisName/Face-X-Ray) | Blending boundary detection |
| F3-Net | 2020 | PyTorch | [GitHub](https://github.com/yyk-wew/F3-Net) | Frequency-based detection |
| Multi-Attention | 2021 | PyTorch | [GitHub](https://github.com/yoctta/multiple-attention) | Attention mechanism |
| DIRE | 2023 | PyTorch | [GitHub](https://github.com/ZhendongWang6/DIRE) | Diffusion reconstruction |
| UnivFD | 2023 | PyTorch | [GitHub](https://github.com/WisconsinAIVision/UniversalFakeDetect) | Universal detector |

## Benchmark Frameworks

| Framework | Description | Link |
|-----------|-------------|------|
| DeepfakeBench | Unified benchmark for 15+ methods | [GitHub](https://github.com/SCLBD/DeepfakeBench) |
| FakeAVCeleb | Audio-visual deepfake benchmark | [GitHub](https://github.com/DASH-Lab/FakeAVCeleb) |

## Pre-trained Models

Most implementations provide pre-trained weights. Check individual repositories for:
- Model weights (`.pth`, `.pt`)
- Configuration files
- Evaluation scripts

## Running Experiments

Basic requirements for most implementations:
```bash
pip install torch torchvision
pip install opencv-python pillow
pip install numpy scipy scikit-learn
```

See `requirements.txt` in the root directory for common dependencies.
