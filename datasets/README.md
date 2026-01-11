# 📊 Deepfake Detection Datasets

This directory contains information about datasets used in deepfake detection research.

## Video Deepfake Datasets

### FaceForensics++ (2019)
- **Size**: 1,000 original videos, 4,000 manipulated videos (~1.8M frames)
- **Manipulation Types**: DeepFakes, Face2Face, FaceSwap, NeuralTextures
- **Quality Levels**: c0 (raw), c23 (light compression), c40 (heavy compression)
- **Paper**: [arXiv:1901.08971](https://arxiv.org/abs/1901.08971)
- **Link**: [GitHub](https://github.com/ondyari/FaceForensics)

### Celeb-DF v2 (2020)
- **Size**: 590 real videos, 5,639 deepfake videos
- **Quality**: High-quality face swaps with reduced visual artifacts
- **Paper**: [arXiv:1909.12962](https://arxiv.org/abs/1909.12962)
- **Link**: [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics)

### DFDC (2020)
- **Size**: 128,154 videos (100,000+ unique subjects)
- **Manipulation Types**: Multiple deepfake generation methods
- **Link**: [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge)

### DeeperForensics-1.0 (2020)
- **Size**: 60,000 videos
- **Features**: Diverse perturbations (compression, blur, noise)
- **Paper**: [arXiv:2001.03024](https://arxiv.org/abs/2001.03024)
- **Link**: [GitHub](https://github.com/EndlessSora/DeeperForensics-1.0)

### WildDeepfake (2020)
- **Size**: 7,314 face sequences from 707 videos
- **Features**: Real-world deepfakes collected from the internet
- **Link**: [GitHub](https://github.com/deepfakeinthewild/deepfake-in-the-wild)

---

## Image Deepfake Datasets

### UFD / CNNDetection (2020)
- **Size**: 720K images
- **Generator**: ProGAN-based images
- **Paper**: [arXiv:1912.11035](https://arxiv.org/abs/1912.11035)
- **Link**: [GitHub](https://github.com/PeterWang512/CNNDetection)

### iFakeFaceDB (2020)
- **Size**: 63K images
- **Paper**: GANprintR [DOI:10.1109/JSTSP.2020.3007250](http://dx.doi.org/10.1109/JSTSP.2020.3007250)
- **Link**: [GitHub](https://github.com/socialabubi/iFakeFaceDB)

### GenImage (2023)
- **Size**: 1.2M images
- **Generators**: Midjourney, Stable Diffusion, BigGAN
- **Paper**: [arXiv:2306.08571](https://arxiv.org/abs/2306.08571)
- **Link**: [GitHub](https://github.com/GenImage-Dataset/GenImage)

### DiffusionDB (2023)
- **Size**: 14M images
- **Generator**: Stable Diffusion with prompts
- **Paper**: [arXiv:2210.14896](https://arxiv.org/abs/2210.14896)
- **Link**: [Website](https://poloclub.github.io/diffusiondb/) | [HuggingFace](https://huggingface.co/datasets/poloclub/diffusiondb)

### DeepFakeFace (2023)
- **Size**: 120K images
- **Paper**: [arXiv:2309.02218](https://arxiv.org/abs/2309.02218)
- **Link**: [GitHub](https://github.com/OpenRL-Lab/DeepFakeFace)

### DF40 (2024)
- **Size**: Large-scale
- **Features**: 40 generation techniques
- **Paper**: [arXiv:2406.13495](https://arxiv.org/abs/2406.13495)
- **Link**: [GitHub](https://github.com/YZY-stack/DF40)

### ArtiFact (2023)
- **Size**: 2.5M images
- **Features**: 25 generative models
- **Paper**: [arXiv:2302.11970](https://arxiv.org/abs/2302.11970)
- **Link**: [arXiv](https://arxiv.org/abs/2302.11970)

### DiffusionForensics (2024)
- **Size**: 1.2M images
- **Features**: Diffusion-step metadata
- **Link**: [IEEE](https://ieeexplore.ieee.org/document/10373498)

---

## Social Media & In-the-Wild Datasets

### SID-Set / SIDA (2025)
- **Size**: 300K images
- **Features**: Social media AI images with localization
- **Paper**: [arXiv:2412.04292](https://arxiv.org/abs/2412.04292)
- **Link**: [GitHub](https://github.com/hzlsaber/SIDA)

### OpenFake (2025)
- **Size**: 963K + 3M images
- **Features**: Human-indistinguishable quality
- **Paper**: [arXiv:2509.09495](https://arxiv.org/abs/2509.09495)
- **Additional**: [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/)

### Deepfake-Eval-2024 (2025)
- **Size**: In-the-wild
- **Features**: Multi-modal benchmark of deepfakes circulated in 2024
- **Paper**: [arXiv:2503.02857](https://arxiv.org/abs/2503.02857)

### Community Forensics (2025)
- **Size**: 1000+ generators
- **Features**: Large generator diversity
- **Paper**: [arXiv:2411.04125](https://arxiv.org/abs/2411.04125)
- **Link**: [Dataset Download](https://jespark.net/projects/2024/community_forensics/download_dataset.html)

---

## Benchmark Datasets

### FFW (2022)
- **Size**: 130K images
- **Features**: Fake faces in the wild

### CDDB (2022)
- **Size**: 500K images
- **Features**: Cross-domain benchmark
- **Link**: [CVPR 2022 Workshop](https://openaccess.thecvf.com/content/CVPR2022W/)

---

## Download Script

See `download_datasets.sh` for automated download links (where publicly available).
