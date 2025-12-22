# 📊 Deepfake Detection Datasets

This directory contains information about datasets used in deepfake detection research.

## Face Manipulation Datasets

### FaceForensics++ (2019)
- **Size**: 1,000 original videos, 4,000 manipulated videos (~1.8M frames)
- **Manipulation Types**: DeepFakes, Face2Face, FaceSwap, NeuralTextures
- **Quality Levels**: c0 (raw), c23 (light compression), c40 (heavy compression)
- **Link**: [GitHub](https://github.com/ondyari/FaceForensics)

### Celeb-DF v2 (2020)
- **Size**: 590 real videos, 5,639 deepfake videos
- **Quality**: High-quality face swaps with reduced visual artifacts
- **Link**: [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics)

### DFDC (2020)
- **Size**: 128,154 videos (100,000+ unique subjects)
- **Manipulation Types**: Multiple deepfake generation methods
- **Link**: [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge)

### DeeperForensics-1.0 (2020)
- **Size**: 60,000 videos
- **Features**: Diverse perturbations (compression, blur, noise)
- **Link**: [GitHub](https://github.com/EndlessSora/DeeperForensics-1.0)

### WildDeepfake (2020)
- **Size**: 7,314 face sequences from 707 videos
- **Features**: Real-world deepfakes collected from the internet
- **Link**: [GitHub](https://github.com/deepfakeinthewild/deepfake-in-the-wild)

## Synthetic Image Datasets

### GenImage (2023)
- **Size**: 1.35M images
- **Generators**: Stable Diffusion, Midjourney, DALL-E, GLIDE, etc.
- **Link**: [GitHub](https://github.com/GenImage-Dataset/GenImage)

### DiffusionDB (2023)
- **Size**: 14M images
- **Generator**: Stable Diffusion with prompts
- **Link**: [GitHub](https://github.com/poloclub/diffusiondb)

## Download Script

See `download_datasets.sh` for automated download links (where publicly available).
