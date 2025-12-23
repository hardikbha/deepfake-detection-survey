<p align="center">
  <h1 align="center">🎭 Deepfake Detection Survey</h1>
  <p align="center">
    <strong>Comprehensive Review of Image-Based Deepfake Detection Techniques and Datasets</strong>
  </p>
  <p align="center">
    <a href="#-datasets">Datasets</a> •
    <a href="#-detection-methods">Methods</a> •
    <a href="#-implementations">Code</a> •
    <a href="#-benchmarks">Benchmarks</a> •
    <a href="REFERENCES.md">Citations</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Papers-50+-blue" alt="Papers">
  <img src="https://img.shields.io/badge/Datasets-30+-green" alt="Datasets">
  <img src="https://img.shields.io/badge/GitHub%20Repos-15+-orange" alt="Repos">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
</p>

---

## 📖 About

This repository accompanies our survey paper:

> **"Comprehensive Review of Image-Based Deepfake Detection Techniques and Datasets"**  
> *Submitted to IEEE Transactions on Information Forensics and Security, 2025*

A curated collection of deepfake detection literature, datasets, and implementations covering **image**, **video**, and **audio** modalities.

---

## 📊 Datasets

### 🎬 Video Deepfake Datasets

| Dataset | Year | Real | Fake | Key Features | Link |
|---------|:----:|-----:|-----:|--------------|:----:|
| **FaceForensics++ (FF++)** | 2019 | 1,000 | 4,000 | 4 manipulation methods, 3 quality levels (c0/c23/c40) | [🔗](https://github.com/ondyari/FaceForensics) |
| **DFDC** | 2020 | 23,654 | 104,500 | Largest challenge dataset, diverse subjects | [🔗](https://ai.facebook.com/datasets/dfdc/) |
| **Celeb-DF (v2)** | 2020 | 590 | 5,639 | High-quality celebrity swaps, reduced artifacts | [🔗](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **DeeperForensics-1.0** | 2020 | 50,000 | 10,000 | 7 distortion types, hidden test set | [🔗](https://github.com/EndlessSora/DeeperForensics-1.0) |
| **WildDeepfake** | 2020 | 3,805 | 3,509 | Real-world internet deepfakes | [🔗](https://github.com/deepfakeinthewild/deepfake-in-the-wild) |
| **ForgeryNet** | 2021 | 99,630 | 121,617 | 15 methods, 8 manipulation types | [🔗](https://yinanhe.github.io/projects/forgerynet.html) |
| **KoDF** | 2021 | 62,166 | 175,776 | Korean celebrities, 6 methods | [🔗](https://moneybrain-research.github.io/kodf/) |
| **FFIW-10K** | 2021 | 10,000 | 10,000 | Face forensics in the wild | — |
| **DF-Platter** | 2023 | — | — | Multi-generator platter | [🔗](https://github.com/AustralianAI/DF-Platter) |
| **Deepfake-Eval-2024** | 2024 | — | — | Latest evaluation benchmark | — |

### 🖼️ Image Deepfake Datasets

| Dataset | Year | Real | Fake | Key Features | Link |
|---------|:----:|-----:|-----:|--------------|:----:|
| **DFFD** | 2020 | 58,703 | 240,336 | Multiple GAN types | [🔗](http://cvlab.cse.msu.edu/dffd-dataset.html) |
| **iFakeFaceDB** | 2020 | — | 87,000 | GAN-generated faces | [🔗](https://github.com/socialabubi/iFakeFaceDB) |
| **100K-Faces** | 2019 | — | 100,000 | StyleGAN generated | [🔗](https://generated.photos) |
| **CelebA-Spoof** | 2020 | 202,599 | 468,882 | 10 spoof types, 40 attributes | [🔗](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) |
| **GenImage** | 2023 | 1,331,167 | — | 8 generators (SD, Midjourney, DALL-E) | [🔗](https://github.com/GenImage-Dataset/GenImage) |
| **ArtiFact** | 2023 | — | — | Artifact-focused dataset | — |
| **CIFAKE** | 2023 | 60,000 | 60,000 | CIFAR-style fake images | [🔗](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) |
| **OpenForensics** | 2021 | 45,473 | 69,881 | Open-source benchmark | [🔗](https://github.com/ltnghia/openforensics) |

### 🔊 Audio Deepfake Datasets

| Dataset | Year | Real | Fake | Key Features | Link |
|---------|:----:|-----:|-----:|--------------|:----:|
| **ASVspoof 2019 LA** | 2019 | 14,816 | 133,360 | Logical access attacks | [🔗](https://www.asvspoof.org/) |
| **ASVspoof 2021** | 2021 | — | — | Codec/transmission effects | [🔗](https://www.asvspoof.org/) |
| **ASVspoof5** | 2024 | 188,819 | 815,262 | Crowdsourced, adversarial attacks | [🔗](https://zenodo.org/record/asvspoof5) |
| **FakeAVCeleb** | 2022 | — | — | Audio-visual celebrity fakes | [🔗](https://github.com/DASH-Lab/FakeAVCeleb) |
| **WaveFake** | 2021 | 16,283 | 117,985 | TTS vocoders (MelGAN, WaveGlow) | [🔗](https://zenodo.org/record/wavefake) |
| **In the Wild** | 2022 | 19,963 | 11,816 | Real-world web audio | [🔗](https://deepfake-demo.owncloud.com/) |
| **MLAAD** | 2024 | 201,000 | — | 40 languages, 101 TTS models | [🔗](https://huggingface.co/datasets/mlaad) |

---

## 🔬 Detection Methods

### 🆕 ICCV 2025 — Latest Methods

| Paper | Method | Code | Status |
|-------|--------|:----:|:------:|
| FakeRadar: Probing Forgery Outliers | Forgery outlier detection | — | ❌ |
| DeepShield: Local and Global Forgery Analysis | Multi-scale analysis | — | ❌ |
| **FakeSTormer**: Vulnerability-Aware Spatio-Temporal Learning | Transformer + temporal | [GitHub](https://github.com/10Ring/FakeSTormer) | ✅ |
| Generalization-Preserved Learning | Continual learning | — | ❌ |
| **PwTF-DVD**: Pixel-wise Temporal Frequency | Frequency analysis | [GitHub](https://github.com/rama0126/PwTF-DVD) | ✅ |
| **AdvOU**: Open-Unfairness Adversarial Mitigation | Fairness-aware | [GitHub](https://github.com/liacaaa/AdvOU) | ⏳ |
| Audio-visual Synchronization | Cross-modal sync | [GitHub](https://github.com/AshutoshAnshul/ics-av-deepfake) | ⏳ |

### 📅 WACV 2024

| Paper | Method | Code | Status |
|-------|--------|:----:|:------:|
| Improving Fairness in Deepfake Detection | Fair detection | [GitHub](https://github.com/littlejuyan/DF_Fairness) | ⏳ |
| **D4**: Adversarial Diffusion Deepfakes | Ensemble detection | [GitHub](https://github.com/nmangaokar/wacv_24_d4) | ✅ |
| Motion Magnification for Deepfake Source | Motion analysis | — | ❌ |

### 🎯 BMVC 2024-2025

| Paper | Method | Code | Status |
|-------|--------|:----:|:------:|
| **LFM**: Local Focusing Mechanism | Attention mechanism | [GitHub](https://github.com/lmlpy/LFM) | ✅ |
| Unsupervised Multimodal Detection | Cross-modal | — | ❌ |
| **DFS-GDD**: Decoupling Forgery Semantics | Feature decoupling | [GitHub](https://github.com/leaffeall/DFS-GDD) | ✅ |

### 🏆 CVPR 2023

| Paper | Method | Code | Status |
|-------|--------|:----:|:------:|
| Implicit Identity Driven Detection | Identity analysis | — | — |
| Implicit Identity Leakage | Generalization study | — | — |

### 📚 Classic Methods

| Method | Year | Type | Paper |
|--------|:----:|------|-------|
| MesoNet | 2018 | CNN | [Link](https://doi.org/10.1109/WIFS.2018.8630761) |
| FaceForensics++ (XceptionNet) | 2019 | CNN | [Link](https://doi.org/10.1109/ICCV.2019.00009) |
| Capsule-Forensics | 2019 | Capsule Network | [Link](https://doi.org/10.1109/ICASSP.2019.8682602) |
| Face X-Ray | 2020 | Blending detection | [Link](https://doi.org/10.1109/CVPR42600.2020.00505) |
| F3-Net | 2020 | Frequency domain | [Link](https://arxiv.org/abs/2007.09355) |
| Multi-Attention | 2021 | Attention | [Link](https://doi.org/10.1109/CVPR46437.2021.00188) |
| SFANet | 2022 | Spatial-Frequency | [Link](https://doi.org/10.1109/TIFS.2022.3197678) |
| DIRE | 2023 | Diffusion-aware | [Link](https://arxiv.org/abs/2303.09295) |

---

## 🛠️ Implementations

### Available Code Repositories

| Method | Conference | Framework | GitHub | Stars |
|--------|------------|-----------|--------|:-----:|
| DeepfakeBench | — | PyTorch | [SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) | ⭐ |
| FaceForensics++ | ICCV 2019 | PyTorch | [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) | ⭐ |
| F3-Net | ECCV 2020 | PyTorch | [yyk-wew/F3-Net](https://github.com/yyk-wew/F3-Net) | ⭐ |
| DIRE | ICCV 2023 | PyTorch | [ZhendongWang6/DIRE](https://github.com/ZhendongWang6/DIRE) | ⭐ |
| UnivFD | CVPR 2023 | PyTorch | [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) | ⭐ |
| FakeSTormer | ICCV 2025 | PyTorch | [10Ring/FakeSTormer](https://github.com/10Ring/FakeSTormer) | ⭐ |
| PwTF-DVD | ICCV 2025 | PyTorch | [rama0126/PwTF-DVD](https://github.com/rama0126/PwTF-DVD) | ⭐ |
| D4 | WACV 2024 | PyTorch | [nmangaokar/wacv_24_d4](https://github.com/nmangaokar/wacv_24_d4) | ⭐ |
| LFM | BMVC 2025 | PyTorch | [lmlpy/LFM](https://github.com/lmlpy/LFM) | ⭐ |
| DFS-GDD | BMVC 2024 | PyTorch | [leaffeall/DFS-GDD](https://github.com/leaffeall/DFS-GDD) | ⭐ |

---

## 📈 Benchmarks

### Cross-Dataset Performance (AUC %)

| Method | FF++ (c23) | Celeb-DF | DFDC | WildDeepfake |
|--------|:----------:|:--------:|:----:|:------------:|
| MesoNet | 83.1 | 70.5 | 65.2 | — |
| XceptionNet | 95.7 | 73.4 | 70.8 | — |
| F3-Net | 97.5 | 76.3 | 72.1 | — |
| Multi-Attention | 97.6 | 79.2 | 73.5 | — |
| SFANet | 99.2 | 91.1 | 81.4 | — |

*c23 = light compression quality in FF++*

---

## 📁 Repository Structure

```
deepfake-detection-survey/
├── 📄 README.md              # This file
├── 📄 REFERENCES.md          # All BibTeX citations
├── 📄 QUESTIONS.md           # Missing info & TODO
├── 📂 datasets/              # Dataset documentation
├── 📂 benchmarks/            # Performance results
├── 📂 implementations/       # Code links
└── 📂 papers/                # Survey paper
```

---

## ✍️ Citation

```bibtex
@article{deepfakesurvey2025,
  title   = {Comprehensive Review of Image-Based Deepfake Detection Techniques and Datasets},
  author  = {Sachin Chaudhary and Praful Hambarde and Akshay Dudhane},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2025},
  note    = {Under Review}
}
```

📖 **Full citations available in [REFERENCES.md](REFERENCES.md)**

---

## 🤝 Contributing

Contributions welcome! Please:
- Add new papers/datasets via Pull Request
- Report issues or corrections
- Suggest improvements

---

## 📜 License

For academic purposes. Please cite our paper if you use this resource.

---

<p align="center">
  <b>Last Updated:</b> December 2025
</p>
