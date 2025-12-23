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
  <img src="https://img.shields.io/badge/Papers-80+-blue" alt="Papers">
  <img src="https://img.shields.io/badge/Datasets-40+-green" alt="Datasets">
  <img src="https://img.shields.io/badge/GitHub%20Repos-25+-orange" alt="Repos">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
</p>

---

## 📖 About

This repository accompanies our survey paper:

> **"Comprehensive Review of Image-Based Deepfake Detection Techniques and Datasets"**  
> *Authors: Hardik Sharma, Sachin Chaudhary, Praful Hambarde, Akshay Dudhane*

A curated collection of deepfake detection literature, datasets, and implementations covering **image**, **video**, and **audio** modalities.

📧 **Contact:** [hhardikssharma@gmail.com](mailto:hhardikssharma@gmail.com)

**📚 Related Resources:**
- [Awesome-Comprehensive-Deepfake-Detection](https://github.com/qiqitao77/Awesome-Comprehensive-Deepfake-Detection) - Comprehensive paper list

---

## 📊 Datasets

### 🎬 Video Deepfake Datasets

| Dataset | Year | Real | Fake | Key Features | Link |
|---------|:----:|-----:|-----:|--------------|:----:|
| **FaceForensics++ (FF++)** | 2019 | 1,000 | 4,000 | 4 manipulation methods, 3 quality levels | [🔗](https://github.com/ondyari/FaceForensics) |
| **DFDC** | 2020 | 23,654 | 104,500 | Largest challenge dataset | [🔗](https://ai.facebook.com/datasets/dfdc/) |
| **Celeb-DF (v2)** | 2020 | 590 | 5,639 | High-quality celebrity swaps | [🔗](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **Celeb-DF++** | 2024 | — | — | Extended Celeb-DF benchmark | [🔗](https://github.com/OUC-VAS/Celeb-DF-PP) |
| **DeeperForensics-1.0** | 2020 | 50,000 | 10,000 | 7 distortion types | [🔗](https://github.com/EndlessSora/DeeperForensics-1.0) |
| **WildDeepfake** | 2020 | 3,805 | 3,509 | Real-world internet deepfakes | [🔗](https://github.com/deepfakeinthewild/deepfake-in-the-wild) |
| **ForgeryNet** | 2021 | 99,630 | 121,617 | 15 methods, 8 types | [🔗](https://yinanhe.github.io/projects/forgerynet.html) |
| **KoDF** | 2021 | 62,166 | 175,776 | Korean celebrities | [🔗](https://moneybrain-research.github.io/kodf/) |
| **FFIW-10K** | 2021 | 10,000 | 10,000 | Face forensics in the wild | [🔗](https://github.com/tfzhou/FFIW) |
| **DF-Platter** | 2023 | — | — | Multi-face heterogeneous | [🔗](https://iab-rubric.org/df-platter-database) |
| **DFDM** | 2023 | — | — | Deepfakes from different models | [🔗](https://drive.google.com/drive/folders/1aXxeMdA2qwjDytyIgr4CBMVy4pAWizdX) |
| **Deepfake-Eval-2024** | 2024 | — | — | Latest evaluation benchmark | [🔗](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024) |

### 🖼️ Image Deepfake Datasets

| Dataset | Year | Size | Key Features | Link |
|---------|:----:|-----:|--------------|:----:|
| **SID-Set (SIDA)** 🆕 | 2025 | 300K | Social media deepfakes with GT masks, CVPR 2025 | [🔗](https://huggingface.co/datasets/saberzl/SID_Set) |
| **DFFD** | 2020 | 299K | Multiple GAN types | [🔗](http://cvlab.cse.msu.edu/dffd-dataset.html) |
| **GenImage** | 2023 | 1.3M | 8 generators (SD, Midjourney, DALL-E) | [🔗](https://github.com/GenImage-Dataset/GenImage) |
| **CelebA-Spoof** | 2020 | 671K | 10 spoof types, 40 attributes | [🔗](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) |
| **DiffusionFace** | 2024 | — | Diffusion-based forgery analysis | [🔗](https://github.com/Rapisurazurite/DiffFace) |
| **DeepFakeFace** | 2023 | — | Diffusion model robustness | [🔗](https://github.com/OpenRL-Lab/DeepFakeFace) |
| **DiFF** | 2024 | — | Diffusion facial forgery | [🔗](https://github.com/xaCheng1996/DiFF) |
| **CIFAKE** | 2023 | 120K | CIFAR-style fake images | [🔗](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) |
| **OpenForensics** | 2021 | 115K | Open-source benchmark | [🔗](https://github.com/ltnghia/openforensics) |

### 🔊 Audio Deepfake Datasets

| Dataset | Year | Size | Key Features | Link |
|---------|:----:|-----:|--------------|:----:|
| **ASVspoof 2019 LA** | 2019 | 148K | Logical access attacks | [🔗](https://www.asvspoof.org/) |
| **ASVspoof5** | 2024 | 1M+ | Crowdsourced, adversarial | [🔗](https://zenodo.org/record/asvspoof5) |
| **FakeAVCeleb** | 2022 | — | Audio-visual celebrity fakes | [🔗](https://github.com/DASH-Lab/FakeAVCeleb) |
| **WaveFake** | 2021 | 134K | TTS vocoders | [🔗](https://zenodo.org/record/wavefake) |
| **MLAAD** | 2024 | 201K | 40 languages, 101 TTS models | [🔗](https://huggingface.co/datasets/mlaad) |

---

## 🔬 Detection Methods

### 🆕 CVPR 2025 — Latest Methods

| Paper | Method | Code | Status |
|-------|--------|:----:|:------:|
| **SIDA**: Social Media Image Deepfake Detection with LMM | Detection + Localization + Explanation | [GitHub](https://github.com/hzlsaber/SIDA) | ✅ |
| **D3**: Scaling Up Deepfake Detection by Learning from Discrepancy | Data augmentation | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_D3_Scaling_Up_Deepfake_Detection_by_Learning_from_Discrepancy_CVPR_2025_paper.pdf) | — |
| Face Forgery Video Detection via Temporal Forgery Cue | Temporal analysis | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Face_Forgery_Video_Detection_via_Temporal_Forgery_Cue_Unraveling_CVPR_2025_paper.pdf) | — |
| Generalizing Deepfake with Plug-and-Play | Video-level blending | [Paper](https://arxiv.org/pdf/2408.17065) | — |

### 🆕 ICCV 2025

| Paper | Method | Code | Status |
|-------|--------|:----:|:------:|
| FakeRadar: Probing Forgery Outliers | Forgery outlier detection | — | ❌ |
| DeepShield: Local and Global Forgery Analysis | Multi-scale analysis | — | ❌ |
| **FakeSTormer**: Vulnerability-Aware Learning | Transformer + temporal | [GitHub](https://github.com/10Ring/FakeSTormer) | ✅ |
| **PwTF-DVD**: Pixel-wise Temporal Frequency | Frequency analysis | [GitHub](https://github.com/rama0126/PwTF-DVD) | ✅ |
| **AdvOU**: Open-Unfairness Mitigation | Fairness-aware | [GitHub](https://github.com/liacaaa/AdvOU) | ⏳ |

### 📅 CVPR 2024

| Paper | Method | Code | Status |
|-------|--------|:----:|:------:|
| **LAA-Net**: Localized Artifact Attention | Quality-agnostic detection | [Paper](https://arxiv.org/abs/2401.13856) | — |
| Rethinking Up-Sampling in CNN | Generalizable detection | [Paper](https://arxiv.org/abs/2312.10461) | — |
| Exploiting Style Latent Flows | Video detection | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Choi_Exploiting_Style_Latent_Flows_for_Generalizing_Deepfake_Video_Detection_CVPR_2024_paper.html) | — |

### 📅 WACV 2024-2025

| Paper | Method | Code | Status |
|-------|--------|:----:|:------:|
| **D4**: Adversarial Diffusion Deepfakes | Ensemble detection | [GitHub](https://github.com/nmangaokar/wacv_24_d4) | ✅ |
| **DiffFake**: Differential Anomaly Detection | Anomaly-based | [Paper](https://arxiv.org/pdf/2502.16247) | — |
| Improving Fairness in Deepfake Detection | Fair detection | [GitHub](https://github.com/littlejuyan/DF_Fairness) | ⏳ |

### 🎯 BMVC 2024-2025

| Paper | Method | Code | Status |
|-------|--------|:----:|:------:|
| **LFM**: Local Focusing Mechanism | Attention | [GitHub](https://github.com/lmlpy/LFM) | ✅ |
| **DFS-GDD**: Decoupling Forgery Semantics | Feature decoupling | [GitHub](https://github.com/leaffeall/DFS-GDD) | ✅ |

### 📚 Classic Methods

| Method | Year | Type | Paper |
|--------|:----:|------|:-----:|
| MesoNet | 2018 | CNN | [🔗](https://doi.org/10.1109/WIFS.2018.8630761) |
| FaceForensics++ (XceptionNet) | 2019 | CNN | [🔗](https://arxiv.org/abs/1901.08971) |
| Face X-Ray | 2020 | Blending | [🔗](https://doi.org/10.1109/CVPR42600.2020.00505) |
| F3-Net | 2020 | Frequency | [🔗](https://arxiv.org/abs/2007.09355) |
| Multi-Attention | 2021 | Attention | [🔗](https://doi.org/10.1109/CVPR46437.2021.00188) |
| DIRE | 2023 | Diffusion | [🔗](https://arxiv.org/abs/2303.09295) |

---

## 🛠️ Implementations

| Method | Conference | Framework | GitHub | Status |
|--------|------------|-----------|--------|:------:|
| **SIDA** 🆕 | CVPR 2025 | PyTorch | [hzlsaber/SIDA](https://github.com/hzlsaber/SIDA) | ✅ |
| DeepfakeBench | — | PyTorch | [SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) | ✅ |
| FaceForensics++ | ICCV 2019 | PyTorch | [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) | ✅ |
| F3-Net | ECCV 2020 | PyTorch | [yyk-wew/F3-Net](https://github.com/yyk-wew/F3-Net) | ✅ |
| DIRE | ICCV 2023 | PyTorch | [ZhendongWang6/DIRE](https://github.com/ZhendongWang6/DIRE) | ✅ |
| UnivFD | CVPR 2023 | PyTorch | [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) | ✅ |
| FakeSTormer | ICCV 2025 | PyTorch | [10Ring/FakeSTormer](https://github.com/10Ring/FakeSTormer) | ✅ |
| PwTF-DVD | ICCV 2025 | PyTorch | [rama0126/PwTF-DVD](https://github.com/rama0126/PwTF-DVD) | ✅ |
| D4 | WACV 2024 | PyTorch | [nmangaokar/wacv_24_d4](https://github.com/nmangaokar/wacv_24_d4) | ✅ |
| LFM | BMVC 2025 | PyTorch | [lmlpy/LFM](https://github.com/lmlpy/LFM) | ✅ |
| DFS-GDD | BMVC 2024 | PyTorch | [leaffeall/DFS-GDD](https://github.com/leaffeall/DFS-GDD) | ✅ |

---

## 📈 Benchmarks

### Cross-Dataset Performance (AUC %)

| Method | FF++ (c23) | Celeb-DF | DFDC |
|--------|:----------:|:--------:|:----:|
| MesoNet | 83.1 | 70.5 | 65.2 |
| XceptionNet | 95.7 | 73.4 | 70.8 |
| F3-Net | 97.5 | 76.3 | 72.1 |
| Multi-Attention | 97.6 | 79.2 | 73.5 |
| SFANet | 99.2 | 91.1 | 81.4 |

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
  author  = {Hardik Sharma and Sachin Chaudhary and Praful Hambarde and Akshay Dudhane},
  year    = {2025}
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
