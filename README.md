<p align="center">
  <h1 align="center">🎭 Awesome Deepfake Detection Survey</h1>
  <p align="center">
    <strong>Comprehensive Review of Image-Based Deepfake Detection Techniques and Datasets</strong>
  </p>
  <p align="center">
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
    <img src="https://img.shields.io/badge/Papers-150+-blue" alt="Papers">
    <img src="https://img.shields.io/badge/Datasets-50+-green" alt="Datasets">
    <img src="https://img.shields.io/badge/Code-30+-orange" alt="Code">
  </p>
  <p align="center">
    <a href="#-datasets">Datasets</a> •
    <a href="#-detection-methods">Methods</a> •
    <a href="#-implementations">Code</a> •
    <a href="REFERENCES.md">Citations</a>
  </p>
</p>

---

## 📖 About

> **"Comprehensive Review of Image-Based Deepfake Detection Techniques and Datasets"**  
> *Authors: Hardik Sharma, Sachin Chaudhary, Praful Hambarde, Akshay Dudhane*

📧 **Contact:** [hhardikssharma@gmail.com](mailto:hhardikssharma@gmail.com)

**Related Resources:** [Awesome-Comprehensive-Deepfake-Detection](https://github.com/qiqitao77/Awesome-Comprehensive-Deepfake-Detection)

---

## 📑 Contents

- [Datasets](#-datasets)
  - [Video Deepfake Datasets](#-video-deepfake-datasets)
  - [Image Deepfake Datasets](#️-image-deepfake-datasets)
  - [Diffusion-Generated Datasets](#-diffusion-generated-datasets)
  - [Audio Deepfake Datasets](#-audio-deepfake-datasets)
- [Detection Methods](#-detection-methods)
  - [Visual Artifact Detection](#-visual-artifact-detection)
  - [Frequency-Based Detection](#-frequency-based-detection)
  - [Transformer-Based Detection](#-transformer-based-detection)
  - [VLM-Based Detection](#-vlm-based-detection)
  - [Temporal Consistency Detection](#-temporal-consistency-detection)
  - [Data Augmentation Methods](#-data-augmentation-methods)
- [Implementations](#️-implementations)
- [Benchmarks](#-benchmarks)

---

## 📊 Datasets

### 🎬 Video Deepfake Datasets

| Dataset | Year | Real | Fake | Methods | Link |
|---------|:----:|-----:|-----:|---------|:----:|
| **FaceForensics++** | 2019 | 1,000 | 4,000 | DeepFakes, Face2Face, FaceSwap, NeuralTextures | [GitHub](https://github.com/ondyari/FaceForensics) |
| **DFDC** | 2020 | 23,654 | 104,500 | Multiple methods | [Meta AI](https://ai.meta.com/datasets/dfdc/) |
| **Celeb-DF v2** | 2020 | 590 | 5,639 | DeepFake | [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **Celeb-DF++** | 2024 | — | — | Extended benchmark | [GitHub](https://github.com/OUC-VAS/Celeb-DF-PP) |
| **DeeperForensics-1.0** | 2020 | 50,000 | 10,000 | DF-VAE | [GitHub](https://github.com/EndlessSora/DeeperForensics-1.0) |
| **WildDeepfake** | 2020 | 3,805 | 3,509 | Real-world | [GitHub](https://github.com/deepfakeinthewild/deepfake-in-the-wild) |
| **ForgeryNet** | 2021 | 99,630 | 121,617 | 15 methods | [Project](https://yinanhe.github.io/projects/forgerynet.html) |
| **KoDF** | 2021 | 62,166 | 175,776 | 6 methods | [Project](https://moneybrain-research.github.io/kodf/) |
| **FFIW-10K** | 2021 | 10,000 | 10,000 | Face Forensics in the Wild | [GitHub](https://github.com/tfzhou/FFIW) |
| **DF-Platter** | 2023 | — | — | Multi-face heterogeneous | [Project](https://iab-rubric.org/df-platter-database) |
| **DFDM** | 2023 | — | — | Different models | [Drive](https://drive.google.com/drive/folders/1aXxeMdA2qwjDytyIgr4CBMVy4pAWizdX) |
| **Deepfake-Eval-2024** | 2024 | — | — | Evaluation benchmark | [HuggingFace](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024) |

### 🖼️ Image Deepfake Datasets

| Dataset | Year | Size | Key Features | Link |
|---------|:----:|-----:|--------------|:----:|
| **SID-Set (SIDA)** 🆕 | 2025 | 300K | Social media, GT masks, CVPR 2025 | [HuggingFace](https://huggingface.co/datasets/saberzl/SID_Set) |
| **DFFD** | 2020 | 299K | Multiple GAN types | [MSU](http://cvlab.cse.msu.edu/dffd-dataset.html) |
| **GenImage** | 2023 | 1.3M | 8 generators (SD, Midjourney, DALL-E) | [GitHub](https://github.com/GenImage-Dataset/GenImage) |
| **CelebA-Spoof** | 2020 | 671K | 10 spoof types, 40 attributes | [GitHub](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) |
| **OpenForensics** | 2021 | 115K | Open-source benchmark | [GitHub](https://github.com/ltnghia/openforensics) |
| **CIFAKE** | 2023 | 120K | CIFAR-style fake images | [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) |
| **iFakeFaceDB** | 2020 | 87K | GAN-generated faces | [GitHub](https://github.com/socialabubi/iFakeFaceDB) |
| **100K-Faces** | 2019 | 100K | StyleGAN generated | [Generated Photos](https://generated.photos) |

### 🌀 Diffusion-Generated Datasets

| Dataset | Year | Key Features | Link |
|---------|:----:|--------------|:----:|
| **DiffusionFace** | 2024 | Diffusion-based forgery analysis | [GitHub](https://github.com/Rapisurazurite/DiffFace) |
| **DiFF** | 2024 | Diffusion facial forgery | [GitHub](https://github.com/xaCheng1996/DiFF) |
| **DeepFakeFace** | 2023 | Diffusion model robustness | [GitHub](https://github.com/OpenRL-Lab/DeepFakeFace) |
| **Diffusion Deepfake** | 2024 | Surrey deepfake dataset | [Project](https://surrey-uplab.github.io/research/diffusion_deepfake/) |
| **TalkingHeadBench** | 2025 | Diffusion talking head | [HuggingFace](https://huggingface.co/datasets/luchaoqi/TalkingHeadBench) |

### 🔊 Audio Deepfake Datasets

| Dataset | Year | Size | Key Features | Link |
|---------|:----:|-----:|--------------|:----:|
| **ASVspoof 2019 LA** | 2019 | 148K | Logical access attacks | [ASVspoof](https://www.asvspoof.org/) |
| **ASVspoof5** | 2024 | 1M+ | Crowdsourced, adversarial | [Zenodo](https://zenodo.org/record/asvspoof5) |
| **FakeAVCeleb** | 2022 | — | Audio-visual celebrity | [GitHub](https://github.com/DASH-Lab/FakeAVCeleb) |
| **WaveFake** | 2021 | 134K | TTS vocoders | [Zenodo](https://zenodo.org/record/wavefake) |
| **MLAAD** | 2024 | 201K | 40 languages, 101 TTS | [HuggingFace](https://huggingface.co/datasets/mlaad) |

---

## 🔬 Detection Methods

### 👁️ Visual Artifact Detection

#### CVPR 2025
1. **SIDA**: Social Media Image Deepfake Detection with LMM [Paper](https://github.com/hzlsaber/SIDA) | [Code](https://github.com/hzlsaber/SIDA)
2. **D3**: Scaling Up Deepfake Detection by Learning from Discrepancy [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_D3_Scaling_Up_Deepfake_Detection_by_Learning_from_Discrepancy_CVPR_2025_paper.pdf)
3. Face Forgery Video Detection via Temporal Forgery Cue Unraveling [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Face_Forgery_Video_Detection_via_Temporal_Forgery_Cue_Unraveling_CVPR_2025_paper.pdf)
4. Generalizing Deepfake with Plug-and-Play Video-Level Blending [Paper](https://arxiv.org/pdf/2408.17065)
5. **Forensics Adapter**: Adapting CLIP for Generalizable Face Forgery Detection [Paper](https://arxiv.org/abs/2411.19715)
6. Towards Universal Synthetic Video Detector [Paper](https://arxiv.org/pdf/2412.12278)

#### CVPR 2024
1. **LAA-Net**: Localized Artifact Attention Network [Paper](https://arxiv.org/abs/2401.13856)
2. Rethinking Up-Sampling Operations in CNN-based Generative Network [Paper](https://arxiv.org/abs/2312.10461)
3. Transcending Forgery Specificity with Latent Space Augmentation [Paper](https://arxiv.org/abs/2311.11278)
4. Exploiting Style Latent Flows for Generalizing Deepfake Video Detection [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Choi_Exploiting_Style_Latent_Flows_for_Generalizing_Deepfake_Video_Detection_CVPR_2024_paper.html)

#### ICCV 2025
1. **FakeRadar**: Probing Forgery Outliers to Detect Unknown Deepfake Videos [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Li_FakeRadar_Probing_Forgery_Outliers_to_Detect_Unknown_Deepfake_Videos_ICCV_2025_paper.html)
2. **DeepShield**: Fortifying Deepfake Video Detection [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Cai_DeepShield_Fortifying_Deepfake_Video_Detection_with_Local_and_Global_Forgery_Analysis_ICCV_2025_paper.html)
3. **FakeSTormer**: Vulnerability-Aware Spatio-Temporal Learning [Paper](https://arxiv.org/pdf/2501.01184) | [Code](https://github.com/10Ring/FakeSTormer)
4. **PwTF-DVD**: Pixel-wise Temporal Frequency-based Detection [Paper](https://arxiv.org/abs/2507.02398) | [Code](https://github.com/rama0126/PwTF-DVD)
5. **AdvOU**: Open-Unfairness Adversarial Mitigation [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Open-Unfairness_Adversarial_Mitigation_for_Generalized_Deepfake_Detection_ICCV_2025_paper.html) | [Code](https://github.com/liacaaa/AdvOU)

#### ICCV 2023
1. **SeeABLE**: Soft Discrepancies and Bounded Contrastive Learning [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Larue_SeeABLE_Soft_Discrepancies_and_Bounded_Contrastive_Learning_for_Exposing_Deepfakes_ICCV_2023_paper.pdf)
2. **TALL**: Thumbnail Layout for Deepfake Video Detection [Paper](https://arxiv.org/abs/2307.07494)

#### CVPR 2022-2023
1. Detecting Deepfakes with Self-Blended Images [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.pdf)
2. Self-supervised Learning of Adversarial Example [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Self-Supervised_Learning_of_Adversarial_Example_Towards_Good_Generalizations_for_Deepfake_CVPR_2022_paper.pdf)
3. **AltFreezing**: More General Video Face Forgery Detection [Paper](https://arxiv.org/abs/2307.08317)
4. Dynamic Graph Learning with Content-guided Spatial-Frequency [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Dynamic_Graph_Learning_With_Content-Guided_Spatial-Frequency_Relation_Reasoning_for_Deepfake_CVPR_2023_paper.pdf)

#### CVPR 2021
1. **Multi-Attentional Deepfake Detection** [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Multi-Attentional_Deepfake_Detection_CVPR_2021_paper.pdf)
2. Representative Forgery Mining for Fake Face Detection [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Representative_Forgery_Mining_for_Fake_Face_Detection_CVPR_2021_paper.pdf)
3. Lips Don't Lie: A Generalisable and Robust Approach [Paper](https://arxiv.org/abs/2012.07657)

---

### 📊 Frequency-Based Detection

#### CVPR 2025
1. **FreqDebias**: Consistency-Driven Frequency Debiasing [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.pdf)

#### 2024-2025
1. [AAAI 2024] Frequency-Aware Deepfake Detection [Paper](https://arxiv.org/abs/2403.07240)
2. [ICASSP 2024] Frequency Masking for Universal Deepfake Detection [Paper](https://arxiv.org/abs/2401.06506)
3. [NeurIPS 2024] **FreqBlender**: Enhancing DeepFake Detection [Paper](https://arxiv.org/abs/2404.13872)
4. [ACM MM 2025] **SpecXNet**: Dual-Domain Convolutional Network [Paper](https://arxiv.org/abs/2509.22070)
5. [arXiv 2025] **WMamba**: Wavelet-based Mamba for Face Forgery [Paper](https://arxiv.org/abs/2501.09617)
6. [WACV 2025] Wavelet-Driven Generalizable Framework [Paper](https://arxiv.org/pdf/2409.18301)

#### Classic Methods
1. [ECCV 2020] **F3-Net**: Thinking in Frequency [Paper](https://arxiv.org/abs/2007.09355)
2. [CVPR 2021] Spatial-Phase Shallow Learning [Paper](https://arxiv.org/abs/2103.01856)
3. [CVPR 2021] Generalizing Face Forgery with High-frequency Features [Paper](https://arxiv.org/abs/2103.12376)
4. [AAAI 2022] **FrePGAN**: Frequency-level Perturbations [Paper](https://arxiv.org/abs/2202.03347)
5. [AAAI 2022] **ADD**: Frequency Attention and Multi-View KD [Paper](https://arxiv.org/abs/2112.03553)

---

### 🤖 Transformer-Based Detection

#### CVPR/ICCV 2025
1. Towards More General Video-based Deepfake Detection [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Towards_More_General_Video-based_Deepfake_Detection_through_Facial_Component_Guided_CVPR_2025_paper.pdf)
2. [AAAI 2025] Token-Level Shuffling and Mixing [Paper](https://arxiv.org/pdf/2501.04376)
3. [AAAI 2025] Standing on Shoulders of Giants: Reprogramming VLM [Paper](https://arxiv.org/pdf/2409.02664)
4. [WACV 2026] Deepfake Detection that Generalizes Across Benchmarks [Paper](https://arxiv.org/abs/2508.06248) | [Code](https://github.com/yermandy/GenD)

#### 2024
1. [arXiv 2024] **FakeFormer**: Efficient Vulnerability-Driven Transformers [Paper](https://arxiv.org/pdf/2410.21964)
2. [arXiv 2024] Guided and Fused: Efficient Frozen CLIP-ViT [Paper](https://arxiv.org/pdf/2408.13697)
3. [arXiv 2024] Open-Set Deepfake Detection with Forgery Style Mixture [Paper](https://arxiv.org/abs/2408.12791v1)
4. [arXiv 2024] A Timely Survey on Vision Transformer for Deepfake Detection [Paper](https://arxiv.org/abs/2405.08463)
5. [arXiv 2024] Mixture of Low-rank Experts for AI-Generated Image Detection [Paper](https://arxiv.org/abs/2404.04883)

#### 2023
1. [CVPR 2023] **AUNet**: Learning Relations Between Action Units [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Bai_AUNet_Learning_Relations_Between_Action_Units_for_Face_Forgery_Detection_CVPR_2023_paper.pdf)
2. [ACM MM 2023] **UMMAFormer**: Universal Multimodal-adaptive Transformer [Paper](https://dl.acm.org/doi/abs/10.1145/3581783.3613767)
3. [MIPR 2023] Enhancing Face Forgery Detection via ViT with Low-Rank Adaptation [Paper](https://ieeexplore.ieee.org/document/10254409)

---

### 🧠 VLM-Based Detection

#### 2025
1. [CVPR 2025] Rethinking VLM in Face Forensics: Multi-Modal Interpretable Forged Face Detector [Paper](https://arxiv.org/pdf/2503.20188)
2. [ICML 2025] Unlocking VLM Capabilities for Generalizable and Explainable Deepfake Detection [Paper](https://arxiv.org/pdf/2503.14853)
3. [ICLR 2025] **FakeShield**: Explainable Image Forgery Detection with MLLM [Paper](https://arxiv.org/pdf/2410.02761)
4. [IJCAI 2025] **CorrDetail**: Visual Detail Enhanced Self-Correction [Paper](https://arxiv.org/pdf/2507.05302)
5. [ICMLW 2025] Visual Language Models as Zero-Shot Deepfake Detectors [Paper](https://arxiv.org/pdf/2507.22469)
6. [arXiv 2025] **TruthLens**: A Training-Free Paradigm [Paper](https://arxiv.org/pdf/2503.15342)
7. [arXiv 2025] Unlocking Hidden Potential of CLIP [Paper](https://arxiv.org/pdf/2503.19683) | [Code](https://github.com/yermandy/deepfake-detection)
8. [arXiv 2025] MLLM-Enhanced Face Forgery Detection [Paper](https://arxiv.org/pdf/2505.02013)

#### 2024
1. [CVPR Workshop 2024] Can Multi-Modal LLMs Work as Deepfake Detectors [Paper](https://arxiv.org/pdf/2503.20084)
2. [arXiv 2024] Understanding Training-Free AI-Generated Image Detections [Paper](https://arxiv.org/pdf/2411.19117)

---

### ⏱️ Temporal Consistency Detection

1. [NeurIPS 2025] **SpeechForensics**: Audio-Visual Speech Representation [Paper](https://arxiv.org/pdf/2508.09913)
2. [CVPR 2025] Face Forgery Video Detection via Temporal Forgery Cue [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Face_Forgery_Video_Detection_via_Temporal_Forgery_Cue_Unraveling_CVPR_2025_paper.pdf)
3. [ECCV 2024] Learning Natural Consistency Representation [Paper](https://arxiv.org/abs/2407.10550v1)
4. [IJCV 2024] Learning Spatiotemporal Inconsistency via Thumbnail Layout [Paper](https://arxiv.org/abs/2403.10261)
5. [WACV 2023] **TI2Net**: Temporal Identity Inconsistency Network [Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_TI2Net_Temporal_Identity_Inconsistency_Network_for_Deepfake_Detection_WACV_2023_paper.pdf)
6. [ICCV 2021] Exploring Temporal Coherence for More General Detection [Paper](https://arxiv.org/abs/2108.06693)
7. [ACM MM 2020] **DeepRhythm**: Attentional Visual Heartbeat Rhythms [Paper](https://arxiv.org/abs/2006.07634)
8. [WIFS 2018] In Ictu Oculi: Exposing AI Fake Videos by Eye Blinking [Paper](https://ieeexplore.ieee.org/document/8630787/)

---

### 🔄 Data Augmentation Methods

1. [CVPR 2025] D3: Scaling Up Deepfake Detection by Learning from Discrepancy [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_D3_Scaling_Up_Deepfake_Detection_by_Learning_from_Discrepancy_CVPR_2025_paper.pdf)
2. [CVPR 2025] Plug-and-Play: Video-Level Blending and Spatiotemporal Adapter [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_Generalizing_Deepfake_Video_Detection_with_Plug-and-Play_Video-Level_Blending_and_Spatiotemporal_CVPR_2025_paper.pdf)
3. [WACV 2025] **DiffFake**: Differential Anomaly Detection [Paper](https://arxiv.org/pdf/2502.16247)
4. [ECCV 2024] Fake It till You Make It: Curricular Dynamic Forgery Augmentations [Paper](https://arxiv.org/pdf/2409.14444)
5. [arXiv 2024] FSBI: Frequency Enhanced Self-Blended Images [Paper](https://arxiv.org/html/2406.08625v1)

---

## 🛠️ Implementations

| Method | Venue | Code |
|--------|-------|:----:|
| **SIDA** | CVPR 2025 | [hzlsaber/SIDA](https://github.com/hzlsaber/SIDA) |
| **GenD** | WACV 2026 | [yermandy/GenD](https://github.com/yermandy/GenD) |
| **FakeSTormer** | ICCV 2025 | [10Ring/FakeSTormer](https://github.com/10Ring/FakeSTormer) |
| **PwTF-DVD** | ICCV 2025 | [rama0126/PwTF-DVD](https://github.com/rama0126/PwTF-DVD) |
| **D4** | WACV 2024 | [nmangaokar/wacv_24_d4](https://github.com/nmangaokar/wacv_24_d4) |
| **LFM** | BMVC 2025 | [lmlpy/LFM](https://github.com/lmlpy/LFM) |
| **DFS-GDD** | BMVC 2024 | [leaffeall/DFS-GDD](https://github.com/leaffeall/DFS-GDD) |
| **DeepfakeBench** | — | [SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) |
| **FaceForensics++** | ICCV 2019 | [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) |
| **F3-Net** | ECCV 2020 | [yyk-wew/F3-Net](https://github.com/yyk-wew/F3-Net) |
| **DIRE** | ICCV 2023 | [ZhendongWang6/DIRE](https://github.com/ZhendongWang6/DIRE) |
| **UnivFD** | CVPR 2023 | [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) |
| **CLIP Deepfake** | arXiv 2025 | [yermandy/deepfake-detection](https://github.com/yermandy/deepfake-detection) |

---

## 📈 Benchmarks

### Cross-Dataset Performance (AUC %)

| Method | FF++ (c23) | Celeb-DF | DFDC | Year |
|--------|:----------:|:--------:|:----:|:----:|
| MesoNet | 83.1 | 70.5 | 65.2 | 2018 |
| XceptionNet | 95.7 | 73.4 | 70.8 | 2019 |
| F3-Net | 97.5 | 76.3 | 72.1 | 2020 |
| Multi-Attention | 97.6 | 79.2 | 73.5 | 2021 |
| SFANet | 99.2 | 91.1 | 81.4 | 2022 |

---

## ✍️ Citation

```bibtex
@article{deepfakesurvey2025,
  title   = {Comprehensive Review of Image-Based Deepfake Detection Techniques and Datasets},
  author  = {Hardik Sharma and Sachin Chaudhary and Praful Hambarde and Akshay Dudhane},
  year    = {2025}
}
```

📖 **Full citations:** [REFERENCES.md](REFERENCES.md)

---

## 🤝 Contributing

Contributions welcome! Add new papers/datasets via Pull Request.

---

<p align="center">
  <b>Last Updated:</b> December 2025
</p>
