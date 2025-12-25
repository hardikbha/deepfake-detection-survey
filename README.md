<p align="center">
  <h1 align="center">🎭 Comprehensive Review of Image-Based Deepfake Detection</h1>
  <p align="center">
    <strong>Techniques and Datasets</strong>
  </p>
  <p align="center">
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
    <img src="https://img.shields.io/badge/Papers-300+-blue" alt="Papers">
    <img src="https://img.shields.io/badge/Datasets-50+-green" alt="Datasets">
    <img src="https://img.shields.io/badge/Code-80+-orange" alt="Code">
  </p>
  <p align="center">
    <a href="#-deepfake-generation-models">Generation</a> •
    <a href="#-taxonomy-of-manipulations">Taxonomy</a> •
    <a href="#-detection-approaches">Detection</a> •
    <a href="#-datasets--benchmarks">Datasets</a> •
    <a href="#-performance-benchmarks">Benchmarks</a>
  </p>
</p>

---

## 📖 About This Survey

> **"Comprehensive Review of Image-Based Deepfake Detection Techniques and Datasets"**  
> *Authors: Hardik Sharma, Sachin Chaudhary, Praful Hambarde, Akshay Dudhane*

This repository accompanies our survey paper on image-based deepfake detection, covering methods from **2018 to 2025**. The structure follows our paper's organization.

📧 **Contact:** [hhardikssharma@gmail.com](mailto:hhardikssharma@gmail.com)

**Related Resources:** 
- [Awesome-Deepfakes-Detection](https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection)
- [Awesome-Deepfakes-Materials](https://github.com/datamllab/awesome-deepfakes-materials)
- [Awesome-Face-Forgery-Generation-and-Detection](https://github.com/clpeng/Awesome-Face-Forgery-Generation-and-Detection)

---

## 📑 Table of Contents (Aligned with Paper)

1. [Introduction](#-introduction)
2. [Deepfake Generation Models](#-deepfake-generation-models)
   - [Autoencoders & VAE](#autoencoders--vae)
   - [Generative Adversarial Networks](#generative-adversarial-networks)
   - [Diffusion Models](#diffusion-models)
   - [Hybrid & Multimodal](#hybrid--multimodal)
3. [Taxonomy of Manipulations](#-taxonomy-of-manipulations)
   - [Identity Swap](#identity-swap-face-swap)
   - [Expression Reenactment](#expression--pose-reenactment)
   - [Attribute Editing](#attribute-editing)
   - [Face Morphing](#face-morphing--fusion)
   - [Fully Synthetic](#fully-synthetic--multimodal)
4. [Detection Approaches](#-detection-approaches)
   - [Classical & CNN-Based](#classical--cnn-based-detection)
   - [Transformer-Based](#transformer-based-detection)
   - [Frequency Domain](#frequency-domain-analysis)
   - [Hybrid & Multimodal](#hybrid-spatial-spectral-architectures)
5. [Evaluation Metrics](#-evaluation-metrics)
6. [Datasets & Benchmarks](#-datasets--benchmarks)
7. [Performance Comparison](#-performance-benchmarks)
8. [Challenges & Future Directions](#-challenges--future-directions)

---

## 📌 Introduction

The rapid evolution of generative models has enabled the creation of highly realistic synthetic faces, commonly known as **deepfakes**. While these models offer applications in entertainment and education, they also pose serious threats to privacy and digital trust.

**Key Contributions of This Survey:**
- Classification of generative pipelines into 5 families: GAN, VAE, Flow-based, Diffusion, and Hybrid
- Taxonomy of face manipulations: Identity swap, Expression transfer, Attribute editing, Morphing, Fully synthetic
- Chronological and methodological organization of detection methods (2018-2025)
- Comprehensive benchmarking across datasets and architectures

---

## 🎨 Deepfake Generation Models

*Paper Section: Deepfake Generation Models*

### Autoencoders & VAE

| Model | Year | Description | Links |
|-------|:----:|-------------|:-----:|
| **VAE** | 2014 | Variational Autoencoder | [Paper](https://arxiv.org/abs/1312.6114) |
| **β-VAE** | 2017 | Disentangled representation | [Paper](https://openreview.net/forum?id=Sy2fzU9gl) |
| **VQ-VAE** | 2017 | Vector Quantized VAE | [Paper](https://arxiv.org/abs/1711.00937) |

### Generative Adversarial Networks

| Model | Year | Venue | Links |
|-------|:----:|:-----:|:-----:|
| **GAN** (Original) | 2014 | NeurIPS | [Paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) [Code](https://github.com/goodfeli/adversarial) |
| **DCGAN** | 2016 | ICLR | [Paper](https://arxiv.org/pdf/1511.06434.pdf) [Code](https://github.com/Newmu/dcgan_code) |
| **ProGAN** | 2018 | ICLR | [Paper](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf) [Code](https://github.com/tkarras/progressive_growing_of_gans) |
| **StyleGAN** | 2019 | CVPR | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf) [Code](https://github.com/NVlabs/stylegan) |
| **StyleGAN2** | 2020 | CVPR | [Paper](https://arxiv.org/abs/1912.04958) [Code](https://github.com/NVlabs/stylegan2) |
| **StyleGAN3** | 2021 | NeurIPS | [Paper](https://arxiv.org/abs/2106.12423) [Code](https://github.com/NVlabs/stylegan3) |
| **BigGAN** | 2019 | ICLR | [Paper](https://arxiv.org/pdf/1809.11096.pdf) [Code](https://github.com/ajbrock/BigGAN-PyTorch) |

### Diffusion Models

| Model | Year | Venue | Links |
|-------|:----:|:-----:|:-----:|
| **DDPM** | 2020 | NeurIPS | [Paper](https://arxiv.org/abs/2006.11239) [Code](https://github.com/hojonathanho/diffusion) |
| **Latent Diffusion / Stable Diffusion** | 2022 | CVPR | [Paper](https://arxiv.org/abs/2112.10752) [Code](https://github.com/CompVis/stable-diffusion) |
| **DALL·E 2** | 2022 | OpenAI | [Paper](https://arxiv.org/abs/2204.06125) |
| **Imagen** | 2022 | NeurIPS | [Paper](https://arxiv.org/abs/2205.11487) |
| **Midjourney** | 2022 | — | [Website](https://www.midjourney.com/) |
| **GLIDE** | 2022 | ICML | [Paper](https://arxiv.org/abs/2112.10741) [Code](https://github.com/openai/glide-text2im) |

### Hybrid & Multimodal

| Model | Year | Venue | Links |
|-------|:----:|:-----:|:-----:|
| **CLIP** | 2021 | ICML | [Paper](https://arxiv.org/abs/2103.00020) [Code](https://github.com/openai/CLIP) |
| **BLIP-2** | 2023 | ICML | [Paper](https://arxiv.org/abs/2301.12597) [Code](https://github.com/salesforce/LAVIS) |
| **Flamingo** | 2022 | NeurIPS | [Paper](https://arxiv.org/abs/2204.14198) |

---

## 🔍 Taxonomy of Manipulations

*Paper Section: Taxonomy of Image-Based Deepfakes*

| Category | Objective | Representative Models | Potential Misuse |
|----------|-----------|----------------------|------------------|
| **Identity Swap** | Replace one person's face with another's | DeepFaceLab, StyleGAN3, FSGAN | Impersonation, misinformation |
| **Expression/Pose Transfer** | Reanimate facial expressions | Face2Face, ReenactGAN, NeRF-based | Talking-head forgeries, lip-sync |
| **Attribute Editing** | Modify facial attributes | StarGAN, AttGAN, CLIP-diffusion | Identity obfuscation, propaganda |
| **Face Morphing** | Blend multiple identities | Flow-based morphing, StyleGAN interpolation | Biometric fraud |
| **Fully Synthetic** | Generate faces from scratch | Stable Diffusion, Flamingo, HAMMER | Fake datasets, AI impersonation |

### Identity Swap (Face Swap)

| Method | Year | Venue | Links |
|--------|:----:|:-----:|:-----:|
| **DeepFaceLab** | 2020 | arXiv | [Paper](https://arxiv.org/pdf/2005.05535.pdf) [Code](https://github.com/iperov/DeepFaceLab) |
| FaceSwap | — | GitHub | [Code](https://github.com/deepfakes/faceswap) |
| **FSGAN** | 2019 | ICCV | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Nirkin_FSGAN_Subject_Agnostic_Face_Swapping_and_Reenactment_ICCV_2019_paper.pdf) [Code](https://github.com/YuvalNirkin/fsgan) |
| **SimSwap** | 2020 | ACM MM | [Paper](https://arxiv.org/abs/2106.06340) [Code](https://github.com/neuralchen/SimSwap) |
| **HifiFace** | 2021 | IJCAI | [Paper](https://arxiv.org/abs/2106.09965) [Code](https://github.com/johannwyh/HifiFace) |
| One Shot Face Swapping on Megapixels | 2021 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_One_Shot_Face_Swapping_on_Megapixels_CVPR_2021_paper.pdf) [Code](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels) |

### Expression & Pose Reenactment

| Method | Year | Venue | Links |
|--------|:----:|:-----:|:-----:|
| **Face2Face** | 2016 | CVPR | [Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Thies_Face2Face_Real-Time_Face_CVPR_2016_paper.pdf) |
| **ReenactGAN** | 2018 | ECCV | [Paper](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2018_reenactgan.pdf) [Code](https://github.com/wywu/ReenactGAN) |
| **GANimation** | 2018 | ECCV | [Paper](http://www.iri.upc.edu/files/scidoc/2052-GANimation:-Anatomically-aware-Facial-Animation-from-a-Single-Image.pdf) [Code](https://github.com/albertpumarola/GANimation) |
| **AD-NeRF** | 2021 | ICCV | [Paper](https://arxiv.org/abs/2103.11078) [Code](https://github.com/YudongGuo/AD-NeRF) |
| **DaGAN** | 2022 | CVPR | [Paper](https://arxiv.org/abs/2203.06605) [Code](https://github.com/harlanhong/CVPR2022-DaGAN) |
| **Wav2Lip** | 2020 | ACM MM | [Paper](https://arxiv.org/abs/2008.10010) [Code](https://github.com/Rudrabha/Wav2Lip) |

### Attribute Editing

| Method | Year | Venue | Links |
|--------|:----:|:-----:|:-----:|
| **StarGAN** | 2018 | CVPR | [Paper](https://zpascal.net/cvpr2018/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf) [Code](https://github.com/yunjey/stargan) |
| **StarGAN v2** | 2020 | CVPR | [Paper](https://arxiv.org/abs/1912.01865) [Code](https://github.com/clovaai/stargan-v2) |
| **AttGAN** | 2019 | TIP | [Paper](http://vipl.ict.ac.cn/uploadfile/upload/2019112511573287.pdf) [Code](https://github.com/LynnHo/AttGAN-Tensorflow) |
| **InterFaceGAN** | 2020 | TPAMI | [Paper](https://arxiv.org/pdf/2005.09635.pdf) [Code](https://github.com/genforce/interfacegan) |
| **StyleCLIP** | 2021 | ICCV | [Paper](https://arxiv.org/abs/2103.17249) [Code](https://github.com/orpatashnik/StyleCLIP) |
| **DiffusionCLIP** | 2022 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.html) [Code](https://github.com/gwang-kim/DiffusionCLIP) |

---

## 🔬 Detection Approaches

*Paper Section: Detection Approaches*

### Classical & CNN-Based Detection

| Method | Year | Venue | Links |
|--------|:----:|:-----:|:-----:|
| **MesoNet** | 2018 | WIFS | [Paper](https://arxiv.org/pdf/1809.00888.pdf) [Code](https://github.com/DariusAf/MesoNet) |
| **XceptionNet** | 2019 | ICCV | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.pdf) [Code](https://github.com/ondyari/FaceForensics) |
| Capsule-Forensics | 2019 | ICASSP | [Paper](https://arxiv.org/pdf/1810.11215.pdf) [Code](https://github.com/nii-yamagishilab/Capsule-Forensics) |
| In Ictu Oculi: Eye Blinking | 2018 | WIFS | [Paper](https://arxiv.org/pdf/1806.02877.pdf) [Code](https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi) |
| Face Warping Artifacts | 2018 | CVPRW | [Paper](https://arxiv.org/abs/1811.00656) |
| Multi-task Learning | 2019 | BTAS | [Paper](https://arxiv.org/pdf/1906.06876.pdf) [Code](https://github.com/nii-yamagishilab/ClassNSeg) |
| **ADDNet** (WildDeepfake) | 2020 | ACM MM | [Paper](https://dl.acm.org/doi/10.1145/3394171.3413769) [Code](https://github.com/OpenTAI/wild-deepfake) |
| **CNNDetection** (Wang et al.) | 2020 | CVPR | [Paper](https://arxiv.org/abs/1912.11035) [Code](https://github.com/PeterWang512/CNNDetection) |
| **Face X-Ray** | 2020 | CVPR | [Paper](https://arxiv.org/abs/1912.13458) |
| **F3-Net** | 2020 | ECCV | [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570086.pdf) [Code](https://github.com/yyk-wew/F3-Net) |

### Transformer-Based Detection

| Method | Year | Venue | Links |
|--------|:----:|:-----:|:-----:|
| **ViT Detector** | 2022 | arXiv | [Paper](https://arxiv.org/abs/2204.05469) |
| **Efficient FAT** | 2023 | AAAI | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25260) |
| **TALL** | 2023 | ICCV | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Xu_TALL_Thumbnail_Layout_for_Deepfake_Video_Detection_ICCV_2023_paper.html) [Code](https://github.com/YuezunLi/TALL) |
| Token-Level Shuffling | 2025 | AAAI | [Paper](https://arxiv.org/pdf/2501.04376) |
| Facial Component Guided | 2025 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Towards_More_General_Video-based_Deepfake_Detection_through_Facial_Component_Guided_CVPR_2025_paper.pdf) |

### Frequency Domain Analysis

| Method | Year | Venue | Links |
|--------|:----:|:-----:|:-----:|
| Thinking in Frequency | 2020 | CVPR | [Paper](https://arxiv.org/abs/2005.04501) |
| **CNN-FF** | 2021 | arXiv | [Paper](https://arxiv.org/abs/2103.04452) |
| **DeFakeHop** | 2021 | arXiv | [Paper](https://arxiv.org/abs/2103.05891) |
| **FreqDebias** | 2025 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.pdf) |
| **NPR** | 2024 | CVPR | [Paper](https://arxiv.org/abs/2312.10461) [Code](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) |

### Hybrid Spatial-Spectral Architectures

| Method | Year | Venue | Links |
|--------|:----:|:-----:|:-----:|
| **SFANet** | 2022 | arXiv | [Paper](https://arxiv.org/abs/2203.14530) |
| **RECCE** | 2022 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Cao_End-to-End_Reconstruction-Classification_Learning_for_Face_Forgery_Detection_CVPR_2022_paper.pdf) |
| **CrossDF** | 2023 | arXiv | [Paper](https://arxiv.org/abs/2309.00987) |
| **HAMMER** | 2024 | NeurIPS | [Paper](https://arxiv.org/abs/2310.07906) |
| **D³** | 2025 | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_D3_Scaling_Up_Deepfake_Detection_by_Learning_from_Discrepancy_CVPR_2025_paper.pdf) |

### VLM & Multimodal Detection

| Method | Year | Venue | Links |
|--------|:----:|:-----:|:-----:|
| **SIDA** | 2025 | CVPR | [Paper](https://github.com/hzlsaber/SIDA) [Code](https://github.com/hzlsaber/SIDA) |
| **FakeShield** | 2025 | ICLR | [Paper](https://arxiv.org/pdf/2410.02761) |
| **Forensics Adapter** | 2025 | CVPR | [Paper](https://arxiv.org/abs/2411.19715) |
| Rethinking VLM in Face Forensics | 2025 | CVPR | [Paper](https://arxiv.org/pdf/2503.20188) |
| **AntifakePrompt** | 2025 | ICLR | [Paper](https://iclr.cc/virtual/2025/) |
| **FakeVLM** | 2025 | NeurIPS | [Paper](https://neurips.cc/virtual/2025/) |

### Diffusion-Aware Detection

| Method | Year | Venue | Links |
|--------|:----:|:-----:|:-----:|
| **DIRE** | 2023 | ICCV | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_DIRE_Diffusion_Reconstruction_Error_for_Fake_Image_Detection_ICCV_2023_paper.pdf) [Code](https://github.com/ZhendongWang6/DIRE) |
| **DRCT** | 2024 | ICML | [Paper](https://proceedings.mlr.press/v235/chen24ay.html) [Code](https://github.com/beibuwandeluori/DRCT) |
| **DiffusionForensics** | 2024 | arXiv | [Paper](https://arxiv.org/abs/2401.02530) |
| **DiffusionFake** | 2024 | NeurIPS | [Paper](https://neurips.cc/virtual/2024/poster/xxxxx) |
| **D4** | 2024 | WACV | [Paper](https://openaccess.thecvf.com/content/WACV2024/html/Hooda_D4_Detection_of_Adversarial_Diffusion_Deepfakes_Using_Disjoint_Ensembles_WACV_2024_paper.html) [Code](https://github.com/nmangaokar/wacv_24_d4) |

---

## 📏 Evaluation Metrics

*Paper Section: Evaluation Metrics and Forensic Analysis*

| Metric | Definition | Use Case |
|--------|------------|----------|
| **Accuracy** | (TP+TN) / N | Overall correctness |
| **Precision** | TP / (TP+FP) | False positive control |
| **Recall (TPR)** | TP / (TP+FN) | Detection sensitivity |
| **F₁-Score** | 2×(Precision×Recall)/(Precision+Recall) | Balanced measure |
| **AUC** | Area under ROC curve | Threshold-independent |
| **EER** | FPR = FNR intersection | Biometric systems |
| **mAP / AP** | Mean/Average Precision | Multi-class/imbalanced |
| **IoU** | Intersection over Union | Manipulation localization |

---

## 📊 Datasets & Benchmarks

*Paper Section: Datasets and Benchmarks*

### Video Deepfake Datasets

| Dataset | Year | Scale | Manipulation Techniques | Links |
|---------|:----:|:-----:|------------------------|:-----:|
| **FaceForensics++** | 2019 | 1K videos | DeepFakes, Face2Face, FaceSwap, NeuralTextures | [GitHub](https://github.com/ondyari/FaceForensics) |
| **Celeb-DF** | 2020 | 5.6K videos | Improved autoencoder swaps | [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **DFDC** | 2020 | 100K videos | Multiple synthesis pipelines | [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) |
| **DeeperForensics-1.0** | 2020 | 60K clips | FaceSwap, DeepFake variants | [GitHub](https://github.com/EndlessSora/DeeperForensics-1.0) |
| **WildDeepfake** | 2020 | 7,314 faces / 707 videos | In-the-wild internet deepfakes | [HuggingFace](https://huggingface.co/datasets/xingjunm/WildDeepfake) [GitHub](https://github.com/OpenTAI/wild-deepfake) |
| **FakeAVCeleb** | 2021 | 20K clips | Audio-Video multimodal | [Website](https://sites.google.com/view/fakeavcelebdash-lab/) |
| **DFDC Preview** | 2019 | 5K videos | Facebook challenge preview | [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) |

### Image Deepfake Datasets

| Dataset | Year | Scale | Generation Techniques | Links |
|---------|:----:|:-----:|----------------------|:-----:|
| **iFakeFaceDB** | 2021 | 63K images | Autoencoder + GAN hybrids | — |
| **GenImage** | 2023 | 1.2M images | StyleGAN3, BigGAN, Diffusion (8 generators) | [GitHub](https://github.com/GenImage-Dataset/GenImage) [Drive](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS) |
| **DiffusionDB** | 2023 | 14M images | Stable Diffusion, Imagen | [Website](https://poloclub.github.io/diffusiondb/) |
| **DeepFakeFace** | 2023 | — | OpenRL synthetic faces | [GitHub](https://github.com/OpenRL/DeepFakeFace) |
| **UFD / CNNDetection** | 2020 | — | ProGAN generated images | [GitHub](https://github.com/PeterWang512/CNNDetection) |
| **AI-Forensics** | 2024 | 2.1M images | GAN + Diffusion + NeRF | — |
| **SID-Set** (SIDA) | 2025 | 300K images | Social media AI images | [GitHub](https://github.com/hzlsaber/SIDA) |

### Multimodal Datasets

| Dataset | Year | Modalities | Links |
|---------|:----:|:----------:|:-----:|
| **DGM4** | 2024 | Audio-Visual-Text | [GitHub](https://github.com/rshaojimmy/MultiModal-DeepFake) |
| **DeepfakeEval-2024** | 2024 | 88 websites, 52 languages | [Paper](https://arxiv.org/abs/2411.xxxxx) |
| **DF40** | 2024 | 40 deepfake techniques | [NeurIPS](https://neurips.cc/virtual/2024/) |

---

## 📈 Performance Benchmarks

*Paper Section: Performance of Detection Methods*

### Generalization Performance: ID vs OOD Accuracy (%)
*Source: D³ Paper (CVPR 2025)*

| Method | ID Accuracy | OOD Accuracy | Total Accuracy | Venue |
|--------|:-----------:|:------------:|:--------------:|:-----:|
| **CNNDet** | 93.3 | 69.9 | 79.2 | CVPR 2020 |
| **Patchfor** | 97.9 | 78.9 | 86.5 | ECCV 2020 |
| **LNP** | 88.1 | 71.9 | 78.4 | ECCV 2022 |
| **DIRE** | 97.6 | 68.4 | 80.1 | ICCV 2023 |
| **UFD** (UnivFD) | 86.6 | 81.4 | 83.5 | CVPR 2023 |
| **UCF** | 91.7 | 75.0 | 81.7 | ICCV 2023 |
| **NPR** | **98.6** | 78.7 | 86.6 | CVPR 2024 |
| **D³** | 96.6 | **86.7** | **90.7** | CVPR 2025 |

### Cross-Dataset Performance: AUC (%)
*Training: FF++ (c23) → Testing: Other Datasets*

| Method | FF++ (c23) | FF++ (c40) | Celeb-DF | DFDC | DeeperForensics | Year |
|--------|:----------:|:----------:|:--------:|:----:|:---------------:|:----:|
| MesoNet | 83.1 | 70.5 | 54.8 | 65.2 | — | 2018 |
| XceptionNet | 95.7 | 86.5 | 73.4 | 70.8 | 85.2 | 2019 |
| Face X-Ray | 95.4 | 91.2 | 79.5 | 65.5 | 86.8 | 2020 |
| F3-Net | 97.5 | 90.4 | 76.3 | 72.1 | 88.6 | 2020 |
| Multi-Attention | 97.6 | 92.1 | 79.2 | 73.5 | 87.4 | 2021 |
| SBI | 97.5 | 93.2 | 93.2 | 72.4 | 89.7 | 2022 |
| RECCE | 97.6 | 94.3 | 91.1 | 72.1 | 89.1 | 2022 |
| UCF | 96.8 | 93.5 | 82.4 | 75.6 | 88.4 | 2023 |
| TALL | 97.2 | 94.1 | 90.8 | 76.8 | 90.1 | 2023 |
| LAA-Net | **99.1** | **95.8** | 92.4 | 78.3 | **91.5** | 2024 |
| D³ | 98.7 | 95.2 | **95.6** | **86.7** | 91.2 | 2025 |

### Detailed Method Comparison

| Method / Year | Architecture | Backbone | ACC (%) | AUC | EER (%) | Speed |
|---------------|--------------|----------|:-------:|:---:|:-------:|:-----:|
| MesoNet (2018) | CNN (Shallow) | 4 conv layers | 95.3 | N.R. | N.R. | ~150 FPS |
| XceptionNet (2019) | CNN (Deep) | Depthwise separable | 99.3 | 0.99 | 3.5 | ~100 FPS |
| ViT Detector (2022) | Transformer | Self-attention patches | 98.9 | 0.99 | 3.2 | ~55 FPS |
| SFANet (2022) | Hybrid | Swin + Frequency | 99.2 | 0.99 | 3.7 | ~60 FPS |
| HAMMER (2024) | Multimodal | Audio-Visual-Text fusion | 86.8 | 0.94 | 13.2 | ~30 FPS |
| DiffusionForensics (2024) | Diffusion-aware | Diffusion prior embedding | 91.5 | 0.90 | 5.8 | High |

---

## 🚧 Challenges & Future Directions

*Paper Section: Challenges and Future Directions*

### Key Challenges

| Challenge | Description | Promising Approaches |
|-----------|-------------|---------------------|
| **Generalization** | 20-40% accuracy drop on unseen generators | Domain-invariant learning, meta-learning |
| **Diffusion Detection** | Diffusion models preserve spectral statistics | Diffusion priors, latent trajectory signatures |
| **Adversarial Robustness** | Vulnerable to perturbations and post-processing | Adversarial training, certified defenses |
| **Explainability** | Need for forensic interpretability | Attention visualization, frequency heatmaps |
| **Fairness** | Demographic bias in detectors | Inclusive data curation, bias auditing |
| **Efficiency** | Real-time deployment requirements | Neural architecture search, quantization |

### Future Research Directions

1. **Diffusion-Aware Forensics** - Detecting latent trajectory signatures
2. **Multimodal Reasoning** - Joint audio-visual-text analysis
3. **Watermark-Based Provenance** - Proactive verification
4. **Continual Learning** - Adaptation to evolving generators
5. **Privacy-Preserving Detection** - Federated and on-device learning

---

## 🛠️ Code Implementations

| Method | Venue | Code |
|--------|-------|:----:|
| **DeepfakeBench** | NeurIPS 2023 | [SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) |
| **FaceForensics++** | ICCV 2019 | [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) |
| **D³** | CVPR 2025 | Coming Soon |
| **SIDA** | CVPR 2025 | [hzlsaber/SIDA](https://github.com/hzlsaber/SIDA) |
| **LAA-Net** | CVPR 2024 | [10Ring/LAA-Net](https://github.com/10Ring/LAA-Net) |
| **TALL** | ICCV 2023 | [rainy-xu/TALL4Deepfake](https://github.com/rainy-xu/TALL4Deepfake) |
| **DIRE** | ICCV 2023 | [ZhendongWang6/DIRE](https://github.com/ZhendongWang6/DIRE) |
| **UnivFD** | CVPR 2023 | [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) |
| **SBI** | CVPR 2022 | [mapooon/SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages) |
| **F3-Net** | ECCV 2020 | [yyk-wew/F3-Net](https://github.com/yyk-wew/F3-Net) |

---

## ✍️ Citation

```bibtex
@article{sharma2025comprehensive,
  title   = {Comprehensive Review of Image-Based Deepfake Detection Techniques and Datasets},
  author  = {Hardik Sharma and Sachin Chaudhary and Praful Hambarde and Akshay Dudhane},
  journal = {IEEE Transactions on Information Forensics and Security},
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
