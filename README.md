<p align="center">
  <h1 align="center">🎭 Awesome Deepfake Detection Survey</h1>
  <p align="center">
    <strong>Comprehensive Review of Image-Based Deepfake Detection Techniques and Datasets</strong>
  </p>
  <p align="center">
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
    <img src="https://img.shields.io/badge/Papers-300+-blue" alt="Papers">
    <img src="https://img.shields.io/badge/Datasets-50+-green" alt="Datasets">
    <img src="https://img.shields.io/badge/Code-50+-orange" alt="Code">
  </p>
  <p align="center">
    <a href="#-datasets">Datasets</a> •
    <a href="#-competitions">Competitions</a> •
    <a href="#-tools">Tools</a> •
    <a href="#-detection-methods">Methods</a> •
    <a href="REFERENCES.md">Citations</a>
  </p>
</p>

---

## 📖 About

> **"Comprehensive Review of Image-Based Deepfake Detection Techniques and Datasets"**  
> *Authors: Hardik Sharma, Sachin Chaudhary, Praful Hambarde, Akshay Dudhane*

📧 **Contact:** [hhardikssharma@gmail.com](mailto:hhardikssharma@gmail.com)

**Related Resources:** [Awesome-Deepfakes-Detection](https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection)

---

## 📑 Contents

- [Datasets](#-datasets)
  - [Video Deepfake Datasets](#-video-deepfake-datasets)
  - [Image Deepfake Datasets](#️-image-deepfake-datasets)
  - [Diffusion-Generated Datasets](#-diffusion-generated-datasets)
  - [Audio Deepfake Datasets](#-audio-deepfake-datasets)
- [Competitions](#-competitions)
- [Tools](#-tools)
- [Detection Methods](#-detection-methods)
  - [Recent Conference Papers](#-recent-conference-papers)
  - [Survey Papers](#-survey-papers)
  - [Spatiotemporal Based](#-spatiotemporal-based)
  - [Frequency Based](#-frequency-based)
  - [Generalization](#-generalization)
  - [Interpretability](#-interpretability)
  - [Human-Decision](#-human-decision)
  - [Localization](#-localization)
  - [Multi-modal Based](#-multi-modal-based)
  - [Biological Signal](#-biological-signal)
  - [Robustness](#-robustness)
  - [Fairness](#-fairness)
  - [Fingerprint/Watermark](#-fingerprintwatermark)
  - [Identity-Related](#-identity-related)
  - [Adversarial Attack](#-adversarial-attack)
  - [Real Scenario](#-real-scenario)
  - [Anomaly Detection](#-anomaly-detection)
  - [Self-Supervised Learning](#-self-supervised-learning)
  - [Source Model Attribution](#-source-model-attribution)
  - [Transformer-Based Detection](#-transformer-based-detection)
  - [VLM-Based Detection](#-vlm-based-detection)
  - [Data Augmentation Methods](#-data-augmentation-methods)
- [Implementations](#️-implementations)
- [Benchmarks](#-benchmarks)

---

## 📊 Datasets

### 🎬 Video Deepfake Datasets

| Dataset | Real Videos | Fake Videos | Year | Note |
|:-------:|:-----------:|:-----------:|:----:|:-----|
| **UADFV** | 49 | 49 | 2018 | Focus on head pose |
| **EBV** | - | 49 | 2018 | Focus on eye blinking |
| **Deepfake-TIMIT** | 320 | 640 | 2018 | GAN-based methods |
| **DFFD** | 1,000 | 3,000 | 2019 | Multiple SOTA methods |
| **DeepfakeDetection** | 363 | 3,068 | 2019 | From actors with public methods |
| **Celeb-DF (v2)** | 590 | 5,639 | 2019 | High quality |
| **DFDC** | 23,564 | 104,500 | 2019 | Kaggle competition |
| **FaceForensic++** | 1,000 | 5,000 | 2019 | 5 generation methods |
| **FFIW-10K** | 10,000 | 10,000 | 2019 | Multiple faces per frame |
| **WLDR** | - | - | 2019 | Person of interest from YouTube |
| **DeeperForensics-1.0** | 50,000 | 10,000 | 2020 | Real-world perturbations |
| **Wild-Deepfake** | 3,805 | 3,509 | 2021 | Collected from Internet |
| **ForgeryNet** | 99,630 | 121,617 | 2021 | 8 video methods, perturbations |
| **FakeAVCeleb** | 500 | 19,500 | 2021 | Audio-visual multimodal |
| **KoDF** | 62,166 | 175,776 | 2021 | 6 methods |
| **DF-Platter** | - | - | 2023 | Multi-face heterogeneous |
| **DeepSpeak** | 6,226 | 6,799 | 2024 | Lip-sync and face-swap |
| **IDForge** | 79,827 | 169,311 | 2024 | Multimodal with identity info |
| **Celeb-DF++** | 590 | 53,196 | 2025 | Large-scale generalization benchmark |

**Dataset Links:**
* **UADFV**: [Paper](https://arxiv.org/abs/1811.00661)
* **EBV**: [Paper](https://arxiv.org/abs/1806.02877) | [Download](http://www.cs.albany.edu/~lsw/downloads.html)
* **Deepfake-TIMIT**: [Paper](https://arxiv.org/abs/1812.08685) | [Download](https://conradsanderson.id.au/vidtimit/)
* **DFFD**: [Paper](http://cvlab.cse.msu.edu/pdfs/dang_liu_stehouwer_liu_jain_cvpr2020.pdf) | [Download](http://cvlab.cse.msu.edu/dffd-dataset.html)
* **DeepfakeDetection**: [Download](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html)
* **Celeb-DF (v2)**: [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.pdf) | [Download](https://github.com/yuezunli/celeb-deepfakeforensics)
* **DFDC**: [Paper](https://arxiv.org/abs/2006.07397) | [Download](https://www.kaggle.com/c/deepfake-detection-challenge/data)
* **FaceForensic++**: [Paper](https://arxiv.org/abs/1901.08971) | [Download](https://github.com/ondyari/FaceForensics)
* **FFIW-10K**: [Paper](https://arxiv.org/abs/2103.16076) | [Download](https://github.com/tfzhou/FFIW)
* **DeeperForensics-1.0**: [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_DeeperForensics-1.0_A_Large-Scale_Dataset_for_Real-World_Face_Forgery_Detection_CVPR_2020_paper.pdf) | [Download](https://github.com/EndlessSora/DeeperForensics-1.0)
* **Wild Deepfake**: [Paper](https://arxiv.org/abs/2101.01456) | [Download](https://github.com/deepfakeinthewild/deepfake-in-the-wild)
* **ForgeryNet**: [Paper](https://arxiv.org/abs/2103.05630) | [Download](https://github.com/yinanhe/forgerynet)
* **FakeAVCeleb**: [Paper](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper-round2.pdf) | [Download](https://github.com/DASH-Lab/FakeAVCeleb)
* **DeepSpeak**: [Paper](https://arxiv.org/abs/2408.05366) | [Download](https://huggingface.co/datasets/faridlab/deepspeak_v1)
* **IDForge**: [Paper](https://arxiv.org/abs/2401.11764) | [Download](https://github.com/xyyandxyy/IDForge)
* **Celeb-DF++**: [Paper](https://arxiv.org/abs/2507.18015) | [Download](https://github.com/OUC-VAS/Celeb-DF-PP)

### 🖼️ Image Deepfake Datasets

| Dataset | Real Images | Fake Images | Year | Note |
|:-------:|:-----------:|:-----------:|:----:|:-----|
| **DFFD** | 58,703 | 240,336 | 2019 | Multiple SOTA methods |
| **iFakeFaceDB** | - | 87,000 | 2020 | StyleGAN generated |
| **100K Faces** | - | 100,000 | 2021 | StyleGAN generated |
| **DFGC** | 1,000 | N*1,000 | 2021 | Competition dataset |
| **ForgeryNet** | 1,438,201 | 1,457,861 | 2021 | 7 image methods |
| **SID-Set (SIDA)** 🆕 | - | 300K | 2025 | Social media, GT masks, CVPR 2025 |
| **GenImage** | - | 1.3M | 2023 | 8 generators (SD, Midjourney, DALL-E) |
| **CelebA-Spoof** | 671K | - | 2020 | 10 spoof types, 40 attributes |
| **OpenForensics** | 115K | - | 2021 | Open-source benchmark |
| **CIFAKE** | 60K | 60K | 2023 | CIFAR-style fake images |

**Dataset Links:**
* **DFFD**: [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Dang_On_the_Detection_of_Digital_Face_Manipulation_CVPR_2020_paper.pdf) | [Download](http://cvlab.cse.msu.edu/project-ffd.html)
* **iFakeFaceDB**: [Paper](https://arxiv.org/abs/1911.05351) | [Download](https://github.com/socialabubi/iFakeFaceDB)
* **100K Faces**: [Download](https://generated.photos/datasets)
* **DFGC**: [Paper](https://arxiv.org/abs/2106.01217) | [Download](https://github.com/bomb2peng/DFGC_starterkit)
* **SID-Set**: [HuggingFace](https://huggingface.co/datasets/saberzl/SID_Set)
* **GenImage**: [GitHub](https://github.com/GenImage-Dataset/GenImage)

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

## 🏆 Competitions

| Name | Link | Year | Note |
|:----:|:----:|:----:|:-----|
| Deepfake Detection Challenge | [Website](https://www.kaggle.com/c/deepfake-detection-challenge) | 2019 | Video-level detection, DFDC dataset, 2000+ teams |
| DeepForensics Challenge | [Website](https://competitions.codalab.org/competitions/25228) | 2020 | DeeperForensics-1.0, real-world scenarios |
| Deepfake Game Competition | [Website](https://competitions.codalab.org/competitions/29583) | 2021 | Image generation + video detection, Celeb-DF(v2) |
| Face Forgery Analysis Challenge | [Website](https://competitions.codalab.org/competitions/33386) | 2021 | Image + video + temporal localization, ForgeryNet |

---

## 🛠️ Tools

* **Sensity**: [Website](https://sensity.ai/)
* **Deepware**: [Website](https://deepware.ai/)
* **Baidu Security**: [Website](http://weishi.baidu.com/product/deepfake)
* **DeepReal**: [Website](https://deepfakes.real-ai.cn/)

---

## 🔬 Detection Methods

### 📚 Recent Conference Papers

#### CVPR 2025
* **SIDA**: Social Media Image Deepfake Detection with LMM: [Paper](https://github.com/hzlsaber/SIDA) | [Code](https://github.com/hzlsaber/SIDA)
* **D3**: Scaling Up Deepfake Detection by Learning from Discrepancy: [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_D3_Scaling_Up_Deepfake_Detection_by_Learning_from_Discrepancy_CVPR_2025_paper.pdf)
* Face Forgery Video Detection via Temporal Forgery Cue Unraveling: [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Face_Forgery_Video_Detection_via_Temporal_Forgery_Cue_Unraveling_CVPR_2025_paper.pdf)
* Generalizing Deepfake with Plug-and-Play Video-Level Blending: [Paper](https://arxiv.org/pdf/2408.17065)
* **Forensics Adapter**: Adapting CLIP for Generalizable Face Forgery Detection: [Paper](https://arxiv.org/abs/2411.19715)
* **FreqDebias**: Consistency-Driven Frequency Debiasing: [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.pdf)

#### CVPR 2024
* Exploiting Style Latent Flows for Generalizing Video Deepfake Detection: [Paper](https://arxiv.org/abs/2403.06592)
* **AVFF**: Audio-Visual Feature Fusion for Video Deepfake Detection: [Paper](https://arxiv.org/pdf/2406.02951v1)
* Transcending Forgery Specificity with Latent Space Augmentation: [Paper](https://arxiv.org/abs/2311.11278)
* Rethinking Up-Sampling Operations in CNN-based Generative Network: [Paper](https://arxiv.org/abs/2312.10461) | [Github](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
* **LAA-Net**: Localized Artifact Attention Network: [Paper](https://arxiv.org/pdf/2401.13856) | [Github](https://github.com/10Ring/LAA-Net)
* Preserving Fairness Generalization in Deepfake Detection: [Paper](https://arxiv.org/pdf/2402.17229) | [Github](https://github.com/Purdue-M2/Fairness-Generalization)

#### CVPR 2023
* Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection: [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Dong_Implicit_Identity_Leakage_The_Stumbling_Block_to_Improving_Deepfake_Detection_CVPR_2023_paper.pdf) | [Github](https://github.com/megvii-research/CADDM)
* **AltFreezing** for More General Face Forgery Detection: [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_AltFreezing_for_More_General_Video_Face_Forgery_Detection_CVPR_2023_paper.pdf)
* **AUNet**: Learning Relations Between Action Units: [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Bai_AUNet_Learning_Relations_Between_Action_Units_for_Face_Forgery_Detection_CVPR_2023_paper.pdf)
* Dynamic Graph Learning with Content-guided Spatial-Frequency: [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Dynamic_Graph_Learning_With_Content-Guided_Spatial-Frequency_Relation_Reasoning_for_Deepfake_CVPR_2023_paper.pdf)
* **TruFor**: Leveraging all-round clues for trustworthy image forgery detection: [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.pdf) | [Github](https://github.com/grip-unina/TruFor)
* Learning on Gradients (LGrad): [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tan_Learning_on_Gradients_Generalized_Artifacts_Representation_for_GAN-Generated_Images_Detection_CVPR_2023_paper.pdf) | [Github](https://github.com/chuangchuangtan/LGrad)
* Towards Universal Fake Image Detectors: [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.pdf) | [Github](https://github.com/Yuheng-Li/UniversalFakeDetect)

#### ICCV 2025
* **FakeRadar**: Probing Forgery Outliers to Detect Unknown Deepfake Videos: [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Li_FakeRadar_Probing_Forgery_Outliers_to_Detect_Unknown_Deepfake_Videos_ICCV_2025_paper.html)
* **DeepShield**: Fortifying Deepfake Video Detection: [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Cai_DeepShield_Fortifying_Deepfake_Video_Detection_with_Local_and_Global_Forgery_Analysis_ICCV_2025_paper.html)
* **FakeSTormer**: Vulnerability-Aware Spatio-Temporal Learning: [Paper](https://arxiv.org/pdf/2501.01184) | [Code](https://github.com/10Ring/FakeSTormer)
* **PwTF-DVD**: Pixel-wise Temporal Frequency-based Detection: [Paper](https://arxiv.org/abs/2507.02398) | [Code](https://github.com/rama0126/PwTF-DVD)

#### ICCV 2023
* **UCF**: Uncovering Common Features for Generalizable Deepfake Detection: [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_UCF_Uncovering_Common_Features_for_Generalizable_Deepfake_Detection_ICCV_2023_paper.pdf)
* **TALL**: Thumbnail Layout for Deepfake Video Detection: [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_TALL_Thumbnail_Layout_for_Deepfake_Video_Detection_ICCV_2023_paper.pdf) | [Github](https://github.com/rainy-xu/TALL4Deepfake)
* Quality-Agnostic Deepfake Detection with Intra-model Collaborative Learning: [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Le_Quality-Agnostic_Deepfake_Detection_with_Intra-model_Collaborative_Learning_ICCV_2023_paper.pdf)

#### ECCV 2024
* Learning Natural Consistency Representation: [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11103.pdf)
* Contrasting Deepfakes Diffusion via Contrastive Learning: [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08009.pdf)
* Real Appearance Modeling for More General Deepfake Detection: [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06913.pdf)
* Fake It till You Make It: Curricular Dynamic Forgery Augmentations: [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11581.pdf)

#### NeurIPS 2024
* **SpeechForensics**: Audio-Visual Speech Representation Learning: [Paper](https://nips.cc/virtual/2024/poster/94610) | [Github](https://github.com/Eleven4AI/SpeechForensics)
* **FreqBlender**: Enhancing DeepFake Detection by Blending Frequency Knowledge: [Paper](https://nips.cc/virtual/2024/poster/93596)
* Can We Leave Deepfake Data Behind in Training?: [Paper](https://arxiv.org/pdf/2408.17052)
* **DeepfakeBench**: A Comprehensive Benchmark: [Paper](https://papers.nips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf) | [Github](https://github.com/SCLBD/DeepfakeBench)

#### ICLR 2024
* Poisoned Forgery Face: Towards Backdoor Attacks: [Paper](https://openreview.net/pdf?id=8iTpB4RNvP) | [Github](https://github.com/JWLiang007/PFF)

#### ICML 2024
* **DRCT**: Diffusion Reconstruction Contrastive Training: [Paper](https://proceedings.mlr.press/v235/chen24ay.html) | [Github](https://github.com/beibuwandeluori/DRCT)
* How to Trace Latent Generative Model Generated Images: [Paper](https://proceedings.mlr.press/v235/wang24bj.html)

#### AAAI 2023
* Deepfake Video Detection via Facial Action Dependencies Estimation: [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25658)
* Noise Based Deepfake Detection via Multi-Head Relative-Interaction: [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/26701)

#### IJCAI 2022
* Region-Aware Temporal Inconsistency Learning: [Paper](https://www.ijcai.org/proceedings/2022/0129.pdf)
* Anti-Forgery: Stealthy and Robust DeepFake Disruption Attack: [Paper](https://www.ijcai.org/proceedings/2022/0107.pdf) | [Github](https://github.com/AbstractTeen/AntiForgery)

---

### 📖 Survey Papers

* Deepfake: Definitions, Performance Metrics and Standards, *arXiv* 2022: [Paper](https://arxiv.org/abs/2208.10913)
* Deepfakes Generation and Detection: State-of-the-art, *Applied Intelligence* 2022: [Paper](https://link.springer.com/article/10.1007/s10489-022-03766-z)
* DeepFake Detection for Human Face Images and Videos: A Survey, *IEEE Access* 2022: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9712265)
* Deepfake Detection: A Systematic Literature Review, *IEEE Access* 2022: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9721302)
* A Survey on Deepfake Video Detection, *Iet Biometrics* 2021: [Paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/bme2.12031)

---

### ⏱️ Spatiotemporal Based

* **TALL**: Thumbnail Layout for Deepfake Video Detection, *ICCV 2023*: [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_TALL_Thumbnail_Layout_for_Deepfake_Video_Detection_ICCV_2023_paper.pdf) | [Github](https://github.com/rainy-xu/TALL4Deepfake)
* Hierarchical Contrastive Inconsistency Learning, *ECCV* 2022: [Paper](https://link.springer.com/chapter/10.1007/978-3-031-19775-8_35)
* Region-Aware Temporal Inconsistency Learning, *IJCAI* 2022: [Paper](https://www.ijcai.org/proceedings/2022/0129.pdf)
* Delving into the Local: Dynamic Inconsistency Learning, *AAAI* 2022: [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/19955)
* Deepfake video detection with spatiotemporal dropout transformer, *ACM MM* 2022: [Paper](https://dl.acm.org/doi/abs/10.1145/3503161.3547913)
* Exploring Temporal Coherence for More General Detection, *ICCV* 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Zheng_Exploring_Temporal_Coherence_for_More_General_Video_Face_Forgery_Detection_ICCV_2021_paper.html) | [Github](https://github.com/yinglinzheng/FTCN)
* Detecting Deepfake Videos with Temporal Dropout 3DCNN, *IJCAI* 2021: [Paper](https://www.ijcai.org/proceedings/2021/0178.pdf)

---

### 📊 Frequency Based

* **FreqDebias**: Consistency-Driven Frequency Debiasing, *CVPR* 2025: [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.pdf)
* **FreqBlender**: Enhancing DeepFake Detection, *NeurIPS* 2024: [Paper](https://arxiv.org/abs/2404.13872)
* **FrePGAN**: Frequency-level Perturbations, *AAAI* 2022: [Paper](https://arxiv.org/abs/2202.03347)
* Generalizing Face Forgery with High-frequency Features, *CVPR* 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Luo_Generalizing_Face_Forgery_Detection_With_High-Frequency_Features_CVPR_2021_paper.html)
* **F3-Net**: Thinking in Frequency, *ECCV* 2020: [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570086.pdf)
* Spatial-Phase Shallow Learning, *CVPR* 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Spatial-Phase_Shallow_Learning_Rethinking_Face_Forgery_Detection_in_Frequency_Domain_CVPR_2021_paper.pdf)

---

### 🎯 Generalization

* Transcending Forgery Specificity with Latent Space Augmentation, *CVPR* 2024: [Paper](https://arxiv.org/abs/2311.11278)
* Rethinking Up-Sampling Operations for Generalizable Deepfake Detection, *CVPR* 2024: [Paper](https://arxiv.org/abs/2312.10461) | [Github](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
* Exploiting Style Latent Flows for Generalizing Video Deepfake Detection, *CVPR* 2024: [Paper](https://arxiv.org/abs/2403.06592)
* **UCF**: Uncovering Common Features for Generalizable Detection, *ICCV 2023*: [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_UCF_Uncovering_Common_Features_for_Generalizable_Deepfake_Detection_ICCV_2023_paper.pdf)
* Supervised Contrastive Learning for Generalizable DeepFakes Detection, *WACV* 2022: [Paper](https://openaccess.thecvf.com/content/WACV2022W/XAI4B/papers/Xu_Supervised_Contrastive_Learning_for_Generalizable_and_Explainable_DeepFakes_Detection_WACVW_2022_paper.pdf) | [Github](https://github.com/xuyingzhongguo/deepfake_supcon)
* **FReTAL**: Generalizing Deepfake Detection using Knowledge Distillation, *CVPR Workshop* 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021W/WMF/html/Kim_FReTAL_Generalizing_Deepfake_Detection_Using_Knowledge_Distillation_and_Representation_Learning_CVPRW_2021_paper.html)
* ForensicTransfer: Weakly-supervised Domain Adaptation, *arXiv* 2018: [Paper](https://arxiv.org/abs/1812.02510)

---

### 🔍 Interpretability

* Explaining Deepfake Detection by Analysing Image Matching, *ECCV* 2022: [Paper](https://link.springer.com/chapter/10.1007/978-3-031-19781-9_2)
* Detecting and Recovering Sequential DeepFake Manipulation, *ECCV* 2022: [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730710.pdf) | [Github](https://github.com/rshaojimmy/SeqDeepFake)
* Interpretable and Trustworthy Deepfake Detection via Dynamic Prototypes, *WACV* 2021: [Paper](https://openaccess.thecvf.com/content/WACV2021/html/Trinh_Interpretable_and_Trustworthy_Deepfake_Detection_via_Dynamic_Prototypes_WACV_2021_paper.html)
* What makes fake images detectable?, *arXiv* 2020: [Paper](https://arxiv.org/abs/2008.10588) | [Github](https://github.com/chail/patch-forensics)

---

### 👥 Human-Decision

* Video Manipulations Beyond Faces: Human-Machine Analysis, *WACV* 2023: [Paper](https://openaccess.thecvf.com/content/WACV2023W/MAP-A/papers/Mittal_Video_Manipulations_Beyond_Faces_A_Dataset_With_Human-Machine_Analysis_WACVW_2023_paper.pdf)
* Understanding Users' Deepfake Video Verification Strategies, *HCI* 2022: [Paper](https://link.springer.com/chapter/10.1007/978-3-031-19682-9_4)
* Deepfake Caricatures: Amplifying attention to artifacts, *arXiv* 2022: [Paper](https://arxiv.org/abs/2206.00535)

---

### 📍 Localization

* **LAA-Net**: Localized Artifact Attention Network, *CVPR* 2024: [Paper](https://arxiv.org/pdf/2401.13856) | [Github](https://github.com/10Ring/LAA-Net)
* Hierarchical Fine-Grained Image Forgery Detection, *CVPR* 2023: [Paper](https://arxiv.org/abs/2303.17111) | [Github](https://github.com/CHELSEA234/HiFi_IFDL)
* Face X-ray for More General Face Forgery Detection, *CVPR* 2020: [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Face_X-Ray_for_More_General_Face_Forgery_Detection_CVPR_2020_paper.pdf)
* DeepFake Detection Based on Discrepancies Between Faces and Context, *TPAMI* 2021: [Paper](https://ieeexplore.ieee.org/document/9468380/)

---

### 🎭 Multi-modal Based

* **AVFF**: Audio-Visual Feature Fusion, *CVPR* 2024: [Paper](https://arxiv.org/pdf/2406.02951v1)
* Lost in Translation: Lip-Sync Deepfake Detection, *CVPR Workshop* 2024: [Paper](https://openaccess.thecvf.com/content/CVPR2024W/WMF/html/Bohacek_Lost_in_Translation_Lip-Sync_Deepfake_Detection_from_Audio-Video_Mismatch_CVPRW_2024_paper.html)
* **AVoiD-DF**: Audio-Visual Joint Learning, *TIFS* 2023: [Paper](https://ieeexplore.ieee.org/abstract/document/10081373/)
* Joint Audio-Visual Deepfake Detection, *ICCV* 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Joint_Audio-Visual_Deepfake_Detection_ICCV_2021_paper.pdf)
* Emotions Don't Lie: Audio-Visual Deepfake Detection, *ACM MM* 2020: [Paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413570)

---

### 💓 Biological Signal

* Local attention and long-distance interaction of rPPG, *The Visual Computer* 2023: [Paper](https://link.springer.com/article/10.1007/s00371-023-02833-x)
* **FakeCatcher**: Detection using Biological Signals, *TPAMI* 2020: [Paper](https://ieeexplore.ieee.org/abstract/document/9141516/)
* **DeepRhythm**: Attentional Visual Heartbeat Rhythms, *ACM MM* 2020: [Paper](https://dl.acm.org/doi/10.1145/3394171.3413707)
* **DeepFakesON-Phys**: Heart Rate Estimation, *arXiv* 2020: [Paper](https://arxiv.org/abs/2010.00400) | [Github](https://github.com/BiDAlab/DeepFakesON-Phys)

---

### 🛡️ Robustness

* **LAA-Net**: Quality-Agnostic and Generalizable Detection, *CVPR* 2024: [Paper](https://arxiv.org/pdf/2401.13856) | [Github](https://github.com/10Ring/LAA-Net)
* Quality-Agnostic Deepfake Detection with Intra-model Collaborative Learning, *ICCV 2023*: [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Le_Quality-Agnostic_Deepfake_Detection_with_Intra-model_Collaborative_Learning_ICCV_2023_paper.pdf)
* Robust Image Forgery Detection Against OSN Transmission, *CVPR* 2022: [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9686650) | [Github](https://github.com/HighwayWu/ImageForensicsOSN)

---

### ⚖️ Fairness

* Preserving Fairness Generalization in Deepfake Detection, *CVPR* 2024: [Paper](https://arxiv.org/pdf/2402.17229) | [Github](https://github.com/Purdue-M2/Fairness-Generalization)
* **GBDF**: Gender Balanced DeepFake Dataset, *ICPR* 2022: [Paper](https://arxiv.org/abs/2207.10246) | [Github](https://github.com/aakash4305/GBDF)
* An Examination of Fairness of AI Models for Deepfake Detection, *IJCAI* 2021: [Paper](https://www.ijcai.org/proceedings/2021/0079.pdf)

---

### 🔏 Fingerprint/Watermark

* Responsible Disclosure of Generative Models Using Scalable Fingerprinting, *ICLR* 2022: [Paper](https://openreview.net/pdf?id=sOK-zS6WHB) | [Github](https://github.com/ningyu1991/ScalableGANFingerprints)
* DeepFake Disrupter: The Detector of DeepFake Is My Friend, *CVPR* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_DeepFake_Disrupter_The_Detector_of_DeepFake_Is_My_Friend_CVPR_2022_paper.html)
* **CMUA-Watermark**: Cross-Model Universal Adversarial Watermark, *AAAI* 2022: [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/19982) | [Github](https://github.com/VDIGPKU/CMUA-Watermark)
* Artificial Fingerprinting for Generative Models, *ICCV* 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Yu_Artificial_Fingerprinting_for_Generative_Models_Rooting_Deepfake_Attribution_in_Training_ICCV_2021_paper.html) | [Github](https://github.com/ningyu1991/ArtificialGANFingerprints)

---

### 🆔 Identity-Related

* **TI2Net**: Temporal Identity Inconsistency Network, *WACV* 2023: [Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_TI2Net_Temporal_Identity_Inconsistency_Network_for_Deepfake_Detection_WACV_2023_paper.pdf)
* Protecting Celebrities with Identity Consistency Transformer, *CVPR* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Protecting_Celebrities_From_DeepFake_With_Identity_Consistency_Transformer_CVPR_2022_paper.html) | [Github](https://github.com/LightDXY/ICT_DeepFake)
* **ID-Reveal**: Identity-aware DeepFake Video Detection, *ICCV* 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Cozzolino_ID-Reveal_Identity-Aware_DeepFake_Video_Detection_ICCV_2021_paper.html) | [Github](https://github.com/grip-unina/id-reveal)
* Protecting World Leaders Against Deep Fakes, *CVPR Workshop* 2019: [Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.pdf)

---

### ⚔️ Adversarial Attack

* Hiding Faces in Plain Sight: Defending DeepFakes by Disrupting Face Detection, *TDSC* 2025: [Paper](https://ieeexplore.ieee.org/abstract/document/11106399) | [Github](https://github.com/OUC-VAS/FacePoison)
* Poisoned Forgery Face: Towards Backdoor Attacks, *ICLR* 2024: [Paper](https://openreview.net/pdf?id=8iTpB4RNvP) | [Github](https://github.com/JWLiang007/PFF)
* Self-supervised Learning of Adversarial Example, *CVPR* 2022: [Paper](https://arxiv.org/abs/2203.12208) | [Github](https://github.com/liangchen527/SLADD)
* Exploring Frequency Adversarial Attacks, *CVPR* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Jia_Exploring_Frequency_Adversarial_Attacks_for_Face_Forgery_Detection_CVPR_2022_paper.pdf)
* Anti-Forgery: Stealthy and Robust DeepFake Disruption, *IJCAI* 2022: [Paper](https://www.ijcai.org/proceedings/2022/0107.pdf) | [Github](https://github.com/AbstractTeen/AntiForgery/)

---

### 🌍 Real Scenario

* Contrastive Pseudo Learning for Open-World DeepFake Attribution, *ICCV 2023*: [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Contrastive_Pseudo_Learning_for_Open-World_DeepFake_Attribution_ICCV_2023_paper.pdf)
* A Continual Deepfake Detection Benchmark, *WACV* 2023: [Paper](https://openaccess.thecvf.com/content/WACV2023/supplemental/Li_A_Continual_Deepfake_WACV_2023_supplemental.pdf)
* Seeing is Living? Rethinking Facial Liveness Verification, *USENIX* 2022: [Paper](https://www.usenix.org/conference/usenixsecurity22/presentation/li-changjiang)
* **DeepFake-o-meter**: An Open Platform for DeepFake Detection, *SP Workshop* 2021: [Paper](https://arxiv.org/abs/2103.02018)

---

### 🔮 Anomaly Detection

* Self-Supervised Video Forensics by Audio-Visual Anomaly Detection, *CVPR* 2023: [Paper](https://arxiv.org/abs/2301.01767) | [Github](https://github.com/cfeng16/audio-visual-forensics)
* Learning Second Order Local Anomaly, *CVPR* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Fei_Learning_Second_Order_Local_Anomaly_for_General_Face_Forgery_Detection_CVPR_2022_paper.html)

---

### 🔄 Self-Supervised Learning

* End-to-End Reconstruction-Classification Learning, *CVPR* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Cao_End-to-End_Reconstruction-Classification_Learning_for_Face_Forgery_Detection_CVPR_2022_paper.html)
* Leveraging Real Talking Faces via Self-Supervision, *CVPR* 2022: [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Haliassos_Leveraging_Real_Talking_Faces_via_Self-Supervision_for_Robust_Forgery_Detection_CVPR_2022_paper.html)
* **UIA-ViT**: Unsupervised Inconsistency-Aware Method, *ECCV* 2022: [Paper](https://link.springer.com/chapter/10.1007/978-3-031-20065-6_23) | [Github](https://github.com/wany0824/UIA-ViT)
* Dual Contrastive Learning for General Face Forgery Detection, *AAAI* 2022: [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20130)
* **MagDR**: Mask-guided Detection and Reconstruction, *CVPR* 2021: [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_MagDR_Mask-Guided_Detection_and_Reconstruction_for_Defending_Deepfakes_CVPR_2021_paper.html)

---

### 🔬 Source Model Attribution

* Deepfake Network Architecture Attribution, *AAAI* 2022: [Paper](https://aaai-2022.virtualchair.net/poster_aaai4380) | [Github](https://github.com/ICTMCG/DNA-Det)
* Artificial Fingerprinting for Generative Models, *ICCV* 2021: [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Artificial_Fingerprinting_for_Generative_Models_Rooting_Deepfake_Attribution_in_Training_ICCV_2021_paper.pdf) | [Github](https://github.com/ningyu1991/ArtificialGANFingerprints)
* Attributing Fake Images to GANs, *ICCV* 2019: [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Attributing_Fake_Images_to_GANs_Learning_and_Analyzing_GAN_Fingerprints_ICCV_2019_paper.pdf) | [Github](https://github.com/ningyu1991/GANFingerprints)

---

### 🤖 Transformer-Based Detection

* Towards More General Video-based Deepfake Detection, *CVPR* 2025: [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Towards_More_General_Video-based_Deepfake_Detection_through_Facial_Component_Guided_CVPR_2025_paper.pdf)
* Token-Level Shuffling and Mixing, *AAAI* 2025: [Paper](https://arxiv.org/pdf/2501.04376)
* Standing on Shoulders of Giants: Reprogramming VLM, *AAAI* 2025: [Paper](https://arxiv.org/pdf/2409.02664)
* **FakeFormer**: Efficient Vulnerability-Driven Transformers, *arXiv* 2024: [Paper](https://arxiv.org/pdf/2410.21964)
* A Timely Survey on Vision Transformer for Deepfake Detection, *arXiv* 2024: [Paper](https://arxiv.org/abs/2405.08463)

---

### 🧠 VLM-Based Detection

* Rethinking VLM in Face Forensics, *CVPR* 2025: [Paper](https://arxiv.org/pdf/2503.20188)
* Unlocking VLM Capabilities for Generalizable Detection, *ICML* 2025: [Paper](https://arxiv.org/pdf/2503.14853)
* **FakeShield**: Explainable Image Forgery Detection with MLLM, *ICLR* 2025: [Paper](https://arxiv.org/pdf/2410.02761)
* **TruthLens**: A Training-Free Paradigm, *arXiv* 2025: [Paper](https://arxiv.org/pdf/2503.15342)
* Unlocking Hidden Potential of CLIP, *arXiv* 2025: [Paper](https://arxiv.org/pdf/2503.19683) | [Code](https://github.com/yermandy/deepfake-detection)

---

### 🔄 Data Augmentation Methods

* **D3**: Scaling Up Deepfake Detection by Learning from Discrepancy, *CVPR* 2025: [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_D3_Scaling_Up_Deepfake_Detection_by_Learning_from_Discrepancy_CVPR_2025_paper.pdf)
* Plug-and-Play: Video-Level Blending and Spatiotemporal Adapter, *CVPR* 2025: [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_Generalizing_Deepfake_Video_Detection_with_Plug-and-Play_Video-Level_Blending_and_Spatiotemporal_CVPR_2025_paper.pdf)
* **DiffFake**: Differential Anomaly Detection, *WACV* 2025: [Paper](https://arxiv.org/pdf/2502.16247)
* Fake It till You Make It: Curricular Dynamic Forgery Augmentations, *ECCV* 2024: [Paper](https://arxiv.org/pdf/2409.14444)
* Detecting Deepfakes with Self-Blended Images, *CVPR* 2022: [Paper](https://arxiv.org/abs/2204.08376) | [Github](https://github.com/mapooon/SelfBlendedImages)

---

## 🛠️ Implementations

| Method | Venue | Code |
|--------|-------|:----:|
| **SIDA** | CVPR 2025 | [hzlsaber/SIDA](https://github.com/hzlsaber/SIDA) |
| **GenD** | WACV 2026 | [yermandy/GenD](https://github.com/yermandy/GenD) |
| **FakeSTormer** | ICCV 2025 | [10Ring/FakeSTormer](https://github.com/10Ring/FakeSTormer) |
| **PwTF-DVD** | ICCV 2025 | [rama0126/PwTF-DVD](https://github.com/rama0126/PwTF-DVD) |
| **AdvOU** | ICCV 2025 | [liacaaa/AdvOU](https://github.com/liacaaa/AdvOU) |
| **D4** | WACV 2024 | [nmangaokar/wacv_24_d4](https://github.com/nmangaokar/wacv_24_d4) |
| **LFM** | BMVC 2025 | [lmlpy/LFM](https://github.com/lmlpy/LFM) |
| **DFS-GDD** | BMVC 2024 | [leaffeall/DFS-GDD](https://github.com/leaffeall/DFS-GDD) |
| **DeepfakeBench** | NeurIPS 2023 | [SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) |
| **FaceForensics++** | ICCV 2019 | [ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) |
| **F3-Net** | ECCV 2020 | [yyk-wew/F3-Net](https://github.com/yyk-wew/F3-Net) |
| **DIRE** | ICCV 2023 | [ZhendongWang6/DIRE](https://github.com/ZhendongWang6/DIRE) |
| **UnivFD** | CVPR 2023 | [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) |
| **LAA-Net** | CVPR 2024 | [10Ring/LAA-Net](https://github.com/10Ring/LAA-Net) |
| **TALL** | ICCV 2023 | [rainy-xu/TALL4Deepfake](https://github.com/rainy-xu/TALL4Deepfake) |
| **SBI** | CVPR 2022 | [mapooon/SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages) |
| **CADDM** | CVPR 2023 | [megvii-research/CADDM](https://github.com/megvii-research/CADDM) |
| **ICT** | CVPR 2022 | [LightDXY/ICT_DeepFake](https://github.com/LightDXY/ICT_DeepFake) |
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
