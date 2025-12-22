# 📈 Performance Benchmarks

This directory contains quantitative and qualitative benchmark results from our survey.

## Quantitative Results

See `quantitative/` folder for:
- Cross-dataset evaluation tables
- Compression robustness analysis
- Model efficiency comparisons

## Qualitative Results

See `qualitative/` folder for:
- Visualization of detection heatmaps
- Failure case analysis
- Comparison of manipulation artifacts

## Key Findings

### Cross-Dataset Generalization
- Models trained on FF++ typically see 15-25% AUC drop on Celeb-DF
- Frequency-aware methods show better generalization
- Diffusion-aware detectors struggle with GAN-generated content and vice versa

### Compression Robustness
- Heavy compression (c40) reduces detection accuracy by ~15%
- Frequency-domain methods are more sensitive to compression
- Multi-scale approaches show better robustness

### Efficiency vs. Accuracy Trade-off
- Lightweight models (MesoNet) achieve ~150 FPS with moderate accuracy
- Transformer models achieve highest accuracy but ~30-50 FPS
- Hybrid approaches balance both aspects
