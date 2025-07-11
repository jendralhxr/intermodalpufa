# GC-FID ↔ FT-NIR Latent Alignment

This project implements a cross-modal autoencoder framework to align gas chromatography (GC-FID) and FT-NIR spectral data in a shared latent space. The goal is to enable signal translation or fusion between two complementary sensing modalities, even with partially paired data.

## 🧠 Model Overview

- Two encoders: one for GC-FID chromatograms, one for FT-NIR spectra.
- Two decoders for reconstructing input modalities.
- A shared latent space trained with:
  - Reconstruction loss (MSE)
  - Latent alignment loss (MSE between GC and NIR embeddings)

## 🔧 Requirements
- Python 3.7+
- PyTorch
- NumPy
