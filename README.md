# Federated CT Pulmonary Diagnosis (Non-COVID) — Reference Implementation (FedAvg)

This repository provides a **fully runnable Python/PyTorch** reference implementation of a paper-style pipeline:

**Federated Deep Learning for Privacy-Preserving Chest CT Analysis in Pulmonary Disease Diagnosis (Non-COVID)**

It includes:
- a **simple FedAvg** simulator (no external FL framework required)
- a **CT volume dataset interface** (expects `.npz` volumes)
- a compact **2.5D CNN** baseline (slice-based training + volume aggregation)
- **centralized baseline** training for comparison
- evaluation utilities and reproducible configs
- a **toy dataset generator** so you can run end-to-end immediately

> Note: This code is designed to be **practical, readable, and reproducible**. It is not tied to any single dataset release.
> For real CT datasets (e.g., LIDC-IDRI), you will first export volumes into `.npz` format (see below).

---

## 1) Setup

### Create environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Data format

### Expected per-sample file format
Each CT study should be stored as a single `.npz` file with:
- `volume`: `float32` array of shape `(D, H, W)` (HU-normalized or standardized)
- `label`: integer class label `0..(C-1)`

Example:
```
data/
  client_01/
    train/
      case_0001.npz
      case_0002.npz
    val/
      case_0101.npz
    test/
      case_0201.npz
  client_02/
    train/ ...
```

This layout naturally maps **each client folder** to a *federated site*.

---

## 3) Quickstart (runs end-to-end)

### (A) Generate a toy multi-client dataset
```bash
python tools/make_toy_dataset.py --out data --clients 3 --classes 3 --train 120 --val 30 --test 30
```

### (B) Run federated training (FedAvg)
```bash
python train_federated.py --data_root data --rounds 20 --clients_per_round 3
```

### (C) Run centralized training baseline
```bash
python train_centralized.py --data_root data --epochs 10
```

### (D) Evaluate a checkpoint
```bash
python evaluate.py --data_root data --ckpt runs/federated/best.pt --split test
```

Outputs (metrics, logs) are stored in:
- `runs/federated/`
- `runs/centralized/`

---

## 4) Using real CT data (e.g., LIDC-IDRI)

1. Convert each CT study into a `(D,H,W)` `float32` numpy array.
2. Apply **basic harmonization**:
   - resample spacing (optional, if you have spacing metadata)
   - clip HU (e.g., `[-1000, 400]`) and normalize (e.g., z-score or min-max)
   - optionally apply a lung mask (optional)
3. Save as `.npz` with `volume` and `label`.

This project does **not** include a DICOM/NIfTI converter because CT datasets vary in licensing and formats.
However, the dataset interface is intentionally simple so you can plug in your exporter.

---

## 5) Notes on privacy/security

This simulator demonstrates:
- **Client-side local training**
- **Server-side aggregation**
- optional **secure aggregation placeholder**
- optional **differential privacy noise** on model updates (simple Gaussian mechanism)

For a production setting, you would replace the placeholder with a cryptographic secure aggregation protocol and
a calibrated DP accountant.

---

## 6) Reproducibility

All scripts accept `--seed`. Determinism will still depend on your CUDA/cuDNN settings if using GPU.

---

## License
MIT (for the code in this repository).
