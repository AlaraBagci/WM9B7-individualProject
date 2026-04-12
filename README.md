# SortSmart: Deep Learning Waste Classification for Mobile and Industrial Deployment

## WM9B7 — Artificial Intelligence & Deep Learning | Individual Assessment

This project addresses automated waste material classification by comparing three deep learning architectures across a **mobile-to-industrial deployment spectrum** on the **RealWaste** dataset (Single et al., 2023).

### Results

| Model | Test Accuracy | F1-Score | Params | Size | Target Deployment |
|---|---|---|---|---|---|
| Baseline CNN (scratch) | 33.61% | 31.73% | 1.2M | 4.6 MB | Reference only |
| EfficientNet-B0 | 73.11% | 73.17% | 4.0M | 15.3 MB | Mobile app |
| **ConvNeXt-Tiny** | **94.12%** | **94.24%** | 27.8M | 106.2 MB | MRF / industrial edge |
| *Paper best (Inception V3)* | *89.19%* | *90.25%* | *~24M* | *~96 MB* | *Single et al. (2023)* |

ConvNeXt-Tiny surpasses the original paper's best result by **+4.93 percentage points**.

---

## How to Run

### Prerequisites
- Python 3.11 or 3.12
- NVIDIA GPU recommended (tested on Colab T4, 15 GB VRAM)

### Steps
1. Clone this repository.
2. Download the [RealWaste dataset](https://www.kaggle.com/datasets/joebeachcapital/realwaste) from Kaggle.
3. Extract so the folder contains 9 class subfolders (`Cardboard/`, `Food Organics/`, `Glass/`, etc.).
4. Place the `RealWaste/` folder next to the notebook, or update the path in the dataset loading cell.
5. Open `SortSmart_Notebook.ipynb` in **VS Code** or **Google Colab**.
6. Press **"Run All"**. All dependencies install automatically in Cell 1.

### Google Colab
- Upload dataset to Google Drive at `/content/drive/MyDrive/RealWaste/`
- Select **T4 GPU** runtime: Runtime → Change runtime type → T4 GPU
- Runtime: ~45–60 minutes

---

## Project Structure

```
├── SortSmart_Notebook.ipynb          # Main notebook (Part 1)
└── README.md                         # This file
```

## Dataset

- **RealWaste** — 4,752 images, 9 waste classes, captured from Whyte Gully landfill (Australia)
- **Licence:** CC BY-NC-SA 4.0
- **Paper:** Single, S., Iranmanesh, S. & Raad, R. (2023). RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning. *Information*, 14(12), 633.

## Key Features
- Mixed precision training (AMP) for GPU memory efficiency
- RAM-cached dataset for fast epoch iteration
- Two-phase transfer learning (frozen head → full fine-tuning)
- Weighted cross-entropy + weighted sampling for class imbalance
- Grad-CAM interpretability analysis
- Deployment profiling (params, model size, inference latency)
