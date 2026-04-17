# SortSmart: Deep Learning Waste Classification for Mobile and Industrial Deployment

## WM9B7 — Artificial Intelligence & Deep Learning | Individual Assessment

This project addresses automated waste material classification by comparing three deep learning architectures across a **mobile-to-industrial deployment spectrum** on the **RealWaste** dataset (Single et al., 2023). The main notebook has been updated to run in **Azure Machine Learning Studio** with a **Python 3.10 ML kernel**, while the original fully trained notebook with outputs remains available for the longer Google Colab run.

### Results

| Model | Test Accuracy | F1-Score | Params | Size | Target Deployment |
|---|---|---|---|---|---|
| Baseline CNN (scratch) | 33.61% | 31.73% | 1.2M | 4.6 MB | Reference only |
| EfficientNet-B0 | 73.11% | 73.17% | 4.0M | 15.3 MB | Mobile app |
| **ConvNeXt-Tiny** | **94.12%** | **94.24%** | 27.8M | 106.2 MB | MRF / industrial edge |
| *Paper best (Inception V3)* | *89.19%* | *90.25%* | *~24M* | *~96 MB* | *(Single et al., 2023)* |

ConvNeXt-Tiny surpasses the original paper's best result by **+4.93 percentage points**.

---

## How to Run

### Prerequisites
- Azure Machine Learning Studio
- Python 3.10 ML kernel
- NVIDIA GPU recommended, such as an Azure Standard T4 GPU

### Steps
1. Open `SortSmart_Notebook.ipynb` in **Azure Machine Learning Studio**.
2. Select the **Python 3.10** ML kernel.
3. Press **"Run All"**. Dependencies install automatically in the setup cell.
4. The notebook downloads the **RealWaste** dataset automatically from the UCI Archive at runtime, so no manual dataset download is required.

### Runtime Note
- The notebook uses **reduced training settings** so the full pipeline completes in about **15 minutes** for reproducibility.
- These shorter runs are intended to showcase the architecture and workflow.
- The full reported results in the project reflection were obtained with longer training runs and early stopping.

### Original Notebook
- `SortSmart_Notebook_with_outputs.ipynb` is the original notebook with saved outputs.
- It was designed for **Google Colab**.
- Full training took about **5–6 hours** to complete in Colab.
- Use this notebook if you want to inspect the original end-to-end results and figures.

---

## Project Structure

```
├── SortSmart_Notebook.ipynb          # Main notebook (Part 1)
├── SortSmart_Notebook_with_outputs.ipynb  # Original Colab notebook with outputs
└── README.md                         # This file
```

## Dataset

- **RealWaste** — 4,752 images, 9 waste classes, captured from Whyte Gully landfill (Australia)
- **Licence:** CC BY-NC-SA 4.0
- **Paper:** [Single, S., Iranmanesh, S. and Raad, R. (2023) ‘RealWaste: a novel real-life data set for landfill waste classification using deep learning’, *Information*, 14(12), 633.](https://www.mdpi.com/2078-2489/14/12/633)
- **Access:** downloaded automatically from the [UCI Archive dataset page](https://archive.ics.uci.edu/dataset/908/realwaste) when the notebook runs

### Citations / Acknowledgements

- **Dataset link:** https://archive.ics.uci.edu/dataset/908/realwaste
- **Paper link:** https://www.mdpi.com/2078-2489/14/12/633

S. Single, S. Iranmanesh, R. Raad, RealWaste, electronic dataset, The UCI Machine Learning Repository, Wollongong City Council, CC BY 4.0

## Key Features
- Mixed precision training (AMP) for GPU memory efficiency
- RAM-cached dataset for fast epoch iteration
- Two-phase transfer learning (frozen head → full fine-tuning)
- Weighted cross-entropy + weighted sampling for class imbalance
- Grad-CAM interpretability analysis
- Deployment profiling (params, model size, inference latency)
- Reduced-epoch reproducibility mode to keep runtime near 15 minutes in Azure ML Studio
