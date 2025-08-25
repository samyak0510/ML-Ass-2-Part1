# Binary Classification with Fully-Connected Neural Network (Assignment 2)

This repository contains my coursework for an Intro to ML assignment focused on binary classification with a fully-connected neural network (FC-NN) in PyTorch.

- Notebook: `Samyak_Shah_Nischal_Seemantula_assignment2_checkpoint.ipynb`
- Dataset: `datasets/dataset.csv`
- Assignment PDF: `fall24_cse574_d_Assignment_2.pdf`

### Key Results
- Best test accuracy: 87.01% (k-fold selected model)
- Architecture: Input → 64 → 64 → 1 with BatchNorm and Dropout
- Training objective/optimizer: BCELoss + SGD (with momentum), plus ablation of Adam and RMSprop

### Project Highlights (as on my resume)
- Achieved 87% test accuracy on a binary classification task by architecting and tuning an FC-NN (Input-64-64-1) with BatchNorm and Dropout, trained using SGD and BCELoss.
- Engineered a robust ML pipeline: preprocessed data with StandardScaler and IQR outlier handling; implemented train/val/test splits; evaluated performance using accuracy, precision/recall/F1, and ROC analysis via torchmetrics.
- Optimized model performance through systematic hyperparameter sweeps (dropout, learning rate), optimizer ablation studies, k-fold cross-validation, and StepLR scheduling, utilizing early stopping.

## 1) Problem & Data
Binary classification on a tabular dataset stored at `datasets/dataset.csv`. The notebook performs data cleaning, numeric type coercion, IQR-based outlier handling, and scaling.

- Cleaning: coerces invalid/missing tokens to numeric and handles anomalies
- Outliers: IQR rule with whiskers for robust filtering
- Scaling: `StandardScaler` on features
- Splits: train/validation/test splits implemented in the notebook

## 2) Model
- Architecture: `Input → Linear(64) → BN → LeakyReLU → Dropout → Linear(64) → BN → LeakyReLU → Dropout → Linear(1) → Sigmoid`
- Loss: `nn.BCELoss()`
- Optimizer: `torch.optim.SGD` (momentum=0.9) for the main training, with ablations for `Adam` and `RMSprop`
- Regularization: `Dropout` and `BatchNorm1d`

## 3) Training & Evaluation
- Metrics: accuracy, precision, recall, F1; ROC analysis with `torchmetrics`
- Tuning: dropout rate sweep, learning-rate sweep; hidden-size exploration
- Optimization studies: optimizer ablation (SGD vs Adam vs RMSprop)
- Scheduling: `StepLR(step_size=50, gamma=0.1)`
- Early stopping: validation-accuracy-based early stopping with patience
- K-Fold CV: `KFold(n_splits=5)`; tracked per-fold train/val/test curves

### Reported Outcomes (from the notebook)
- Base model test accuracy (initial run): ~80.52%
- Tuned/k-fold selected model test accuracy: ~87.01%
 
## 4) Repository Structure
- `datasets/` — raw dataset CSV
- `Samyak_Shah_Nischal_Seemantula_assignment2_checkpoint.ipynb` — Jupyter notebook
- `report/` — assignment report PDF
- `fall24_cse574_d_Assignment_2.pdf` — assignment handout
- `README.md` — this file
- `requirements.txt` — Python dependencies
- `LICENSE` — MIT license

## 5) Reproducibility
1. Create and activate a virtual environment (recommended):
   - Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter and run the checkpoint notebook end-to-end:
   ```bash
   python -m ipykernel install --user --name ml-assignment2
   jupyter notebook Samyak_Shah_Nischal_Seemantula_assignment2_checkpoint.ipynb
   ```

Notes:
- GPU is optional; CPU is sufficient for this assignment-scale workload.
- If you use GPU, install the appropriate CUDA build of PyTorch from the official site.

## 6) License
This project is licensed under the `MIT LICENSE`.

## 7) Acknowledgements
- University assignment context as per `fall24_cse574_d_Assignment_2.pdf`