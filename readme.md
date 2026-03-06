# Advanced Statistical Learning Individual Assignment

This project implements and compares several binary classification models for predicting ICU mortality (`icu_death_flag`) using a MIMIC-style structured dataset. 
The main analysis is contained in `pipeline.ipynb`, which includes data preprocessing, feature filtering, baseline model training, L1-based feature selection, model evaluation, and result export.

## Main File

The main file of this project is:

- `pipeline.ipynb`

This notebook contains the full end-to-end workflow and is the primary file to run for reproducing the analysis.

## What the Pipeline Does

The notebook performs the following steps:

1. Loads the dataset from `data/mimic_static.csv`.
2. Removes ID columns, leakage-related columns, and optional time-related columns.
3. Filters features based on missingness.
4. Examines low-variance features.
5. Splits the data into training and test sets using stratified sampling.
6. Builds preprocessing pipelines for numeric and categorical variables.
7. Trains baseline models:
   - Logistic Regression
   - Support Vector Machine with RBF kernel
   - Random Forest
   - Gradient Boosting
8. Selects classification thresholds on a validation split using F1 on the precision-recall curve.
9. Saves predictions, metrics, confusion matrices, ROC curves, and PR curves.
10. Performs L1-regularized feature selection.
11. Retrains the same models on the L1-selected feature set.
12. Saves summary tables and combined comparison plots.

## Environment Setup

It is recommended to use a dedicated Python or conda environment.

### Option 1: Using conda

```bash
conda create -n yue python=3.10 -y
conda activate yue
pip install -r requirements.txt
```

### Option 2: Using pip only

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Optional: Add the Environment to Jupyter

If you want this environment to appear in Jupyter Lab as a selectable kernel:

```bash
python -m ipykernel install --user --name yue
```

## How to Run

### 1. Make sure the dataset is in the expected location

The notebook expects the input file at:

```text
data/mimic_static.csv
```

### 2. Start Jupyter Lab

```bash
jupyter lab
```

### 3. Open the notebook

Open:

```text
pipeline.ipynb
```

### 4. Run all cells from top to bottom

The notebook is designed to be executed sequentially. Running all cells will reproduce the full workflow and generate the outputs automatically.

## Output Files

After running the notebook, results will be saved under:

```text
outputs_final/
```

We have a total of two subfolders, baseline and 11_selected, in each subfolder the following contents are included, 
- ROC and PR curves
- confusion matrices
- prediction files (`.csv`)
- summary of metrics 
