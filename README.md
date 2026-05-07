# Pokémon Combat Prediction using SVM

This project aims to predict the outcome of Pokémon battles using Support Vector Machines (SVM). It involves comprehensive data analysis, feature engineering, model tuning via cross-validation, and performance evaluation using various metrics including ROC/AUC.

## Project Structure

```text
.
├── data/                   # Processed data for training and evaluation
│   ├── holdout/            # 80/20 split data
│   └── cross_validation/   # 5-fold CV split data
├── models/                 # Saved SVM models
│   ├── holdout/            # 使用data/holdout/train.scale訓練
│   └── cross_validation/   # 使用data/cross_validation/train_ fold*                
├── merge_data.py       # Merges pokemon stats with combat records
├── data_transform.py   # 資料前處理
├── feature_analysis.py # 找尋重要的特徵
├── split_data.py       # Data splitting (Holdout & CV)
├── train_model.py      # train model (Holdout & CV)
├── predict_model.py    # Prediction and metric calculation
├── roc_plot.py         # ROC curve generation and AUC calculation
├── pokemon.csv         # Original Pokémon dataset
├── combats.csv         # Original combat records
├── merged_data.csv     # 將pokemon.csv和combats.csv合併的資料
├── pokemon_data.txt    # 經過data_transform，只保留重要的特徵
├── pokemon_data.scale  # 經過svm-scale
├── 資料探勘hw2.pdf      # 完整報告
└── svm-*.exe           # LIBSVM executables (Windows)

```

## Key Features

### 1. Feature Engineering
Based on Exploratory Data Analysis (EDA), we found that relative differences between Pokémon stats are more predictive than absolute values for SVM.
- **Diff Features**: Calculated `Diff_Speed`, `Diff_Attack`, `Diff_HP`, etc.
- **Categorical Features**: Handled `Legendary` status and Pokémon `Types`.
- **Scaling**: All features are scaled to a specific range (e.g., [-1, 1]) using `svm-scale` to ensure optimal SVM performance.

### 2. Validation Strategies
- **Hold-out Validation**: An 80/20 split to evaluate the final model's generalization.
- **5-fold Cross-Validation**: Used to tune hyperparameters (Kernel type, C, Gamma) and select the best model configuration.

### 3. Evaluation Metrics
The project evaluates models using:
- Confusion Matrix
- Accuracy, Precision, Recall (TPR), F1-Score
- FPR, TNR, FNR
- **ROC Curve and AUC**: Specifically implemented to evaluate the model's discriminative ability across different thresholds.

## Getting Started

### Prerequisites
- Python 3.x
- LIBSVM executables (`svm-train`, `svm-predict`, `svm-scale`)
- Python libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn` (for data handling only)

### Workflow

1. **Preprocessing**:
   Merge and transform the raw data into LIBSVM format.
   ```bash
   python merge_data.py
   python data_transform.py
   ```

2. **Data Splitting**:
   Prepare files for hold-out and cross-validation.
   ```bash
   python split_data.py
   ```

3. **Model Training**:
   Run the training script to train multiple SVM configurations.
   ```bash
   python train_model.py
   ```

4. **Evaluation**:
   Evaluate model performance and generate reports.
   ```bash
   python predict_model.py
   python roc_plot.py
   ```

## Reports
For detailed analysis and experimental results, please refer to the following reports:
- **[Question 1: Feature Analysis](Question1_Report.md)**
- **[Question 2: Hold-out Validation](Question2_Report.md)**
- **[Question 3: Cross-Validation](Question3_Report.md)**
- **[Question 4: Probability & ROC](Question4_Report.md)**
