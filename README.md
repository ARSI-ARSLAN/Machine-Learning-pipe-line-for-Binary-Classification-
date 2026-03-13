# Machine Learning Pipeline: Serial vs Parallel vs GPU Processing

This project implements a **machine learning pipeline** to compare the performance of three different execution strategies:

- **Serial Processing (Single Core)**
- **Parallel Processing (Multicore CPU)**
- **GPU Processing (CUDA with XGBoost)**

The goal of the project is to evaluate how **parallelism and GPU acceleration affect training time and model performance** in a typical machine learning workflow.

---

# Project Overview

The pipeline performs the following steps:

1. Load dataset from CSV
2. Data preprocessing
   - Handle missing values
   - Standardize numeric features
   - Encode categorical features
3. Train classification model
4. Evaluate performance using:
   - Accuracy
   - F1 Score
   - Confusion Matrix
5. Measure **processing time**
6. Compare results across:
   - Serial
   - Parallel
   - GPU implementations
7. Visualize results using **Matplotlib**

---

# Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib

---

# Dataset Format

The dataset should be a CSV file with the following columns:

```
feature_1
feature_2
feature_3
feature_4
feature_5
feature_6
feature_7
target
```

- **Numeric Features**
  - feature_1
  - feature_2
  - feature_4
  - feature_6
  - feature_7

- **Categorical Features**
  - feature_3
  - feature_5

- **Target**
  - Binary classification label

Example file used in this project:

```
pdc_dataset_with_target.csv
```

---

# Project Structure

```
project-folder
│
├── main.py
├── pdc_dataset_with_target.csv
├── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install required dependencies:

```bash
pip install pandas scikit-learn xgboost matplotlib
```

---

# How to Run

Run the pipeline:

```bash
python main.py
```

The script will:

1. Train the **serial model**
2. Train the **parallel model**
3. Train the **GPU model**
4. Print metrics and execution time
5. Display performance comparison graphs

---

# Evaluation Metrics

The project evaluates models using:

- **Accuracy Score**
- **F1 Score**
- **Confusion Matrix**
- **Processing Time**

---

# Output Example

```
===== Serial Version (No Parallelism) =====
Accuracy: 0.89
F1 Score: 0.87
Processing Time: 12.45 seconds

===== Parallel Version (Multicore) =====
Accuracy: 0.90
F1 Score: 0.88
Processing Time: 4.21 seconds

===== GPU Version (CUDA XGBoost) =====
Accuracy: 0.91
F1 Score: 0.89
Processing Time: 2.10 seconds

Time Reduction: 70%+
```

---

# Visualization

The script generates plots comparing:

- Accuracy vs F1 Score
- Processing Time

This helps visually understand the performance improvement of **parallel and GPU computing**.

---

# Key Features

- End-to-end ML pipeline
- Automated preprocessing
- Multicore parallel training
- GPU acceleration using XGBoost
- Performance benchmarking
- Result visualization

---

# Future Improvements

- Add hyperparameter tuning
- Support larger datasets
- Add cross-validation
- Deploy as an API
- Integrate experiment tracking

---

# Author

Developed as part of a **Machine Learning / Parallel Computing project**.
