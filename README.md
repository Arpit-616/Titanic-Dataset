# 🚢 Titanic Dataset – Data Preprocessing & Feature Engineering

This project performs complete data preprocessing on the Titanic dataset to prepare it for Machine Learning model training.

---

## 📂 Project Overview

The objective of this project is to clean and preprocess the Titanic dataset by:

- Handling missing values
- Encoding categorical variables
- Removing outliers
- Standardizing numerical features
- Preparing clean feature matrix for ML models

---

## 📊 Dataset Information

- Total Records: 891
- Features: 12
- Target Variable: `Survived`

### Key Features:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

---

## ⚙️ Preprocessing Steps

### 1️⃣ Data Exploration
- Checked dataset structure using `.info()`
- Identified missing values using `.isnull().sum()`

### 2️⃣ Handling Missing Values
- Numerical columns filled using **Mean Imputation**
- Categorical columns filled using **Most Frequent Value**

### 3️⃣ Feature Selection
Dropped unnecessary columns:
- PassengerId
- Name
- Ticket
- Cabin

### 4️⃣ Encoding
Applied **One-Hot Encoding** for:
- Sex
- Embarked

### 5️⃣ Outlier Removal
Used **IQR Method** to remove outliers from:
- Age
- Fare

### 6️⃣ Feature Scaling
Applied **StandardScaler** to normalize numerical features.

---

## 📈 Final Dataset

- Records after cleaning: 718
- Features after encoding: 8
- No missing values
- All features numeric
- Ready for ML model training

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## 🚀 How to Run

Install required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
