# -----------------------------
# 1. Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 2. Load Dataset
# -----------------------------

df = pd.read_csv("Titanic-Dataset.csv")

print("Original Shape:", df.shape)
print("\nDataset Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# 3. Drop Irrelevant Columns
# -----------------------------
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)


# -----------------------------
# 4. Handle Missing Values
# -----------------------------
# Numerical columns
num_cols = ["Age", "Fare"]
num_imputer = SimpleImputer(strategy="mean")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical column
cat_cols = ["Embarked"]
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("\nMissing Values After Imputation:")
print(df.isnull().sum())


# -----------------------------
# 5. One-Hot Encoding
# -----------------------------
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

print("\nDataset After Encoding:")
print(df.head())


# -----------------------------
# 6. Outlier Detection (Age & Fare)
# -----------------------------
for col in ["Age", "Fare"]:
    plt.figure()
    plt.boxplot(df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Remove outliers using IQR
for col in ["Age", "Fare"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("Shape After Outlier Removal:", df.shape)


# -----------------------------
# 7. Feature Scaling
# -----------------------------
X = df.drop("Survived", axis=1)
y = df["Survived"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_final = pd.DataFrame(X_scaled, columns=X.columns)

print("\nFinal Feature Shape:", X_final.shape)
print("Target Shape:", y.shape)

print("\nPreprocessing Completed Successfully ✅")

# -----------------------------
# 8. Save Preprocessed Dataset
# -----------------------------
output_path = "Titanic-Dataset-Preprocessed.csv"
df.to_csv(output_path, index=False)

print(f"Preprocessed dataset saved to {output_path}")
