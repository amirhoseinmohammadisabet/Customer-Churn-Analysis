import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found. Please make sure the file is in the same directory.")
    # Exit or handle the error appropriately if the file isn't found
    exit()

# --- 1. Initial Data Inspection ---

print("\n--- 1. Initial Data Inspection ---")
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nDataset Description (Numerical columns):")
print(df.describe())

print("\nShape of the dataset (Rows, Columns):")
print(df.shape)

# Check for duplicate customer IDs
print(f"\nNumber of unique Customer IDs: {df['customerID'].nunique()}")
if df['customerID'].nunique() == df.shape[0]:
    print("No duplicate Customer IDs found.")
else:
    print("Duplicate Customer IDs found. Investigation needed.")

# --- 2. Check for Missing Values ---

print("\n--- 2. Missing Values Check ---")
print(df.isnull().sum())

# 'TotalCharges' is often loaded as object due to some non-numeric values (spaces)
# Convert 'TotalCharges' to numeric, coercing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Re-check missing values after conversion
print("\nMissing values after 'TotalCharges' conversion:")
print(df.isnull().sum())

# Handle missing 'TotalCharges' - often these correspond to new customers with tenure 0
# For simplicity in EDA, we'll fill with 0 or drop. For now, let's see how many.
missing_total_charges = df[df['TotalCharges'].isnull()]
print(f"\nRows with missing TotalCharges: {missing_total_charges.shape[0]}")
print("These are likely new customers with 0 tenure and 0 total charges.")

# Fill NaN in TotalCharges with 0, as these are typically new customers with tenure 0
# You might choose to drop them if your analysis requires non-zero total charges,
# but for churn prediction, keeping them and treating their total charges as 0 is usually fine.
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print("\nMissing values after filling 'TotalCharges' with 0:")
print(df.isnull().sum())

# --- 3. Explore Target Variable: Churn ---

print("\n--- 3. Target Variable Analysis (Churn) ---")
print("Distribution of Churn:")
print(df['Churn'].value_counts())
print("\nPercentage distribution of Churn:")
print(df['Churn'].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Distribution of Customer Churn')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.show()

# --- 4. Explore Categorical Features ---

print("\n--- 4. Categorical Features Analysis ---")
# Exclude 'customerID' and numerical columns (tenure, MonthlyCharges, TotalCharges)
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('Churn') # Churn is our target, we've already analyzed it
# We need to explicitly remove 'customerID' as it's an object but not categorical for plotting
if 'customerID' in categorical_cols:
    categorical_cols.remove('customerID')

# Plot distribution of each categorical feature and their relation with Churn
fig, axes = plt.subplots(len(categorical_cols), 2, figsize=(18, 5 * len(categorical_cols)))
axes = axes.flatten() # Flatten for easier iteration

for i, col in enumerate(categorical_cols):
    # Plot distribution of the feature
    sns.countplot(x=col, data=df, palette='viridis', ax=axes[2*i])
    axes[2*i].set_title(f'Distribution of {col}')
    axes[2*i].tick_params(axis='x', rotation=45)

    # Plot distribution vs Churn
    sns.countplot(x=col, hue='Churn', data=df, palette='magma', ax=axes[2*i + 1])
    axes[2*i + 1].set_title(f'{col} vs Churn')
    axes[2*i + 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# More detailed look at some interesting categorical features vs churn
print("\n--- Detailed Churn Rate by Key Categorical Features ---")

for col in ['Contract', 'InternetService', 'PaymentMethod', 'Partner', 'Dependents', 'SeniorCitizen']:
    if col in df.columns:
        print(f"\nChurn Rate by {col}:")
        churn_rate = df.groupby(col)['Churn'].value_counts(normalize=True).unstack().get('Yes', 0)
        print(churn_rate.sort_values(ascending=False))
        # Plotting the churn rate for better visualization
        plt.figure(figsize=(8, 5))
        sns.barplot(x=churn_rate.index, y=churn_rate.values, palette='coolwarm')
        plt.title(f'Churn Rate by {col}')
        plt.xlabel(col)
        plt.ylabel('Churn Rate')
        plt.ylim(0, 1) # Ensure y-axis from 0 to 1 for percentage
        plt.show()


# --- 5. Explore Numerical Features ---

print("\n--- 5. Numerical Features Analysis ---")
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(15, 5 * len(numerical_cols)))

for i, col in enumerate(numerical_cols):
    # Distribution Plot (Histogram/KDE)
    sns.histplot(df[col], kde=True, ax=axes[i, 0], palette='viridis')
    axes[i, 0].set_title(f'Distribution of {col}')

    # Boxplot vs Churn
    sns.boxplot(x='Churn', y=col, data=df, ax=axes[i, 1], palette='magma')
    axes[i, 1].set_title(f'{col} vs Churn')

plt.tight_layout()
plt.show()

# Correlation Matrix for numerical features
print("\n--- 6. Correlation Matrix (Numerical Features) ---")
numerical_df = df[numerical_cols + ['Churn']].copy()
numerical_df['Churn'] = numerical_df['Churn'].map({'Yes': 1, 'No': 0}) # Convert Churn to numeric for correlation

plt.figure(figsize=(8, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

print("\n--- EDA Complete. Initial Insights Gained. ---")