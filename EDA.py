import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (ensure this path is correct)
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found. Please make sure the file is in the same directory.")
    exit()

# --- Initial Data Cleaning (from your EDA, necessary for preprocessing) ---
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# --- Updated EDA Section (to address FutureWarnings) ---
print("\n--- Updated EDA for Warning Fixes ---")

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
categorical_cols = df.select_dtypes(include='object').columns.tolist()
if 'Churn' in categorical_cols:
    categorical_cols.remove('Churn')
if 'customerID' in categorical_cols:
    categorical_cols.remove('customerID')

fig, axes = plt.subplots(len(categorical_cols), 2, figsize=(18, 5 * len(categorical_cols)))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    # Plot distribution of the feature - Fix for FutureWarning
    sns.countplot(x=col, data=df, palette='viridis', ax=axes[2*i], hue=col, legend=False)
    axes[2*i].set_title(f'Distribution of {col}')
    axes[2*i].tick_params(axis='x', rotation=45)

    # Plot distribution vs Churn
    sns.countplot(x=col, hue='Churn', data=df, palette='magma', ax=axes[2*i + 1])
    axes[2*i + 1].set_title(f'{col} vs Churn')
    axes[2*i + 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("\n--- Detailed Churn Rate by Key Categorical Features ---")
for col in ['Contract', 'InternetService', 'PaymentMethod', 'Partner', 'Dependents', 'SeniorCitizen']:
    if col in df.columns:
        print(f"\nChurn Rate by {col}:")
        # Ensure 'Yes' is handled correctly if it might not exist for some groups
        churn_rate_series = df.groupby(col)['Churn'].value_counts(normalize=True).unstack().get('Yes', pd.Series(0, index=df[col].unique()))
        print(churn_rate_series.sort_values(ascending=False))
        plt.figure(figsize=(8, 5))
        sns.barplot(x=churn_rate_series.index, y=churn_rate_series.values, palette='coolwarm')
        plt.title(f'Churn Rate by {col}')
        plt.xlabel(col)
        plt.ylabel('Churn Rate')
        plt.ylim(0, 1)
        plt.show()

# --- 5. Explore Numerical Features ---
print("\n--- 5. Numerical Features Analysis ---")
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(15, 5 * len(numerical_cols)))

for i, col in enumerate(numerical_cols):
    # Distribution Plot (Histogram/KDE) - Fix for FutureWarning
    sns.histplot(df[col], kde=True, ax=axes[i, 0], color=sns.color_palette('viridis')[0]) # Removed palette, set single color
    axes[i, 0].set_title(f'Distribution of {col}')

    # Boxplot vs Churn - Fix for FutureWarning
    sns.boxplot(x='Churn', y=col, data=df, ax=axes[i, 1], palette='magma')
    axes[i, 1].set_title(f'{col} vs Churn')

plt.tight_layout()
plt.show()

# --- 6. Correlation Matrix (Numerical Features) ---
print("\n--- 6. Correlation Matrix (Numerical Features) ---")
numerical_df = df[numerical_cols + ['Churn']].copy()
numerical_df['Churn'] = numerical_df['Churn'].map({'Yes': 1, 'No': 0})
plt.figure(figsize=(8, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

print("\n--- EDA Complete. Initial Insights Gained. ---")


# --- Data Preprocessing and Feature Engineering ---

print("\n--- Starting Data Preprocessing and Feature Engineering ---")

# Make a copy of the dataframe to work on, preserving original EDA results
df_processed = df.copy()

# Drop customerID as it's not a predictive feature
df_processed = df_processed.drop('customerID', axis=1)

# Convert 'No internet service' and 'No phone service' to 'No' for consistency
for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'MultipleLines']:
    df_processed[col] = df_processed[col].replace('No internet service', 'No')
    df_processed[col] = df_processed[col].replace('No phone service', 'No')

# Convert 'gender' to 0/1 (binary encoding)
df_processed['gender'] = df_processed['gender'].map({'Female': 0, 'Male': 1})

# Convert 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling' to 0/1 (binary encoding)
binary_cols_to_map = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies', 'MultipleLines'] # SeniorCitizen is already int64 0/1

for col in binary_cols_to_map:
    if df_processed[col].dtype == 'object': # Only map if it's an object (string 'Yes'/'No')
        df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})

# SeniorCitizen is already 0/1 integer, no mapping needed
# Ensure it's treated as a numeric column, it will be included in get_dummies if still object.
# Let's ensure it's numeric before one-hot encoding if it was somehow an object
if df_processed['SeniorCitizen'].dtype == 'object':
    df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(int)


# One-Hot Encode remaining categorical features (multi-level categories)
# Exclude 'gender' as it's already handled, and 'Churn' (target)
categorical_cols_onehot = [col for col in df_processed.select_dtypes(include='object').columns if col != 'Churn']

print(f"\nCategorical columns to One-Hot Encode: {categorical_cols_onehot}")

# Before one-hot encoding, let's check the unique values of 'InternetService' to confirm categories
print(f"Unique values in 'InternetService' before One-Hot Encoding: {df_processed['InternetService'].unique()}")

df_processed = pd.get_dummies(df_processed, columns=categorical_cols_onehot, drop_first=True, dtype=int)


# Convert target variable 'Churn' to numerical (0/1)
df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})

print("\n--- Feature Engineering ---")

# Feature 1: Total Services (count of all services a customer uses)
service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Ensure these columns are already numeric (0/1) from previous binary mapping
# If any are still object, explicitly convert them
for col in service_cols:
    if df_processed[col].dtype == 'object':
        df_processed[col] = df_processed[col].map({'Yes':1, 'No':0})

df_processed['TotalServices'] = df_processed[service_cols].sum(axis=1)
print("Created 'TotalServices' feature.")

# Feature 2: Monthly Charge per Tenure (average monthly cost)
# Avoid division by zero for new customers with tenure 0
df_processed['MonthlyChargePerTenure'] = df_processed.apply(
    lambda row: row['MonthlyCharges'] / row['tenure'] if row['tenure'] > 0 else 0, axis=1
)
print("Created 'MonthlyChargePerTenure' feature.")

# --- FIX for KeyError: 'InternetService_DSL' ---
# Instead of hardcoding 'InternetService_Fiber optic' and 'InternetService_DSL',
# let's create 'HasInternetService' based on the original 'InternetService' column
# before it gets one-hot encoded and potentially some columns dropped.

# Re-evaluate HasInternetService based on original 'InternetService' column values.
# This assumes 'df' is still available from the beginning of your script.
# If you only have df_processed after one-hot encoding, you'd need to check for
# 'InternetService_No' instead, as the other two will represent having internet.

# Let's derive it robustly using the original column, or from processed columns.
# A robust way is to check if 'InternetService_No' exists. If it does, then
# HasInternetService is the inverse of that.
# If 'InternetService_No' is dropped (due to drop_first=True), then the sum of
# 'InternetService_Fiber optic' and 'InternetService_DSL' will work if both exist.

# Check which InternetService columns were created after get_dummies:
created_internet_cols = [col for col in df_processed.columns if col.startswith('InternetService_')]
print(f"\nInternetService columns created after One-Hot Encoding: {created_internet_cols}")

# Robust way to create 'HasInternetService':
# If 'InternetService_No' is present, then HasInternetService is the inverse.
# Otherwise, it implies having some internet service if other internet columns are present.
if 'InternetService_No' in df_processed.columns:
    df_processed['HasInternetService'] = 1 - df_processed['InternetService_No']
elif 'InternetService_Fiber optic' in df_processed.columns or 'InternetService_DSL' in df_processed.columns:
    # If 'No' was dropped, then having Fiber Optic or DSL implies internet.
    # We must ensure both 'Fiber optic' and 'DSL' exist, or adjust if only one exists.
    has_fiber = df_processed['InternetService_Fiber optic'].astype(int) if 'InternetService_Fiber optic' in df_processed.columns else 0
    has_dsl = df_processed['InternetService_DSL'].astype(int) if 'InternetService_DSL' in df_processed.columns else 0
    df_processed['HasInternetService'] = (has_fiber | has_dsl).astype(int)
else:
    # This case should ideally not happen if 'InternetService' is handled by get_dummies
    df_processed['HasInternetService'] = 0 # Default to no internet if no specific columns found
print("Created 'HasInternetService' feature.")

# Feature 4: Is a Loyal Customer (based on contract type)
# Similar robust check for Contract columns
if 'Contract_Two year' in df_processed.columns and 'Contract_One year' in df_processed.columns:
    df_processed['IsLoyalCustomer'] = df_processed['Contract_Two year'].astype(int) | df_processed['Contract_One year'].astype(int)
elif 'Contract_Two year' in df_processed.columns: # If 'One year' was dropped
    df_processed['IsLoyalCustomer'] = df_processed['Contract_Two year'].astype(int)
elif 'Contract_One year' in df_processed.columns: # If 'Two year' was dropped
    df_processed['IsLoyalCustomer'] = df_processed['Contract_One year'].astype(int)
else:
    df_processed['IsLoyalCustomer'] = 0 # Default if neither exists
print("Created 'IsLoyalCustomer' feature.")


# --- Feature Scaling ---
print("\n--- Starting Feature Scaling ---")

from sklearn.preprocessing import StandardScaler

# Identify numerical columns for scaling
numerical_cols_for_scaling = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'MonthlyChargePerTenure']

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply scaling to the numerical columns
df_processed[numerical_cols_for_scaling] = scaler.fit_transform(df_processed[numerical_cols_for_scaling])

print("Numerical features scaled using StandardScaler.")

# Final check of the processed dataframe
print("\n--- Processed DataFrame Info ---")
df_processed.info()
print("\nProcessed DataFrame Head:")
print(df_processed.head())
print("\nProcessed DataFrame Tail:") # Good to check tail as well for consistency
print(df_processed.tail())

print("\n--- Data Preprocessing and Feature Engineering Complete ---")