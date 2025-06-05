import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
import io 

# Suppress specific FutureWarnings from seaborn for cleaner output in app
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# --- Function to Load and Preprocess Data ---
# Encapsulate all your data loading and preprocessing logic here
@st.cache_data # Cache the data loading and preprocessing for performance
def load_and_preprocess_data():
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except FileNotFoundError:
        st.error("Dataset 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found. Please ensure it's in the same directory as app.py.")
        st.stop() # Stop the app if data is not found

    # Handle 'TotalCharges' missing values and type conversion (from EDA)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    df_processed = df.copy()
    df_processed = df_processed.drop('customerID', axis=1)

    # Convert 'No internet service' and 'No phone service' to 'No' for consistency
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'MultipleLines']:
        df_processed[col] = df_processed[col].replace('No internet service', 'No')
        df_processed[col] = df_processed[col].replace('No phone service', 'No')

    # Convert 'gender' to 0/1
    df_processed['gender'] = df_processed['gender'].map({'Female': 0, 'Male': 1})

    # Convert 'Yes'/'No' columns to 0/1
    binary_cols_to_map = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'MultipleLines']
    for col in binary_cols_to_map:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})

    # Ensure SeniorCitizen is numeric
    if df_processed['SeniorCitizen'].dtype == 'object':
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(int)

    # One-Hot Encode remaining categorical features
    categorical_cols_onehot = [col for col in df_processed.select_dtypes(include='object').columns if col != 'Churn']
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols_onehot, drop_first=True, dtype=int)

    # Convert target variable 'Churn' to numerical (0/1)
    df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})

    # Feature Engineering
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols: # Ensure these are numeric before summing
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].map({'Yes':1, 'No':0})
    df_processed['TotalServices'] = df_processed[service_cols].sum(axis=1)

    df_processed['MonthlyChargePerTenure'] = df_processed.apply(
        lambda row: row['MonthlyCharges'] / row['tenure'] if row['tenure'] > 0 else 0, axis=1
    )

    # Robust HasInternetService creation (handles cases where 'No' might be dropped)
    if 'InternetService_No' in df_processed.columns:
        df_processed['HasInternetService'] = 1 - df_processed['InternetService_No']
    elif 'InternetService_Fiber optic' in df_processed.columns or 'InternetService_DSL' in df_processed.columns:
        has_fiber = df_processed['InternetService_Fiber optic'].astype(int) if 'InternetService_Fiber optic' in df_processed.columns else 0
        has_dsl = df_processed['InternetService_DSL'].astype(int) if 'InternetService_DSL' in df_processed.columns else 0
        df_processed['HasInternetService'] = (has_fiber | has_dsl).astype(int)
    else:
        df_processed['HasInternetService'] = 0

    # Robust IsLoyalCustomer creation
    if 'Contract_Two year' in df_processed.columns and 'Contract_One year' in df_processed.columns:
        df_processed['IsLoyalCustomer'] = df_processed['Contract_Two year'].astype(int) | df_processed['Contract_One year'].astype(int)
    elif 'Contract_Two year' in df_processed.columns:
        df_processed['IsLoyalCustomer'] = df_processed['Contract_Two year'].astype(int)
    elif 'Contract_One year' in df_processed.columns:
        df_processed['IsLoyalCustomer'] = df_processed['Contract_One year'].astype(int)
    else:
        df_processed['IsLoyalCustomer'] = 0

    # Feature Scaling
    numerical_cols_for_scaling = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'MonthlyChargePerTenure']
    scaler = StandardScaler()
    df_processed[numerical_cols_for_scaling] = scaler.fit_transform(df_processed[numerical_cols_for_scaling])

    return df_processed, df # Return both processed and original for different views

# --- Streamlit App Layout ---

st.set_page_config(
    page_title="Telco Churn Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Telco Customer Churn Analysis Dashboard")
st.markdown("---")

# Load data
df_processed, df_original = load_and_preprocess_data()

st.sidebar.header("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ["Overview & Raw Data", "Exploratory Data Analysis", "Feature Engineering Insights"]
)

# --- Page 1: Overview & Raw Data ---
if page_selection == "Overview & Raw Data":
    st.header("1. Overview & Raw Data")
    st.write("This dashboard provides an interactive analysis of customer churn in a telecommunications dataset. Use the sidebar to navigate through different sections.")

    st.subheader("Raw Data Preview")
    st.write(df_original.head())

    st.subheader("Dataset Information")
    buffer = io.StringIO() # Create a buffer to capture info() output
    df_original.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s) # Display info() output

    st.subheader("Descriptive Statistics")
    st.write(df_original.describe())

    st.subheader("Churn Distribution")
    churn_counts = df_original['Churn'].value_counts(normalize=True) * 100
    st.write(f"Churn Rate: {churn_counts.get('Yes', 0):.2f}%")
    st.write(f"Retention Rate: {churn_counts.get('No', 0):.2f}%")

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Churn', data=df_original, palette='viridis', ax=ax1)
    ax1.set_title('Distribution of Customer Churn')
    st.pyplot(fig1)

# --- Page 2: Exploratory Data Analysis (EDA) ---
elif page_selection == "Exploratory Data Analysis":
    st.header("2. Exploratory Data Analysis (EDA)")
    st.write("Explore the relationships between various customer attributes and churn.")

    st.subheader("Categorical Features vs. Churn")
    categorical_cols_for_eda = [col for col in df_original.select_dtypes(include='object').columns if col not in ['customerID', 'Churn']]
    categorical_cols_for_eda.extend(['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']) # Add binary numeric ones

    selected_cat_col = st.selectbox("Select a Categorical Feature:", categorical_cols_for_eda)

    if selected_cat_col:
        st.write(f"### {selected_cat_col} vs Churn")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.countplot(x=selected_cat_col, hue='Churn', data=df_original, palette='magma', ax=ax2)
        ax2.set_title(f'Churn Distribution by {selected_cat_col}')
        ax2.set_xlabel(selected_cat_col)
        ax2.set_ylabel('Number of Customers')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig2)

        # Display churn rate by selected category
        churn_rate_by_cat = df_original.groupby(selected_cat_col)['Churn'].value_counts(normalize=True).unstack().get('Yes', 0)
        if not isinstance(churn_rate_by_cat, pd.Series): # Handle case where get returns single value 0
            churn_rate_by_cat = pd.Series([churn_rate_by_cat], index=['N/A'])
        st.write(f"**Churn Rate by {selected_cat_col}:**")
        st.dataframe(churn_rate_by_cat.apply(lambda x: f"{x:.2%}"))

    st.subheader("Numerical Features vs. Churn")
    numerical_cols_for_eda = ['tenure', 'MonthlyCharges', 'TotalCharges']
    selected_num_col = st.selectbox("Select a Numerical Feature:", numerical_cols_for_eda)

    if selected_num_col:
        st.write(f"### {selected_num_col} Distribution and vs Churn")
        col1, col2 = st.columns(2)
        with col1:
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            sns.histplot(df_original[selected_num_col], kde=True, ax=ax3, color='skyblue')
            ax3.set_title(f'Distribution of {selected_num_col}')
            st.pyplot(fig3)
        with col2:
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            sns.boxplot(x='Churn', y=selected_num_col, data=df_original, ax=ax4, palette='plasma')
            ax4.set_title(f'{selected_num_col} vs Churn')
            st.pyplot(fig4)

    st.subheader("Correlation Matrix (Numerical Features)")
    numerical_df_corr = df_original[numerical_cols_for_eda + ['Churn']].copy()
    numerical_df_corr['Churn'] = numerical_df_corr['Churn'].map({'Yes': 1, 'No': 0})
    corr_matrix = numerical_df_corr.corr()

    fig5, ax5 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax5)
    ax5.set_title('Correlation Matrix of Numerical Features')
    st.pyplot(fig5)

# --- Page 3: Feature Engineering Insights ---
elif page_selection == "Feature Engineering Insights":
    st.header("3. Feature Engineering Insights")
    st.write("Explore the newly engineered features and their potential relationship with churn.")

    st.subheader("Engineered Features Preview")
    st.write(df_processed[['TotalServices', 'MonthlyChargePerTenure', 'HasInternetService', 'IsLoyalCustomer', 'Churn']].head())

    st.subheader("Distribution of Engineered Features")
    engineered_features = ['TotalServices', 'MonthlyChargePerTenure', 'HasInternetService', 'IsLoyalCustomer']

    # For binary engineered features, treat as categorical for plots
    binary_engineered = ['HasInternetService', 'IsLoyalCustomer']

    for feature in engineered_features:
        st.write(f"### {feature} vs Churn")
        if feature in binary_engineered:
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            sns.countplot(x=feature, hue='Churn', data=df_processed, palette='coolwarm', ax=ax6)
            ax6.set_title(f'Churn Distribution by {feature}')
            st.pyplot(fig6)
            churn_rate_by_engineered = df_processed.groupby(feature)['Churn'].value_counts(normalize=True).unstack().get(1, 0) # Assuming 1 is churned
            st.write(f"**Churn Rate by {feature}:**")
            st.dataframe(churn_rate_by_engineered.apply(lambda x: f"{x:.2%}"))
        else: # Treat as numerical
            colA, colB = st.columns(2)
            with colA:
                fig7, ax7 = plt.subplots(figsize=(7, 5))
                sns.histplot(df_processed[feature], kde=True, ax=ax7, color='lightcoral')
                ax7.set_title(f'Distribution of {feature} (Scaled)')
                st.pyplot(fig7)
            with colB:
                fig8, ax8 = plt.subplots(figsize=(7, 5))
                sns.boxplot(x='Churn', y=feature, data=df_processed, ax=ax8, palette='viridis')
                ax8.set_title(f'{feature} vs Churn (Scaled)')
                st.pyplot(fig8)

    st.subheader("Correlation Matrix (All Scaled Numerical Features)")
    # Include all numerical features, including engineered ones, from processed dataframe
    # Make sure 'Churn' is at the end or correctly included in the heatmap
    numerical_and_engineered_cols = [col for col in df_processed.columns if df_processed[col].dtype in ['float64', 'int64'] and col != 'Churn']
    corr_matrix_processed = df_processed[numerical_and_engineered_cols + ['Churn']].corr()

    fig9, ax9 = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix_processed, annot=True, cmap='coolwarm', fmt=".2f", ax=ax9)
    ax9.set_title('Correlation Matrix of All Relevant Numerical Features (Scaled)')
    st.pyplot(fig9)

# Placeholder for Model Prediction (Future Step)
# st.header("4. Churn Prediction (Coming Soon!)")
# st.write("This section will feature the trained machine learning model for churn prediction and interactive tools to test it.")