import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE 
import warnings
import io

# Suppress specific FutureWarnings from seaborn for cleaner output in app
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
# Suppress specific warnings from scikit-learn related to metrics
warnings.filterwarnings("ignore", message="The feature names should match")


# --- Function to Load and Preprocess Data (Existing Function - no changes needed here unless specified) ---
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except FileNotFoundError:
        st.error("Dataset 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found. Please ensure it's in the same directory as app.py.")
        st.stop()

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    df_processed = df.copy()
    df_processed = df_processed.drop('customerID', axis=1)

    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'MultipleLines']:
        df_processed[col] = df_processed[col].replace('No internet service', 'No')
        df_processed[col] = df_processed[col].replace('No phone service', 'No')

    df_processed['gender'] = df_processed['gender'].map({'Female': 0, 'Male': 1})

    binary_cols_to_map = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'MultipleLines']
    for col in binary_cols_to_map:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})

    if df_processed['SeniorCitizen'].dtype == 'object':
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(int)

    categorical_cols_onehot = [col for col in df_processed.select_dtypes(include='object').columns if col != 'Churn']
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols_onehot, drop_first=True, dtype=int)

    df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})

    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].map({'Yes':1, 'No':0})
    df_processed['TotalServices'] = df_processed[service_cols].sum(axis=1)

    df_processed['MonthlyChargePerTenure'] = df_processed.apply(
        lambda row: row['MonthlyCharges'] / row['tenure'] if row['tenure'] > 0 else 0, axis=1
    )

    if 'InternetService_No' in df_processed.columns:
        df_processed['HasInternetService'] = 1 - df_processed['InternetService_No']
    elif 'InternetService_Fiber optic' in df_processed.columns or 'InternetService_DSL' in df_processed.columns:
        has_fiber = df_processed['InternetService_Fiber optic'].astype(int) if 'InternetService_Fiber optic' in df_processed.columns else 0
        has_dsl = df_processed['InternetService_DSL'].astype(int) if 'InternetService_DSL' in df_processed.columns else 0
        df_processed['HasInternetService'] = (has_fiber | has_dsl).astype(int)
    else:
        df_processed['HasInternetService'] = 0

    if 'Contract_Two year' in df_processed.columns and 'Contract_One year' in df_processed.columns:
        df_processed['IsLoyalCustomer'] = df_processed['Contract_Two year'].astype(int) | df_processed['Contract_One year'].astype(int)
    elif 'Contract_Two year' in df_processed.columns:
        df_processed['IsLoyalCustomer'] = df_processed['Contract_Two year'].astype(int)
    elif 'Contract_One year' in df_processed.columns:
        df_processed['IsLoyalCustomer'] = df_processed['Contract_One year'].astype(int)
    else:
        df_processed['IsLoyalCustomer'] = 0

    numerical_cols_for_scaling = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices', 'MonthlyChargePerTenure']
    scaler = StandardScaler()
    df_processed[numerical_cols_for_scaling] = scaler.fit_transform(df_processed[numerical_cols_for_scaling])

    return df_processed, df # Return both processed and original for different views


# --- New Function: Train and Evaluate Models ---
@st.cache_resource # Cache the trained models and results for performance
def train_and_evaluate_models(df_processed_data):
    st.subheader("Training Models...")

    # Separate features (X) and target (y)
    X = df_processed_data.drop('Churn', axis=1)
    y = df_processed_data['Churn']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify to maintain churn ratio

    st.write(f"Original Churn Ratio (Train): {y_train.value_counts(normalize=True)[1]:.2f}")

    # Handle Class Imbalance using SMOTE
    st.subheader("Handling Class Imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    st.write(f"Resampled Churn Ratio (Train): {y_train_resampled.value_counts(normalize=True)[1]:.2f}")
    st.write(f"Original Training samples: {len(y_train)}, Resampled Training samples: {len(y_train_resampled)}")

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'), # Add class_weight
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'), # Add class_weight
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=(len(y_train)-y_train.sum())/y_train.sum()) # Use scale_pos_weight
    }

    results = {}
    best_model_name = None
    best_f1 = -1

    for name, model in models.items():
        st.write(f"**Training {name}...**")
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] # Probability of positive class

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, output_dict=True) # Get dict for better display

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "classification_report": clf_report
        }

        if f1 > best_f1: # We prioritize F1-score for imbalanced classification
            best_f1 = f1
            best_model_name = name

    st.success("Model Training and Evaluation Complete!")
    return results, best_model_name, X_train_resampled.columns.tolist() # Return feature names for importance


# --- Streamlit App Layout ---

st.set_page_config(
    page_title="Telco Churn Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Telco Customer Churn Analysis Dashboard")
st.markdown("---")

# Load data (only once due to st.cache_data)
df_processed, df_original = load_and_preprocess_data()

st.sidebar.header("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ["Overview & Raw Data", "Exploratory Data Analysis", "Feature Engineering Insights", "Model Training & Evaluation"] # Added new page
)

# --- Page 1: Overview & Raw Data ---
if page_selection == "Overview & Raw Data":
    st.header("1. Overview & Raw Data")
    st.write("This dashboard provides an interactive analysis of customer churn in a telecommunications dataset. Use the sidebar to navigate through different sections.")

    st.subheader("Raw Data Preview")
    st.write(df_original.head())

    st.subheader("Dataset Information")
    buffer = io.StringIO()
    df_original.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

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

        churn_rate_by_cat = df_original.groupby(selected_cat_col)['Churn'].value_counts(normalize=True).unstack().get('Yes', 0)
        if not isinstance(churn_rate_by_cat, pd.Series):
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

    binary_engineered = ['HasInternetService', 'IsLoyalCustomer']

    for feature in engineered_features:
        st.write(f"### {feature} vs Churn")
        if feature in binary_engineered:
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            sns.countplot(x=feature, hue='Churn', data=df_processed, palette='coolwarm', ax=ax6)
            ax6.set_title(f'Churn Distribution by {feature}')
            st.pyplot(fig6)
            churn_rate_by_engineered = df_processed.groupby(feature)['Churn'].value_counts(normalize=True).unstack().get(1, 0)
            st.write(f"**Churn Rate by {feature}:**")
            st.dataframe(churn_rate_by_engineered.apply(lambda x: f"{x:.2%}"))
        else:
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
    numerical_and_engineered_cols = [col for col in df_processed.columns if df_processed[col].dtype in ['float64', 'int64'] and col != 'Churn']
    corr_matrix_processed = df_processed[numerical_and_engineered_cols + ['Churn']].corr()

    fig9, ax9 = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix_processed, annot=True, cmap='coolwarm', fmt=".2f", ax=ax9)
    ax9.set_title('Correlation Matrix of All Relevant Numerical Features (Scaled)')
    st.pyplot(fig9)

# --- NEW Page: Model Training & Evaluation ---
elif page_selection == "Model Training & Evaluation":
    st.header("4. Model Training & Evaluation")
    st.write("Here, we train and evaluate various machine learning models to predict customer churn.")

    if st.button("Train and Evaluate Models"):
        results, best_model_name, feature_names = train_and_evaluate_models(df_processed)

        st.subheader("Model Performance Comparison")
        metrics_df = pd.DataFrame.from_dict({
            name: {
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "F1-Score": res["f1_score"],
                "ROC-AUC": res["roc_auc"]
            } for name, res in results.items()
        }, orient='index')
        st.dataframe(metrics_df.style.highlight_max(axis=0, props='background-color: yellow;')) # Highlight best score

        st.markdown(f"**Best Model (based on F1-Score): {best_model_name}**")
        st.write("F1-Score is prioritized due to class imbalance.")

        for name, res in results.items():
            st.subheader(f"Detailed Results for {name}")
            st.write("#### Classification Report:")
            st.json(res["classification_report"]) # Display as JSON for readability
            # st.text(classification_report(y_test_from_res, y_pred_from_res)) # Need to get y_test etc from results, or just display the dict

            st.write("#### Confusion Matrix:")
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(res["confusion_matrix"], annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=['Predicted No Churn', 'Predicted Churn'],
                        yticklabels=['Actual No Churn', 'Actual Churn'])
            ax_cm.set_ylabel('Actual Label')
            ax_cm.set_xlabel('Predicted Label')
            ax_cm.set_title(f'Confusion Matrix for {name}')
            st.pyplot(fig_cm)

        # Feature Importance for tree-based models
        st.subheader("Feature Importance (from Best Tree-Based Model)")
        if best_model_name in ["Random Forest", "XGBoost"]:
            best_model = results[best_model_name]['model']
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'): # For linear models like Logistic Regression (absolute coefficients)
                importances = np.abs(best_model.coef_[0])
            else:
                importances = None # Fallback

            if importances is not None:
                feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

                fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), ax=ax_fi, palette='viridis')
                ax_fi.set_title(f'Top 15 Feature Importances for {best_model_name}')
                ax_fi.set_xlabel('Importance')
                ax_fi.set_ylabel('Feature')
                st.pyplot(fig_fi)
            else:
                st.write("Feature importance is not directly available for the best model type.")
        else:
            st.write("Feature importance visualization is currently implemented for Random Forest and XGBoost models.")


# Placeholder for Model Prediction (Future Step - could be integrated into Model Training page or a separate one)
# st.header("5. Churn Prediction (Coming Soon!)")
# st.write("This section will feature the trained machine learning model for churn prediction and interactive tools to test it.")