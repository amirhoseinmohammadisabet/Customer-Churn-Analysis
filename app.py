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
import joblib # NEW: Import joblib for saving/loading models

# Suppress specific FutureWarnings from seaborn for cleaner output in app
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
# Suppress specific warnings from scikit-learn related to metrics
warnings.filterwarnings("ignore", message="The feature names should match")


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

    # Store initial column order before one-hot encoding for consistent prediction input later
    original_cols_before_ohe = df_processed.columns.tolist()

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
    # Fit scaler here and save it too, as it's needed for prediction
    df_processed[numerical_cols_for_scaling] = scaler.fit_transform(df_processed[numerical_cols_for_scaling])

    # Store the fitted scaler and the columns used in training for consistency in prediction
    st.session_state['scaler'] = scaler # Use session_state to store scaler
    st.session_state['numerical_cols_for_scaling'] = numerical_cols_for_scaling
    st.session_state['ohe_columns'] = categorical_cols_onehot # Store list of columns that were OHE
    st.session_state['ohe_categories'] = {
        col: df[col].astype('category').cat.categories.tolist()
        for col in categorical_cols_onehot
    } # Store categories for OHE for prediction

    return df_processed, df # Return both processed and original for different views


# --- Function to Train and Evaluate Models (Modified to save best model) ---
@st.cache_resource # Cache the trained models and results for performance
def train_and_evaluate_models(df_processed_data):
    st.subheader("Training Models...")

    X = df_processed_data.drop('Churn', axis=1)
    y = df_processed_data['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.write(f"Original Churn Ratio (Train): {y_train.value_counts(normalize=True)[1]:.2f}")

    st.subheader("Handling Class Imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    st.write(f"Resampled Churn Ratio (Train): {y_train_resampled.value_counts(normalize=True)[1]:.2f}")
    st.write(f"Original Training samples: {len(y_train)}, Resampled Training samples: {len(y_train_resampled)}")

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=(len(y_train)-y_train.sum())/y_train.sum())
    }

    results = {}
    best_model_name = None
    best_f1 = -1
    best_model_obj = None # NEW: To store the best model object

    for name, model in models.items():
        st.write(f"**Training {name}...**")
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "model": model, # Store the model object itself
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "classification_report": clf_report
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model # Store the best model object

    # NEW: Save the best model
    model_filename = 'best_churn_model.pkl'
    joblib.dump(best_model_obj, model_filename)
    st.success(f"Best model ({best_model_name}) saved as '{model_filename}'")
    st.session_state['best_model_loaded'] = True # Set a flag in session state
    st.session_state['best_model_name'] = best_model_name # Store best model name

    st.success("Model Training and Evaluation Complete!")
    return results, best_model_name, X_train_resampled.columns.tolist(), best_model_obj # NEW: Return best_model_obj


# --- NEW FUNCTION: Preprocess single input for prediction ---
def preprocess_input(input_data, original_df_for_ohe, scaler, numerical_cols, ohe_columns, ohe_categories):
    # Convert input_data dictionary to a pandas DataFrame
    df_input = pd.DataFrame([input_data])

    # Convert 'No internet service' and 'No phone service' to 'No' (as done in training)
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'MultipleLines']:
        if col in df_input.columns and df_input[col].iloc[0] == 'No internet service':
            df_input[col] = 'No'
        if col in df_input.columns and df_input[col].iloc[0] == 'No phone service':
            df_input[col] = 'No'

    # Binary Encoding (gender, Partner, Dependents, PhoneService, PaperlessBilling, Services)
    df_input['gender'] = df_input['gender'].map({'Female': 0, 'Male': 1})
    binary_map_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'MultipleLines', 'SeniorCitizen'] # SeniorCitizen is already 0/1 int
    for col in binary_map_cols:
        if col in df_input.columns and df_input[col].dtype == 'object':
            df_input[col] = df_input[col].map({'Yes': 1, 'No': 0})
        # If SeniorCitizen is read as object (from form), convert it to int
        if col == 'SeniorCitizen' and df_input[col].dtype == 'object':
            df_input[col] = df_input[col].astype(int)


    # One-Hot Encoding for remaining categorical features (Contract, InternetService, PaymentMethod)
    # Ensure all possible categories from training are considered, even if not in current input
    for col in ohe_columns:
        # Create dummy variables for the specific column
        dummies = pd.get_dummies(df_input[col], prefix=col, dtype=int)
        # Add any missing columns from original categories (due to drop_first=True or missing category in input)
        for cat in ohe_categories[col]:
            if f'{col}_{cat}' not in dummies.columns and f'{col}_{cat}' not in df_input.columns and cat != ohe_categories[col][0]: # Exclude the dropped_first category
                dummies[f'{col}_{cat}'] = 0
        df_input = pd.concat([df_input.drop(columns=[col]), dummies], axis=1)

    # Feature Engineering (must be consistent with training)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_input['TotalServices'] = df_input[[col for col in service_cols if col in df_input.columns]].sum(axis=1)

    df_input['MonthlyChargePerTenure'] = df_input.apply(
        lambda row: row['MonthlyCharges'] / row['tenure'] if row['tenure'] > 0 else 0, axis=1
    )

    # Robust HasInternetService creation (handles cases where 'No' might be dropped in OHE)
    if 'InternetService_No' in df_input.columns: # Check if 'No' category was part of OHE
        df_input['HasInternetService'] = 1 - df_input['InternetService_No']
    elif 'InternetService_Fiber optic' in df_input.columns or 'InternetService_DSL' in df_input.columns:
        has_fiber = df_input['InternetService_Fiber optic'].astype(int) if 'InternetService_Fiber optic' in df_input.columns else 0
        has_dsl = df_input['InternetService_DSL'].astype(int) if 'InternetService_DSL' in df_input.columns else 0
        df_input['HasInternetService'] = (has_fiber | has_dsl).astype(int)
    else: # Default if no internet service columns are present (e.g., if only 'No' existed in training and was dropped)
        df_input['HasInternetService'] = 0

    # Robust IsLoyalCustomer creation
    if 'Contract_Two year' in df_input.columns and 'Contract_One year' in df_input.columns:
        df_input['IsLoyalCustomer'] = df_input['Contract_Two year'].astype(int) | df_input['Contract_One year'].astype(int)
    elif 'Contract_Two year' in df_input.columns:
        df_input['IsLoyalCustomer'] = df_input['Contract_Two year'].astype(int)
    elif 'Contract_One year' in df_input.columns:
        df_input['IsLoyalCustomer'] = df_input['Contract_One year'].astype(int)
    else:
        df_input['IsLoyalCustomer'] = 0

    # Align columns with training data - CRUCIAL STEP
    # Ensure df_input has all columns that X_train had, in the same order
    # Fill missing columns with 0
    # Drop extra columns not in X_train (shouldn't happen if OHE is consistent)
    X_train_cols = st.session_state['X_train_cols'] # Retrieve column order from training
    final_input = pd.DataFrame(0, index=[0], columns=X_train_cols)
    for col in final_input.columns:
        if col in df_input.columns:
            final_input[col] = df_input[col].iloc[0]

    # Scale numerical features
    final_input[numerical_cols] = scaler.transform(final_input[numerical_cols])

    return final_input

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

# Initialize session state for model loaded flag
if 'best_model_loaded' not in st.session_state:
    st.session_state['best_model_loaded'] = False
if 'X_train_cols' not in st.session_state:
    st.session_state['X_train_cols'] = None # To store column order from training


st.sidebar.header("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ["Overview & Raw Data", "Exploratory Data Analysis", "Feature Engineering Insights", "Model Training & Evaluation", "Make a Churn Prediction"] # Added new page
)

# --- Page 1: Overview & Raw Data (Existing) ---
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

# --- Page 2: Exploratory Data Analysis (EDA) (Existing) ---
elif page_selection == "Exploratory Data Analysis":
    st.header("2. Exploratory Data Analysis (EDA)")
    st.write("Explore the relationships between various customer attributes and churn.")

    st.subheader("Categorical Features vs. Churn")
    categorical_cols_for_eda = [col for col in df_original.select_dtypes(include='object').columns if col not in ['customerID', 'Churn']]
    categorical_cols_for_eda.extend(['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling'])

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

# --- Page 3: Feature Engineering Insights (Existing) ---
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

# --- Page 4: Model Training & Evaluation (Modified to store X_train columns) ---
elif page_selection == "Model Training & Evaluation":
    st.header("4. Model Training & Evaluation")
    st.write("Here, we train and evaluate various machine learning models to predict customer churn.")

    if st.button("Train and Evaluate Models"):
        results, best_model_name, feature_names, best_model_obj = train_and_evaluate_models(df_processed)
        st.session_state['X_train_cols'] = feature_names # Store feature names for prediction input consistency

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
        st.dataframe(metrics_df.style.highlight_max(axis=0, props='background-color: yellow;'))

        st.markdown(f"**Best Model (based on F1-Score): {best_model_name}**")
        st.write("F1-Score is prioritized due to class imbalance.")

        for name, res in results.items():
            st.subheader(f"Detailed Results for {name}")
            st.write("#### Classification Report:")
            st.json(res["classification_report"])

            st.write("#### Confusion Matrix:")
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(res["confusion_matrix"], annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=['Predicted No Churn', 'Predicted Churn'],
                        yticklabels=['Actual No Churn', 'Actual Churn'])
            ax_cm.set_ylabel('Actual Label')
            ax_cm.set_xlabel('Predicted Label')
            ax_cm.set_title(f'Confusion Matrix for {name}')
            st.pyplot(fig_cm)

        st.subheader("Feature Importance (from Best Tree-Based Model)")
        if best_model_name in ["Random Forest", "XGBoost"]:
            best_model = results[best_model_name]['model']
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
            elif hasattr(best_model, 'coef_'):
                importances = np.abs(best_model.coef_[0])
            else:
                importances = None

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


# --- NEW PAGE: Make a Churn Prediction ---
elif page_selection == "Make a Churn Prediction":
    st.header("5. Make a Churn Prediction")
    st.write("Enter customer details below to get a churn prediction.")

    # Check if a model has been trained and saved
    if not st.session_state.get('best_model_loaded', False):
        st.warning("Please train the models first on the 'Model Training & Evaluation' page to enable predictions.")
    else:
        # Load the best model (cached)
        try:
            best_model = joblib.load('best_churn_model.pkl')
            st.write(f"Using **{st.session_state['best_model_name']}** for prediction.")
        except FileNotFoundError:
            st.error("Best model file not found. Please train models on the 'Model Training & Evaluation' page.")
            st.stop()

        # Define input fields based on original features
        st.subheader("Customer Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", df_original['gender'].unique())
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            partner = st.selectbox("Partner", df_original['Partner'].unique())
            dependents = st.selectbox("Dependents", df_original['Dependents'].unique())
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", df_original['PhoneService'].unique())
            multiple_lines = st.selectbox("Multiple Lines", df_original['MultipleLines'].unique())

        with col2:
            internet_service = st.selectbox("Internet Service", df_original['InternetService'].unique())
            online_security = st.selectbox("Online Security", df_original['OnlineSecurity'].unique())
            online_backup = st.selectbox("Online Backup", df_original['OnlineBackup'].unique())
            device_protection = st.selectbox("Device Protection", df_original['DeviceProtection'].unique())
            tech_support = st.selectbox("Tech Support", df_original['TechSupport'].unique())
            streaming_tv = st.selectbox("Streaming TV", df_original['StreamingTV'].unique())
            streaming_movies = st.selectbox("Streaming Movies", df_original['StreamingMovies'].unique())

        with col3:
            contract = st.selectbox("Contract", df_original['Contract'].unique())
            paperless_billing = st.selectbox("Paperless Billing", df_original['PaperlessBilling'].unique())
            payment_method = st.selectbox("Payment Method", df_original['PaymentMethod'].unique())
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=120.0, value=50.0, step=0.1)
            # TotalCharges is derived, so we don't ask for it directly. For simplicity, we can assume
            # TotalCharges = MonthlyCharges * Tenure for prediction, or keep it 0 if tenure is 0.
            # Let's derive it to be consistent with preprocessing logic.
            total_charges_input = monthly_charges * tenure if tenure > 0 else 0
            st.text(f"Derived Total Charges: {total_charges_input:.2f}")


        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges_input # Use the derived total charges
        }

        if st.button("Predict Churn"):
            # Ensure scaler, numerical_cols, ohe_columns, ohe_categories, X_train_cols are available from session_state
            if (st.session_state.get('scaler') is None or
                st.session_state.get('numerical_cols_for_scaling') is None or
                st.session_state.get('ohe_columns') is None or
                st.session_state.get('ohe_categories') is None or
                st.session_state.get('X_train_cols') is None):
                st.error("Preprocessing components not found. Please re-run 'Model Training & Evaluation' once to initialize.")
            else:
                try:
                    processed_input = preprocess_input(
                        input_data,
                        df_original, # Pass df_original for OHE categories reference
                        st.session_state['scaler'],
                        st.session_state['numerical_cols_for_scaling'],
                        st.session_state['ohe_columns'],
                        st.session_state['ohe_categories']
                    )

                    prediction = best_model.predict(processed_input)
                    prediction_proba = best_model.predict_proba(processed_input)[:, 1] # Probability of churn

                    st.subheader("Prediction Result:")
                    if prediction[0] == 1:
                        st.error(f"**This customer is likely to CHURN!** (Probability: {prediction_proba[0]:.2%})")
                        st.image('https://media.giphy.com/media/l0HeoX6Y1xJzP2bPa/giphy.gif', width=100) # Sad gif
                        st.warning("Consider proactive retention strategies for this customer.")
                    else:
                        st.success(f"**This customer is likely to stay.** (Probability: {prediction_proba[0]:.2%})")
                        st.image('https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMzQ2YjgyOGQ5YmM4ZDRlZDRlMmMyZGE2MzgxM2I2ZTI3YmJmZGMwZSZlcD12MV9pbnRlcm5hbF9naWZzX2lkPTcwNjg3Mjg1/QyN90o84594sM/giphy.gif', width=100) # Happy gif
                        st.info("Continue to monitor and ensure satisfaction.")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.warning("Please ensure all inputs are valid and try training the models again if this persists.")