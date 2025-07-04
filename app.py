# --- Core and Data Handling Imports ---
# --- Core and Third-Party Libraries ---
import base64
import io
import streamlit as st
import numpy as np
import pandas as pd
import dateparser
from typing import Callable, Dict, List
from word2number import w2n

# --- Visualization Libraries ---
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# --- Scikit-Learn Preprocessing and Pipeline ---
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# --- Scikit-Learn Metrics ---
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error, r2_score)

# --- Scikit-Learn Models ---
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression,
                                  LogisticRegression, Ridge)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# --- Other Machine Learning Libraries ---
import lightgbm as lgb
import xgboost as xgb
# Metrics
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
# ==================================================================================================
# 2. APPLICATION CONFIGURATION AND CONSTANTS
# ==================================================================================================

APP_TITLE = "CSV Data Cleaner"
APP_ICON = "-🧹"

# Define the structure of our application pages
PAGES = {
    "🏠 Home: Upload & Inspect": "render_home_page",
    "📊 Data Profiling & Overview": "render_profiling_page",
    "❓ Missing Value Manager": "render_missing_values_page",
    "🏛️ Column Operations": "render_column_management_page",
    "📑 Row & Duplicate Manager": "render_duplicate_handling_page",
    "📈 Outlier Detection & Handling": "render_outlier_page",
    "🔬 Data Transformation": "render_transformation_page",
    "🤖 ML Modeler": "render_ml_modeler_page",
    "📜 Action History": "render_history_page",
    "📥 Download & Export": "render_download_page",
}

# ==================================================================================================
# 3. ROBUST STATE MANAGEMENT
# ==================================================================================================

def initialize_session_state():
    """
    Initializes all necessary keys in st.session_state on first run.
    This centralized function prevents state-related errors.
    """
    # --- Core Data Storage ---
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None

    # --- Control and UI Flags ---
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'active_page' not in st.session_state:
        st.session_state.active_page = list(PAGES.keys())[0]
    if 'file_name' not in st.session_state:
        st.session_state.file_name = None

    # --- Action History Log ---
    if 'history' not in st.session_state:
        st.session_state.history = []

def reset_app_state():
    """
    Resets the application to its initial state, ready for a new file.
    This function is carefully designed to reset all relevant state variables.
    """
    st.session_state.df = None
    st.session_state.df_original = None
    st.session_state.file_uploaded = False
    st.session_state.file_uploader_key += 1
    st.session_state.active_page = list(PAGES.keys())[0]
    st.session_state.history = []
    st.session_state.file_name = None
    st.toast("Application has been reset. Please upload a new file.", icon="🔄")
    st.rerun()

def log_action(description: str, code_snippet: str = None):
    """
    Logs a user action to the history.

    Args:
        description (str): A user-friendly description of the action.
        code_snippet (str, optional): The equivalent pandas code. Defaults to None.
    """
    st.session_state.history.append({"description": description, "code": code_snippet})
    st.toast(f"Action logged: {description}", icon="✅")

# ==================================================================================================
# 4. HELPER & UTILITY FUNCTIONS
# ==================================================================================================

def get_dataframe_info(df: pd.DataFrame) -> str:
    """Captures df.info() output into a string."""
    buffer = io.StringIO()
    df.info(buf=buffer, verbose=True)
    return buffer.getvalue()

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Returns a list of numeric column names."""
    if df is None:
        return []
    return df.select_dtypes(include=np.number).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Returns a list of categorical/object column names."""
    if df is None:
        return []
    return df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Returns a list of datetime column names."""
    if df is None:
        return []
    return df.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()

# ==================================================================================================
# 5. UI PAGE RENDERING FUNCTIONS
#
# Each function is responsible for a single page in the application. This modular approach
# keeps the code clean and manageable.
# ==================================================================================================

# ---------------------------------- 5.1 HOME / UPLOAD PAGE --------------------------------------
def render_home_page():
    """Renders the initial landing page for file upload and instructions."""
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("""
    Welcome to your one-stop solution for data cleaning! This powerful and robust tool,
    built with Streamlit, allows you to systematically clean, preprocess, and prepare your
    CSV data for analysis or machine learning.

    **How to use this application:**
    1.  **Upload Your Data**: Use the file uploader below to load your CSV file. Try the advanced options if you encounter encoding errors.
    2.  **Navigate & Clean**: Use the sidebar on the left to navigate through different cleaning modules.
    3.  **Track Your Progress**: The 'Action History' page keeps a log of all transformations you apply.
    4.  **Download**: Once satisfied, proceed to the 'Download & Export' page to get your cleaned data.
    """)

    st.subheader("1. Upload Your CSV File")

    with st.expander("Upload Options", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            key=f"file_uploader_{st.session_state.file_uploader_key}",
            help="Upload a CSV file to begin the cleaning process."
        )
        st.markdown("---")
        st.subheader("Advanced Options")
        col1, col2 = st.columns(2)
        with col1:
            separator = st.text_input("Column Separator (e.g., ',' or ';')", value=",")
        with col2:
            encoding = st.selectbox("File Encoding", ["utf-8", "latin1", "iso-8859-1", "cp1252"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
            st.session_state.df_original = df.copy()
            st.session_state.df = df.copy()
            st.session_state.file_uploaded = True
            st.session_state.file_name = uploaded_file.name
            log_action(f"File '{uploaded_file.name}' loaded successfully. Shape: {df.shape}")
            st.success("File uploaded and processed successfully!")
            st.rerun() # Rerun to move to the next logical view
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.warning("Please check the separator, encoding, or file integrity.")
            st.session_state.file_uploaded = False

    if st.session_state.file_uploaded:
        st.subheader("2. Quick Inspection")
        st.info(f"File **'{st.session_state.file_name}'** is loaded. Shape: **{st.session_state.df.shape}**. Use the sidebar to start cleaning.")
        st.dataframe(st.session_state.df.head())

# ---------------------------------- 5.2 DATA PROFILING PAGE -------------------------------------
def render_profiling_page():
    """Renders the data profiling page with detailed statistics and info."""
    st.header("📊 Data Profiling & Overview")
    st.markdown("Get a deep understanding of your dataset's structure, types, and statistics.")

    if st.session_state.df is None:
        st.warning("Please upload a file first on the Home page.")
        return

    df = st.session_state.df

    # Create tabs for different profiling views
    tab1, tab2, tab3, tab4 = st.tabs(["DataFrame Info", "Statistical Summary", "Value Counts", "Column Correlations"])

    with tab1:
        st.subheader("DataFrame Structure and Memory")
        info_str = get_dataframe_info(df)
        st.text(info_str)

    with tab2:
        st.subheader("Descriptive Statistics")
        st.markdown("Summary statistics for all numeric columns in your dataset.")
        # Defensive check for numeric columns
        if not get_numeric_columns(df):
            st.info("No numeric columns found in the dataset to describe.")
        else:
            st.dataframe(df.describe(include=np.number))

        st.markdown("Summary statistics for all non-numeric columns.")
        # Defensive check for categorical columns
        if not get_categorical_columns(df):
            st.info("No categorical/object columns found to describe.")
        else:
            st.dataframe(df.describe(include=['object', 'category']))

    with tab3:
        st.subheader("Categorical Column Value Counts")
        categorical_cols = get_categorical_columns(df)
        if not categorical_cols:
            st.info("No categorical columns found to analyze value counts.")
        else:
            selected_col = st.selectbox(
                "Select a column to view its value distribution:",
                options=categorical_cols,
                help="Choose a column to see the frequency of each unique value."
            )
            if selected_col:
                value_counts_df = df[selected_col].value_counts().reset_index()
                value_counts_df.columns = [selected_col, 'Count']
                st.dataframe(value_counts_df)
                if st.checkbox(f"Show bar chart for '{selected_col}'?"):
                    fig = px.bar(value_counts_df, x=selected_col, y='Count', title=f"Value Counts for {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Numeric Column Correlation Analysis")
        numeric_cols = get_numeric_columns(df)
        if len(numeric_cols) < 2:
            st.info("You need at least two numeric columns to compute a correlation matrix.")
        else:
            st.markdown("A heatmap showing the Pearson correlation between numeric variables. Values close to 1 or -1 indicate a strong linear relationship.")
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)


# ---------------------------------- 5.3 MISSING VALUES PAGE -------------------------------------
def render_missing_values_page():
    """Renders the page for handling missing values (NaNs)."""
    st.header("❓ Missing Value Manager")
    st.markdown("Analyze, visualize, and handle missing data in your dataset.")

    if st.session_state.df is None:
        st.warning("Please upload a file first.")
        return

    df = st.session_state.df
    missing_data = df.isnull().sum()
    missing_data_percent = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage (%)': missing_data_percent})
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)

    if missing_df.empty:
        st.success("🎉 Excellent! No missing values found in your dataset.")
        return

    st.subheader("1. Missing Value Analysis")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(missing_df)
    with col2:
        st.markdown("#### Heatmap of Missing Values")
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("2. Handle Missing Values")

    with st.expander("🚮 Option A: Drop Missing Values"):
        st.markdown("Permanently remove rows or columns containing missing values.")
        drop_choice = st.radio("Select drop method:", ["Drop rows with any NaNs", "Drop columns with any NaNs", "Drop columns based on a threshold"])
        
        if drop_choice == "Drop columns based on a threshold":
            threshold = st.slider("Percentage of missing values threshold", 0, 100, 50)
            if st.button("Drop Columns by Threshold"):
                cols_to_drop = missing_df[missing_df['Percentage (%)'] > threshold].index.tolist()
                if not cols_to_drop:
                    st.warning("No columns met the threshold to be dropped.")
                else:
                    st.session_state.df.drop(columns=cols_to_drop, inplace=True)
                    log_action(f"Dropped columns with >{threshold}% missing values: {', '.join(cols_to_drop)}", f"df.drop(columns={cols_to_drop}, inplace=True)")
                    st.rerun()

        elif st.button("Apply Drop Operation"):
            if drop_choice == "Drop rows with any NaNs":
                st.session_state.df.dropna(axis=0, inplace=True)
                log_action("Dropped rows with any missing values.", "df.dropna(axis=0, inplace=True)")
            else: # Drop columns with any NaNs
                st.session_state.df.dropna(axis=1, inplace=True)
                log_action("Dropped columns with any missing values.", "df.dropna(axis=1, inplace=True)")
            st.rerun()

    with st.expander("✍️ Option B: Impute (Fill) Missing Values"):
        st.markdown("Replace missing values with a calculated or constant value.")
        impute_cols = missing_df.index.tolist()
        selected_col_impute = st.selectbox("Select column to impute:", impute_cols)
        
        if selected_col_impute:
            col_type = df[selected_col_impute].dtype
            impute_methods = ['Mode', 'Custom Value']
            if pd.api.types.is_numeric_dtype(col_type):
                impute_methods = ['Mean', 'Median', 'Mode', 'Interpolate', 'Custom Value']
            
            imputation_method = st.selectbox("Select imputation method:", impute_methods)
            custom_value = None
            if imputation_method == 'Custom Value':
                custom_value = st.text_input("Enter custom value:")

            if st.button("Apply Imputation"):
                try:
                    fill_value = None
                    code_snippet = ""
                    if imputation_method == 'Mean':
                        fill_value = df[selected_col_impute].mean()
                        code_snippet = f"df['{selected_col_impute}'].fillna(df['{selected_col_impute}'].mean(), inplace=True)"
                    elif imputation_method == 'Median':
                        fill_value = df[selected_col_impute].median()
                        code_snippet = f"df['{selected_col_impute}'].fillna(df['{selected_col_impute}'].median(), inplace=True)"
                    elif imputation_method == 'Mode':
                        fill_value = df[selected_col_impute].mode()[0]
                        code_snippet = f"df['{selected_col_impute}'].fillna(df['{selected_col_impute}'].mode()[0], inplace=True)"
                    elif imputation_method == 'Custom Value':
                        fill_value = custom_value
                        code_snippet = f"df['{selected_col_impute}'].fillna('{custom_value}', inplace=True)"
                    elif imputation_method == 'Interpolate':
                        st.session_state.df[selected_col_impute].interpolate(method='linear', inplace=True)
                        log_action(f"Interpolated missing values in '{selected_col_impute}'.", f"df['{selected_col_impute}'].interpolate(method='linear', inplace=True)")
                        st.rerun()
                        
                    if fill_value is not None:
                        st.session_state.df[selected_col_impute].fillna(fill_value, inplace=True)
                        log_action(f"Imputed '{selected_col_impute}' with {imputation_method}: '{fill_value}'.", code_snippet)
                        st.rerun()
                except Exception as e:
                    st.error(f"Imputation failed: {e}. Check if method is compatible with column type.")


# ---------------------------------- 5.4 COLUMN MANAGEMENT PAGE ----------------------------------
def render_column_management_page():
    """Renders page for date conversion, splitting, dropping, renaming, and type-casting columns."""
    st.header("🏛️ Column Operations")
    st.markdown("Perform column-level operations like converting dates, splitting, dropping, renaming, or changing types.")
    
    if st.session_state.df is None:
        st.warning("Please upload a file first.")
        return

    df = st.session_state.df
    all_cols = df.columns.tolist()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📅 Convert to Datetime",
        "📊 Analyze Column Types", 
        "✂️ Split Column", 
        "🗑️ Drop Columns", 
        "✏️ Rename Column", 
        "🔄 Change Column Type"
    ])

    # =========================================================================
    # --- UPGRADED FEATURE: Intelligent Date Conversion with Two Engines ---
    # =========================================================================
    with tab1:
        st.subheader("Intelligent Date Conversion")
        st.markdown("Convert columns with mixed date formats into a standardized datetime format.")

        candidate_cols = get_categorical_columns(df)
        if not candidate_cols:
            st.info("No text-based (object) columns found to convert.")
        else:
            selected_col = st.selectbox(
                "1. Select column to convert to datetime:",
                options=candidate_cols,
                key="date_convert_select"
            )

            if selected_col:
                st.markdown("#### 2. Choose Your Parsing Engine")
                
                # --- Let the user choose the engine ---
                parser_engine = st.radio(
                    "Select a parser:",
                    ["**Pandas (Fast)** - Good for standard formats like YYYY-MM-DD.", 
                     "**Dateparser (Flexible & Robust)** - Best for messy, mixed formats like 'March 3, 2021'."],
                    index=1 # Default to the more powerful option
                )

                dayfirst_param = st.checkbox(
                    "Assume Day is First (for formats like DD/MM/YYYY)",
                    value=False,
                    help="Crucial for both engines to interpret ambiguous dates like '01/04/2021'."
                )

                st.markdown("#### 3. Preview Results")
                try:
                    # Helper function to apply dateparser safely
                    def parse_with_dateparser(date_string):
                        if date_string is None: return pd.NaT
                        # The settings dictionary directly controls dateparser's behavior
                        parsed_date = dateparser.parse(str(date_string), settings={'DAY_FIRST': dayfirst_param})
                        return parsed_date if parsed_date is not None else pd.NaT

                    # --- Logic to generate the preview based on the selected engine ---
                    if "Pandas" in parser_engine:
                        preview_series = pd.to_datetime(df[selected_col].astype(str).str.strip(), dayfirst=dayfirst_param, errors='coerce')
                    else: # Dateparser engine
                        preview_series = df[selected_col].apply(parse_with_dateparser)

                    preview_df = pd.DataFrame({
                        f"Original Text in '{selected_col}'": df[selected_col].head(20),
                        "Parsed Datetime (YYYY-MM-DD)": preview_series.head(20)
                    })
                    st.dataframe(preview_df.dropna(subset=[f"Original Text in '{selected_col}'"]), use_container_width=True)
                    
                    failed_parses = preview_series.isna().sum() - df[selected_col].isna().sum()
                    if failed_parses > 0:
                        st.warning(f"Warning: {failed_parses} values could not be parsed and were converted to NaT (Not a Time).")

                except Exception as e:
                    st.error(f"An error occurred during preview generation: {e}")

                if st.button("✅ Apply Datetime Conversion", type="primary"):
                    # --- Logic to apply the conversion based on the selected engine ---
                    if "Pandas" in parser_engine:
                        st.session_state.df[selected_col] = pd.to_datetime(df[selected_col].astype(str).str.strip(), dayfirst=dayfirst_param, errors='coerce')
                    else: # Dateparser engine
                        # Define the helper function again for the apply action
                        def parse_with_dateparser_apply(date_string):
                            if date_string is None: return pd.NaT
                            parsed_date = dateparser.parse(str(date_string), settings={'DAY_FIRST': dayfirst_param})
                            return parsed_date if parsed_date is not None else pd.NaT
                        st.session_state.df[selected_col] = df[selected_col].apply(parse_with_dateparser_apply)
                    
                    log_action(f"Converted '{selected_col}' to datetime using {parser_engine.split(' ')[0]} engine.")
                    st.success(f"Successfully converted '{selected_col}' to a standardized datetime format!")
                    st.rerun()

    # =========================================================================
    # --- All Other Column Operations ---
    # =========================================================================
    with tab2:
        st.subheader("Analyze Column Data Types")
        numeric_cols = get_numeric_columns(df)
        categorical_cols = get_categorical_columns(df)
        datetime_cols = get_datetime_columns(df)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔢 Numeric Columns"); st.dataframe(pd.DataFrame(numeric_cols, columns=["Column Name"]), use_container_width=True)
            st.markdown("---")
            st.markdown("#### 📅 Datetime Columns"); st.dataframe(pd.DataFrame(datetime_cols, columns=["Column Name"]), use_container_width=True)
        with col2:
            st.markdown("#### 🅰️ Categorical / Object Columns"); st.dataframe(pd.DataFrame(categorical_cols, columns=["Column Name"]), use_container_width=True)

    with tab3:
        st.subheader("Split a Column into Categorical and Numerical Parts")
        candidate_cols_split = get_categorical_columns(df)
        if not candidate_cols_split:
            st.info("No suitable (text-based) columns found to split.")
        else:
            with st.form("split_column_form"):
                source_col = st.selectbox("Select column to split:", options=candidate_cols_split)
                col1, col2 = st.columns(2)
                with col1: new_cat_col_name = st.text_input("New Text Column Name", value=f"{source_col}_cat")
                with col2: new_num_col_name = st.text_input("New Number Column Name", value=f"{source_col}_num")
                drop_original = st.checkbox("Drop original column?", value=True)
                submitted = st.form_submit_button("Apply Split")
                if submitted:
                    st.session_state.df[new_cat_col_name] = df[source_col].str.extract(r'([^\d]*)', expand=False).str.strip()
                    numeric_part = df[source_col].str.extract(r'(\d+)', expand=False)
                    st.session_state.df[new_num_col_name] = pd.to_numeric(numeric_part, errors='coerce')
                    log_desc = f"Split '{source_col}' into '{new_cat_col_name}' and '{new_num_col_name}'."
                    if drop_original:
                        st.session_state.df.drop(columns=[source_col], inplace=True)
                        log_desc += f" Original column dropped."
                    log_action(log_desc)
                    st.rerun()

    with tab4:
        st.subheader("Drop Unnecessary Columns")
        cols_to_drop = st.multiselect("Select columns to remove:", options=all_cols)
        if st.button("Drop Selected Columns"):
            st.session_state.df.drop(columns=cols_to_drop, inplace=True)
            log_action(f"Dropped columns: {', '.join(cols_to_drop)}")
            st.rerun()

    with tab5:
        st.subheader("Rename a Column")
        col_to_rename = st.selectbox("Select a column to rename:", options=all_cols, key="rename_select")
        new_col_name = st.text_input("Enter the new column name:", value=col_to_rename)
        if st.button("Rename Column"):
            st.session_state.df.rename(columns={col_to_rename: new_col_name}, inplace=True)
            log_action(f"Renamed '{col_to_rename}' to '{new_col_name}'.")
            st.rerun()

    with tab6:
        st.subheader("Change Column Data Type (Manual)")
        col_to_change = st.selectbox("Select column:", options=all_cols, key="type_change_select")
        new_type = st.selectbox("Select new data type:", ['object (string)', 'int64', 'float64', 'category', 'bool'])
        if st.button("Apply Type Change"):
            try:
                st.session_state.df[col_to_change] = st.session_state.df[col_to_change].astype(new_type)
                log_action(f"Manually changed type of '{col_to_change}' to {new_type}.")
                st.rerun()
            except Exception as e:
                st.error(f"Conversion failed. Error: {e}")

# ---------------------------------- 5.5 DUPLICATE HANDLING PAGE ---------------------------------
def render_duplicate_handling_page():
    """Renders the page for identifying and removing duplicate records."""
    st.header("📑 Row & Duplicate Manager")
    
    if st.session_state.df is None:
        st.warning("Please upload a file first.")
        return
        
    df = st.session_state.df
    
    tab1, tab2 = st.tabs(["Duplicate Rows", "Filter Rows"])

    with tab1:
        st.subheader("Handle Duplicate Rows")
        st.markdown("Find and remove duplicate rows from your dataset. Duplicates can skew analysis and machine learning model training.")
        
        duplicates = df[df.duplicated(keep=False)]
        num_duplicates_to_remove = df.duplicated().sum()

        if duplicates.empty:
            st.success("🎉 No duplicate rows found in the dataset.")
        else:
            st.warning(f"Found **{num_duplicates_to_remove}** duplicate rows (a total of {len(duplicates)} rows are part of a duplicate set).")
            st.dataframe(duplicates.sort_values(by=df.columns.tolist()))
            
            if st.button("Remove Duplicate Rows (keep first)", type="primary"):
                st.session_state.df.drop_duplicates(keep='first', inplace=True)
                log_action(f"Removed {num_duplicates_to_remove} duplicate rows.", "df.drop_duplicates(keep='first', inplace=True)")
                st.rerun()

    with tab2:
        st.subheader("Filter Rows with a Custom Query")
        st.markdown("Filter your dataset using pandas' powerful `query` syntax. This is useful for isolating subsets of your data for closer inspection.")
        st.info("""
        **Query Examples:**
        - Numeric: `Age > 30` or `Salary >= 50000`
        - String: `Country == "USA"` or `Name.str.contains("John", na=False)`
        - Combined: `Age > 30 and Country != "Canada"`
        
        Note: Column names with spaces or special characters must be enclosed in backticks (e.g., \`Column Name\`).
        """)

        query_string = st.text_area("Enter your pandas query string:", height=100)
        
        if st.button("Apply Filter"):
            if not query_string:
                st.warning("Please enter a query string.")
            else:
                try:
                    df_before_shape = df.shape
                    filtered_df = df.query(query_string)
                    rows_removed = df_before_shape[0] - filtered_df.shape[0]
                    st.session_state.df = filtered_df
                    log_action(f"Applied filter query: '{query_string}'. Removed {rows_removed} rows.", f"df = df.query('{query_string}')")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid query: {e}")
                    st.warning("Please check your syntax and column names.")

# ---------------------------------- 5.6 OUTLIER HANDLING PAGE -----------------------------------
def render_outlier_page():
    """Renders page for outlier detection and removal."""
    st.header("📈 Outlier Detection & Handling")
    st.markdown("Identify and manage outliers, which are data points that significantly differ from other observations.")
    
    if st.session_state.df is None:
        st.warning("Please upload a file first.")
        return
        
    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)

    if not numeric_cols:
        st.info("No numeric columns found for outlier detection.")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_col = st.selectbox("Select a numeric column:", numeric_cols)
    with col2:
        method = st.selectbox("Select detection method:", ["Inter-Quartile Range (IQR)", "Z-Score"])

    # --- Visualization ---
    st.subheader("1. Visualize Distribution")
    fig = px.box(df, y=selected_col, title=f"Box Plot for '{selected_col}'", points="all")
    st.plotly_chart(fig, use_container_width=True)

    # --- Detection and Removal ---
    st.subheader("2. Detect and Remove Outliers")
    if method == "Inter-Quartile Range (IQR)":
        st.markdown("This method defines outliers as data points that fall below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR`.")
        multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
        
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
        st.warning(f"Detected **{len(outliers)}** outliers in '{selected_col}' using the IQR method.")
        
        if not outliers.empty:
            if st.checkbox("Show detected outliers?"):
                st.dataframe(outliers)
            
            if st.button("Remove these outliers", type="primary"):
                df_cleaned = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)]
                st.session_state.df = df_cleaned
                log_action(f"Removed {len(outliers)} outliers from '{selected_col}' using IQR (multiplier={multiplier}).")
                st.rerun()

    else: # Z-Score
        st.markdown("This method defines outliers as data points with a Z-score (number of standard deviations from the mean) greater than a threshold.")
        threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
        
        from scipy.stats import zscore
        z_scores = zscore(df[selected_col].dropna())
        abs_z_scores = np.abs(z_scores)
        
        # We need to align the z-scores with the original dataframe index
        outlier_indices = df[selected_col].dropna()[abs_z_scores > threshold].index
        outliers = df.loc[outlier_indices]
        
        st.warning(f"Detected **{len(outliers)}** outliers in '{selected_col}' using Z-score (threshold={threshold}).")

        if not outliers.empty:
            if st.checkbox("Show detected outliers?"):
                st.dataframe(outliers)
            
            if st.button("Remove these outliers", type="primary"):
                df_cleaned = df.drop(outlier_indices)
                st.session_state.df = df_cleaned
                log_action(f"Removed {len(outliers)} outliers from '{selected_col}' using Z-Score (threshold={threshold}).")
                st.rerun()

# ---------------------------------- 5.7 TRANSFORMATION PAGE -----------------------------------
# --- Make sure this import is at the top of your app.py file! ---
def render_transformation_page():
    """Renders page for text-to-number, find/replace, text cleaning, normalization, scaling, and date extraction."""
    st.header("🔬 Data Transformation")
    st.markdown("Apply common transformations to prepare your data for modeling or analysis.")

    if st.session_state.df is None:
        st.warning("Please upload a file first.")
        return

    df = st.session_state.df

    # Define the tabs for this page
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Text-to-Number",
        "🔎 Find & Replace", 
        "📊 Normalize Categories", 
        "🔡 Text Cleaning", 
        "🔢 Numeric Scaling", 
        "📅 Datetime Feature Extraction"
    ])

    with tab1:
        st.subheader("Convert Number Words to Digits")
        st.markdown("Automatically convert text like 'thirty' into its digit form '30'.")
        
        def convert_word_to_number(value):
            try: return w2n.word_to_num(str(value))
            except ValueError: return value

        candidate_cols = get_categorical_columns(df) + get_numeric_columns(df)
        if not candidate_cols:
            st.info("No suitable columns found for this operation.")
        else:
            selected_col = st.selectbox("1. Select a column to convert:", options=candidate_cols, key="w2n_col_select")
            if selected_col:
                st.markdown("#### 2. Preview of Conversion")
                temp_series = df[selected_col].apply(convert_word_to_number)
                temp_series_numeric = pd.to_numeric(temp_series, errors='coerce')
                preview_df = pd.DataFrame({
                    "Original Value": df[selected_col].head(20),
                    "Converted Value": temp_series_numeric.head(20)
                })
                st.dataframe(preview_df.dropna(subset=[f"Original Value"]), use_container_width=True)
                if st.button("✅ Apply Text-to-Number Conversion", type="primary"):
                    converted_series = st.session_state.df[selected_col].apply(convert_word_to_number)
                    st.session_state.df[selected_col] = pd.to_numeric(converted_series, errors='coerce')
                    log_action(f"Converted number-words to digits in column '{selected_col}'.")
                    st.rerun()

    with tab2:
        st.subheader("Find and Replace Values in a Column")
        categorical_cols = get_categorical_columns(df)
        if not categorical_cols:
            st.info("No categorical/text columns to perform replacements on.")
        else:
            with st.form("find_replace_form"):
                selected_col_fr = st.selectbox("1. Select a column:", categorical_cols, key="fr_col_select")
                match_case = st.checkbox("Match Case", value=False)
                st.markdown("2. Define replacement rules:")
                rules_df = pd.DataFrame([{"Value to Find": "", "Replace With": ""}])
                edited_rules = st.data_editor(rules_df, num_rows="dynamic", use_container_width=True, key="find_replace_editor")
                submitted_fr = st.form_submit_button("Apply Replacements")
                if submitted_fr:
                    valid_rules = edited_rules.dropna(subset=["Value to Find"]).loc[edited_rules["Value to Find"] != ""]
                    if valid_rules.empty:
                        st.warning("No replacement rules were defined.")
                    else:
                        temp_col = st.session_state.df[selected_col_fr].astype(str)
                        if match_case:
                            replace_dict = dict(zip(valid_rules["Value to Find"], valid_rules["Replace With"]))
                            temp_col.replace(replace_dict, inplace=True)
                        else:
                            for _, rule in valid_rules.iterrows():
                                temp_col = temp_col.str.replace(f'^{rule["Value to Find"]}$', rule["Replace With"], case=False, regex=True)
                        st.session_state.df[selected_col_fr] = temp_col
                        log_action(f"Applied Find/Replace in '{selected_col_fr}'.")
                        st.rerun()

    with tab3:
        st.subheader("Normalize Categories (Visual Mapper)")
        categorical_cols_norm = get_categorical_columns(df)
        if not categorical_cols_norm:
            st.info("No categorical/text columns in the dataset.")
        else:
            selected_col_norm = st.selectbox("Select a column to normalize:", categorical_cols_norm, key="norm_col_select_visual")
            if selected_col_norm:
                with st.form("visual_normalization_form"):
                    unique_values = df[selected_col_norm].dropna().unique()
                    mapping_df = pd.DataFrame({"Original Value": unique_values, "New Value": unique_values})
                    edited_mapping_df = st.data_editor(mapping_df, use_container_width=True, key=f"editor_{selected_col_norm}")
                    submitted_visual = st.form_submit_button("Apply Visual Normalization")
                    if submitted_visual:
                        mapping_dict = {k: v for k, v in zip(edited_mapping_df["Original Value"], edited_mapping_df["New Value"]) if k != v}
                        if mapping_dict:
                            st.session_state.df[selected_col_norm] = st.session_state.df[selected_col_norm].replace(mapping_dict)
                            log_action(f"Normalized values in '{selected_col_norm}'.")
                            st.rerun()
                        else: st.warning("No changes were made.")

    with tab4:
        st.subheader("Clean Text Columns")
        text_cols = get_categorical_columns(df)
        if not text_cols:
            st.info("No text/categorical columns found.")
        else:
            selected_col_clean = st.selectbox("Select a text column to clean:", text_cols, key="text_clean_col")
            with st.form("text_cleaning_form"):
                to_lowercase = st.checkbox("Convert to lowercase")
                strip_whitespace = st.checkbox("Strip leading/trailing whitespace")
                remove_punctuation = st.checkbox("Remove punctuation")
                submitted_clean = st.form_submit_button("Apply Text Cleaning")
                if submitted_clean:
                    cleaned_series = df[selected_col_clean].astype(str)
                    log_items = []
                    if to_lowercase: cleaned_series = cleaned_series.str.lower(); log_items.append("lowercase")
                    if strip_whitespace: cleaned_series = cleaned_series.str.strip(); log_items.append("strip whitespace")
                    if remove_punctuation: cleaned_series = cleaned_series.str.replace(r'[^\w\s]', '', regex=True); log_items.append("remove punctuation")
                    st.session_state.df[selected_col_clean] = cleaned_series
                    if log_items:
                        log_action(f"Applied text cleaning ({', '.join(log_items)}) to '{selected_col_clean}'.")
                    else: st.warning("No cleaning options were selected.")

    with tab5:
        st.subheader("Scale Numeric Columns")
        numeric_cols = get_numeric_columns(df)
        if not numeric_cols:
            st.info("No numeric columns found for scaling.")
        else:
            scaler_type = st.radio("Select Scaler Type:", ["Min-Max Scaler", "Standard Scaler"])
            cols_to_scale = st.multiselect("Select numeric columns to scale:", numeric_cols)
            if st.button("Apply Scaler"):
                if cols_to_scale:
                    scaler = MinMaxScaler() if scaler_type.startswith("Min-Max") else StandardScaler()
                    st.session_state.df[cols_to_scale] = scaler.fit_transform(st.session_state.df[cols_to_scale])
                    log_action(f"Applied {scaler.__class__.__name__} to: {', '.join(cols_to_scale)}.")
                else: st.warning("Please select at least one column to scale.")

    with tab6:
        st.subheader("Extract Features from Datetime Columns")
        datetime_cols = get_datetime_columns(df)
        if not datetime_cols:
            st.info("No datetime columns found.")
        else:
            selected_col_dt = st.selectbox("Select a datetime column:", datetime_cols, key="dt_col")
            features_to_extract = st.multiselect("Select features:", ["Year", "Month", "Day", "Day of Week", "Hour"])
            if st.button("Extract Datetime Features"):
                if features_to_extract:
                    for feature in features_to_extract:
                        new_col_name = f"{selected_col_dt}_{feature.lower().replace(' ', '_')}"
                        if feature == "Year": st.session_state.df[new_col_name] = df[selected_col_dt].dt.year
                        if feature == "Month": st.session_state.df[new_col_name] = df[selected_col_dt].dt.month
                        if feature == "Day": st.session_state.df[new_col_name] = df[selected_col_dt].dt.day
                        if feature == "Day of Week": st.session_state.df[new_col_name] = df[selected_col_dt].dt.dayofweek
                        if feature == "Hour": st.session_state.df[new_col_name] = df[selected_col_dt].dt.hour
                    log_action(f"Extracted datetime features from '{selected_col_dt}'.")
                    st.rerun()
                else: st.warning("Please select features to extract.")

# ... (the rest of your imports) ...

# ---------------------------------- 5.4 COLUMN MANAGEMENT PAGE ----------------------------------
def render_column_management_page():
    """Renders page for date conversion, splitting, dropping, renaming, and type-casting columns."""
    st.header("🏛️ Column Operations")
    st.markdown("Perform column-level operations like converting dates, splitting, dropping, renaming, or changing types.")
    
    if st.session_state.df is None:
        st.warning("Please upload a file first.")
        return

    df = st.session_state.df
    all_cols = df.columns.tolist()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📅 Convert to Datetime",
        "📊 Analyze Column Types", 
        "✂️ Split Column", 
        "🗑️ Drop Columns", 
        "✏️ Rename Column", 
        "🔄 Change Column Type"
    ])
  # =========================================================================
    # --- CORRECTED FEATURE: Intelligent Date Conversion with Two Engines ---
    # =========================================================================
    with tab1:
        st.subheader("Intelligent Date Conversion")
        st.markdown("Convert columns with mixed date formats into a standardized datetime format.")

        candidate_cols = get_categorical_columns(df)
        if not candidate_cols:
            st.info("No text-based (object) columns found to convert.")
        else:
            selected_col = st.selectbox(
                "1. Select column to convert to datetime:",
                options=candidate_cols,
                key="date_convert_select"
            )

            if selected_col:
                st.markdown("#### 2. Choose Your Parsing Engine")
                
                parser_engine = st.radio(
                    "Select a parser:",
                    ["**Pandas (Fast)** - Good for standard formats like YYYY-MM-DD.", 
                     "**Dateparser (Flexible & Robust)** - Best for messy, mixed formats like 'March 3, 2021'."],
                    index=1
                )

                dayfirst_param = st.checkbox(
                    "Assume Day is First (for formats like DD/MM/YYYY)",
                    value=False,
                    help="Crucial for both engines to interpret ambiguous dates like '01/04/2021'."
                )

                st.markdown("#### 3. Preview Results")
                try:
                    # Helper function with the CORRECTED dateparser setting
                    def parse_with_dateparser(date_string):
                        if pd.isna(date_string): return pd.NaT
                        # THIS IS THE FIX: Use 'PREFER_DAY_OF_MONTH' instead of 'DAY_FIRST'
                        dateparser_settings = {
                            'PREFER_DAY_OF_MONTH': 'first' if dayfirst_param else 'last'
                        }
                        return dateparser.parse(str(date_string), settings=dateparser_settings)

                    if "Pandas" in parser_engine:
                        preview_series = pd.to_datetime(df[selected_col].astype(str).str.strip(), dayfirst=dayfirst_param, errors='coerce')
                    else: # Dateparser engine
                        preview_series = df[selected_col].apply(parse_with_dateparser)

                    preview_df = pd.DataFrame({
                        f"Original Text in '{selected_col}'": df[selected_col].head(20),
                        "Parsed Datetime (YYYY-MM-DD)": preview_series.head(20)
                    })
                    st.dataframe(preview_df.dropna(subset=[f"Original Text in '{selected_col}'"]), use_container_width=True)
                    
                    failed_parses = preview_series.isna().sum() - df[selected_col].isna().sum()
                    if failed_parses > 0:
                        st.warning(f"Warning: {failed_parses} values could not be parsed and were converted to NaT (Not a Time).")

                except Exception as e:
                    st.error(f"An error occurred during preview generation: {e}")

                if st.button("✅ Apply Datetime Conversion", type="primary"):
                    if "Pandas" in parser_engine:
                        st.session_state.df[selected_col] = pd.to_datetime(df[selected_col].astype(str).str.strip(), dayfirst=dayfirst_param, errors='coerce')
                    else: # Dateparser engine
                        st.session_state.df[selected_col] = df[selected_col].apply(parse_with_dateparser)
                    
                    log_action(f"Converted '{selected_col}' to datetime using {parser_engine.split(' ')[0]} engine.")
                    st.success(f"Successfully converted '{selected_col}' to a standardized datetime format!")
                    st.rerun()

    # =========================================================================
    # --- All Other Column Operations ---
    # =========================================================================
    with tab2:
        st.subheader("Analyze Column Data Types")
        numeric_cols = get_numeric_columns(df)
        categorical_cols = get_categorical_columns(df)
        datetime_cols = get_datetime_columns(df)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔢 Numeric Columns"); st.dataframe(pd.DataFrame(numeric_cols, columns=["Column Name"]), use_container_width=True)
            st.markdown("---")
            st.markdown("#### 📅 Datetime Columns"); st.dataframe(pd.DataFrame(datetime_cols, columns=["Column Name"]), use_container_width=True)
        with col2:
            st.markdown("#### 🅰️ Categorical / Object Columns"); st.dataframe(pd.DataFrame(categorical_cols, columns=["Column Name"]), use_container_width=True)

    with tab3:
        st.subheader("Split a Column into Categorical and Numerical Parts")
        candidate_cols_split = get_categorical_columns(df)
        if not candidate_cols_split:
            st.info("No suitable (text-based) columns found to split.")
        else:
            with st.form("split_column_form"):
                source_col = st.selectbox("Select column to split:", options=candidate_cols_split)
                col1, col2 = st.columns(2)
                with col1: new_cat_col_name = st.text_input("New Text Column Name", value=f"{source_col}_cat")
                with col2: new_num_col_name = st.text_input("New Number Column Name", value=f"{source_col}_num")
                drop_original = st.checkbox("Drop original column?", value=True)
                submitted = st.form_submit_button("Apply Split")
                if submitted:
                    st.session_state.df[new_cat_col_name] = df[source_col].str.extract(r'([^\d]*)', expand=False).str.strip()
                    numeric_part = df[source_col].str.extract(r'(\d+)', expand=False)
                    st.session_state.df[new_num_col_name] = pd.to_numeric(numeric_part, errors='coerce')
                    log_desc = f"Split '{source_col}' into '{new_cat_col_name}' and '{new_num_col_name}'."
                    if drop_original:
                        st.session_state.df.drop(columns=[source_col], inplace=True)
                        log_desc += f" Original column dropped."
                    log_action(log_desc)
                    st.rerun()

    with tab4:
        st.subheader("Drop Unnecessary Columns")
        cols_to_drop = st.multiselect("Select columns to remove:", options=all_cols)
        if st.button("Drop Selected Columns"):
            st.session_state.df.drop(columns=cols_to_drop, inplace=True)
            log_action(f"Dropped columns: {', '.join(cols_to_drop)}")
            st.rerun()

    with tab5:
        st.subheader("Rename a Column")
        col_to_rename = st.selectbox("Select a column to rename:", options=all_cols, key="rename_select")
        new_col_name = st.text_input("Enter the new column name:", value=col_to_rename)
        if st.button("Rename Column"):
            st.session_state.df.rename(columns={col_to_rename: new_col_name}, inplace=True)
            log_action(f"Renamed '{col_to_rename}' to '{new_col_name}'.")
            st.rerun()

    with tab6:
        st.subheader("Change Column Data Type (Manual)")
        col_to_change = st.selectbox("Select column:", options=all_cols, key="type_change_select")
        new_type = st.selectbox("Select new data type:", ['object (string)', 'int64', 'float64', 'category', 'bool'])
        if st.button("Apply Type Change"):
            try:
                st.session_state.df[col_to_change] = st.session_state.df[col_to_change].astype(new_type)
                log_action(f"Manually changed type of '{col_to_change}' to {new_type}.")
                st.rerun()
            except Exception as e:
                st.error(f"Conversion failed. Error: {e}")


# ==================================================================================================
# 5.8 ML MODELER PAGE (Corrected and Properly Encapsulated)
# ==================================================================================================
# ==================================================================================================
# 5.8 ML MODELER PAGE (Upgraded with Performance Visualizations)
# ==================================================================================================
# ==================================================================================================
# 5.8 ML MODELER PAGE (Upgraded with Comprehensive Metrics and Visualizations)
# ==================================================================================================
# ==================================================================================================
# 5.8 ML MODELER PAGE (Corrected with Hyperparameter Tuning AND Comprehensive Metrics)
# ==================================================================================================
def render_ml_modeler_page():
    """Renders the ML page with hyperparameter tuning, full metrics, and visualizations."""
    st.header("🤖 ML Modeler")
    st.markdown("Select an algorithm, tune its hyperparameters, and evaluate its performance in detail.")

    if st.session_state.df is None:
        st.warning("Please upload and prepare your data before modeling.")
        return

    df = st.session_state.df.copy()

    # --- Sidebar for Model Configuration ---
    with st.sidebar:
        st.header("1. Define Your Goal")
        target_variable = st.selectbox("Select Target Column (Y)", df.columns, help="This is the column to predict.")
        
        df.dropna(subset=[target_variable], inplace=True)
        if df.empty:
            st.error("DataFrame is empty after removing rows with missing targets.")
            st.stop()

        problem_type = "Regression" if pd.api.types.is_numeric_dtype(df[target_variable].dtype) and df[target_variable].nunique() > 25 else "Classification"
        st.info(f"Problem Type: **{problem_type}**")

        st.header("2. Choose Your Model")
        placeholder_text = "--Select an Algorithm--"
        if problem_type == "Classification":
            model_options = [placeholder_text, "Logistic Regression", "Random Forest Classifier", "Gradient Boosting", "XGBoost Classifier", "LightGBM Classifier", "SVC", "KNeighbors Classifier", "Decision Tree Classifier"]
        else:
            model_options = [placeholder_text, "Linear Regression", "Ridge", "Lasso", "Random Forest Regressor", "Gradient Boosting Regressor", "XGBoost Regressor", "LightGBM Regressor", "SVR"]
        
        selected_model_name = st.selectbox("Select an algorithm:", model_options, index=0)

    # --- Main Page Logic ---
    if selected_model_name == placeholder_text:
        st.info("💡 Please select an algorithm from the sidebar to configure its parameters and train it.")
        return

    # >>>>> HYPERPARAMETER TUNING LOGIC IS RESTORED HERE <<<<<
    with st.sidebar:
        st.header("3. Tune Model Hyperparameters")
        params = {}
        
        if selected_model_name == "Logistic Regression":
            params['C'] = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
            params['solver'] = st.selectbox("Solver", ['liblinear', 'lbfgs', 'saga'])
            params['max_iter'] = st.slider("Max Iterations", 100, 1000, 100, 50)
        elif selected_model_name in ["Random Forest Classifier", "Random Forest Regressor"]:
            params['n_estimators'] = st.slider("Number of Trees", 10, 1000, 100, 10)
            params['max_depth'] = st.slider("Max Depth of Trees", 2, 50, 10, 1)
            params['min_samples_leaf'] = st.slider("Min Samples per Leaf", 1, 20, 1, 1)
        elif selected_model_name in ["Gradient Boosting", "Gradient Boosting Regressor"]:
            params['n_estimators'] = st.slider("Number of Estimators", 10, 1000, 100, 10)
            params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            params['max_depth'] = st.slider("Max Depth", 2, 15, 3, 1)
        elif selected_model_name in ["XGBoost Classifier", "XGBoost Regressor"]:
            params['n_estimators'] = st.slider("Number of Estimators", 10, 1000, 100, 10)
            params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            params['max_depth'] = st.slider("Max Depth", 2, 15, 3, 1)
            params['subsample'] = st.slider("Subsample", 0.5, 1.0, 1.0, 0.1)
        elif selected_model_name == "SVC":
             params['C'] = st.slider("Regularization (C)", 0.01, 100.0, 1.0)
             params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
             params['probability'] = True # Always enable for metrics/plots
        # ... Add any other elif blocks for other models here ...

        with st.expander("Advanced Settings"):
            test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random Seed", value=42)

    if st.button(f"🚀 Train {selected_model_name}", type="primary", use_container_width=True):
        with st.spinner("Preparing data and training the model with your parameters..."):
            # (Pipeline and training code is the same)
            X = df.drop(columns=[target_variable])
            y = df[target_variable]
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
            numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
            model_class_map = {
                "Logistic Regression": LogisticRegression, "Random Forest Classifier": RandomForestClassifier, "Gradient Boosting": GradientBoostingClassifier,
                "XGBoost Classifier": xgb.XGBClassifier, "LightGBM Classifier": lgb.LGBMClassifier, "SVC": SVC,
                "KNeighbors Classifier": KNeighborsClassifier, "Decision Tree Classifier": DecisionTreeClassifier, "Linear Regression": LinearRegression,
                "Ridge": Ridge, "Lasso": Lasso, "Random Forest Regressor": RandomForestRegressor,
                "Gradient Boosting Regressor": GradientBoostingRegressor, "XGBoost Regressor": xgb.XGBRegressor,
                "LightGBM Regressor": lgb.LGBMRegressor, "SVR": SVR
            }
            ModelClass = model_class_map[selected_model_name]
            
            # Add random_state to params if model supports it
            if 'random_state' in ModelClass().get_params():
                params['random_state'] = random_state
            
            # Handle special cases for certain models
            if "XGBoost" in selected_model_name:
                params['eval_metric'] = 'logloss' if problem_type == "Classification" else 'rmse'
            
            model = ModelClass(**params) # Use the user-tuned params
            ml_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=(y if problem_type == "Classification" else None))
            ml_pipeline.fit(X_train, y_train)
            y_pred = ml_pipeline.predict(X_test)

        st.success("Model training complete!")
        st.header("Model Performance")
        
        tab1, tab2 = st.tabs(["📊 Metrics Dashboard", "📈 Visualizations"])

        with tab1:
            # >>>>> THE NEW COMPREHENSIVE METRICS DASHBOARD <<<<<
            if problem_type == "Classification":
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
                y_proba = ml_pipeline.predict_proba(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted') if len(y.unique()) > 2 else roc_auc_score(y_test, y_proba[:, 1])

                st.subheader("Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{acc:.3f}")
                col2.metric("F1-Score (Weighted)", f"{f1:.3f}")
                col3.metric("AUC (ROC)", f"{auc_score:.3f}")
                col4, col5, col6 = st.columns(3)
                col4.metric("Precision (Weighted)", f"{prec:.3f}")
                col5.metric("Recall (Weighted)", f"{recall:.3f}")
                col6.metric("MCC", f"{mcc:.3f}")

                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose())
            else: # Regression
                from sklearn.metrics import mean_absolute_error
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                st.subheader("Performance Metrics")
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                col1.metric("R-squared (R²)", f"{r2:.3f}")
                col2.metric("Mean Absolute Error (MAE)", f"{mae:.3f}")
                col3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.3f}")
                col4.metric("Mean Squared Error (MSE)", f"{mse:.3f}")

        with tab2:
            # (The visualization code remains the same)
            st.subheader("Performance Plots")
            if problem_type == "Classification":
                from sklearn.metrics import roc_curve, auc, precision_recall_curve
                if hasattr(ml_pipeline, "predict_proba"):
                    y_proba = ml_pipeline.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.2f})', labels=dict(x='False Positive Rate', y='True Positive Rate'))
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig_roc, use_container_width=True)
                st.markdown("#### Confusion Matrix")
                fig_cm, ax_cm = plt.subplots()
                cm = confusion_matrix(y_test, y_pred, labels=ml_pipeline.classes_)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=ml_pipeline.classes_, yticklabels=ml_pipeline.classes_)
                st.pyplot(fig_cm)
            else: # Regression
                st.markdown("#### Actual vs. Predicted Values")
                fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title="Actual vs. Predicted")
                fig_pred.add_shape(type='line', line=dict(dash='dash'), x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max())
                st.plotly_chart(fig_pred, use_container_width=True)
                st.markdown("#### Residuals Plot")
                residuals = y_test - y_pred
                fig_res = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'}, title="Residuals vs. Predicted Values")
                fig_res.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig_res, use_container_width=True)
            model_step = ml_pipeline.named_steps['model']
            if hasattr(model_step, 'feature_importances_') or hasattr(model_step, 'coef_'):
                st.markdown("#### Feature Importance")
                importances = model_step.feature_importances_ if hasattr(model_step, 'feature_importances_') else model_step.coef_[0]
                feature_names = ml_pipeline.named_steps['preprocessor'].get_feature_names_out()
                feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(importances)}).sort_values(by='Importance', ascending=False).head(20)
                fig_imp = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', title="Top 20 Most Important Features")
                st.plotly_chart(fig_imp, use_container_width=True)
# ---------------------------------- 5.9 DOWNLOAD PAGE -------------------------------------------
def render_download_page():
    """Renders the final page for downloading the cleaned data."""
    st.header("📥 Download & Export")
    st.markdown("Your data has been processed! Download the cleaned version as a new CSV file.")
    st.balloons()
    
    if st.session_state.df is None:
        st.warning("No data available to download.")
        return
        
    df_cleaned = st.session_state.df
    st.subheader("Final Data Preview")
    st.dataframe(df_cleaned.head())
    st.info(f"The final dataset has **{df_cleaned.shape[0]} rows** and **{df_cleaned.shape[1]} columns**.")

    # --- Download Button ---
    csv = df_cleaned.to_csv(index=False).encode('utf-8')
    suggested_filename = f"cleaned_{st.session_state.file_name}" if st.session_state.file_name else "cleaned_data.csv"
    
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name=suggested_filename,
        mime="text/csv",
        type="primary",
        help="Click to save the cleaned data to your local machine."
    )

# ==================================================================================================
# 6. MAIN APPLICATION ORCHESTRATOR
# ==================================================================================================

def main():
    """
    The main function that orchestrates the Streamlit application.
    It sets up the page configuration, initializes state, and handles navigation.
    """
    # --- Page Configuration ---
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Initialize Session State (crucial for stability) ---
    initialize_session_state()

    # --- Sidebar Navigation ---
    with st.sidebar:
        st.header("⚙️ Cleaning Workflow")

        # 1. Determine available pages based on file upload status
        if st.session_state.file_uploaded:
            available_pages = list(PAGES.keys())
        else:
            available_pages = [list(PAGES.keys())[0]]

        # 2. Defensive check for active page validity
        if st.session_state.active_page not in available_pages:
            st.session_state.active_page = available_pages[0]

        # 3. Create the radio button with a guaranteed valid index
        default_index = available_pages.index(st.session_state.active_page)
        st.session_state.active_page = st.radio(
            "Go to:",
            options=available_pages,
            index=default_index,
            key='navigation_radio'
        )
        
        st.markdown("---")

        # --- Sidebar Action Buttons ---
        if st.session_state.file_uploaded:
            if st.button("↩️ Reset All Changes", help="Revert to the originally uploaded data."):
                st.session_state.df = st.session_state.df_original.copy()
                st.session_state.history = []
                log_action("All changes have been reset to the original file state.")
                st.rerun()

            if st.button("⬆️ Upload a New File", help="Reset the entire app and upload a new file."):
                reset_app_state() # This function already calls rerun

        st.markdown("---")
        st.info("Created by Chohan.")

    # --- Page Routing ---
    # Retrieve the function name from the PAGES dictionary and call it
    page_function_name = PAGES.get(st.session_state.active_page)
    if page_function_name:
        # Use getattr to dynamically call the correct render function
        page_function = globals().get(page_function_name)
        if page_function:
            page_function()
        else:
            st.error(f"Error: Could not find the function {page_function_name}.")
            render_home_page()
    else:
        render_home_page() # Fallback to home page


if __name__ == "__main__":
    main()
