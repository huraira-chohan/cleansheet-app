# --- Core and Data Handling Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from typing import List, Dict, Callable
from word2number import w2n
# --- Visualization Imports ---
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import dateparser
# --- Machine Learning & Preprocessing Imports ---
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
# --- Make sure to add this new import at the top of your app.py file! ---
import dateparser

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
# --- Make sure to add this import at the top of your app.py file! ---
from word2number import w2n

# ... (the rest of your imports) ...

# ---------------------------------- 5.7 TRANSFORMATION PAGE -----------------------------------
def render_transformation_page():
    """Renders page for text-to-number, find/replace, text cleaning, normalization, scaling, and date extraction."""
    st.header("🔬 Data Transformation")
    st.markdown("Apply common transformations to prepare your data for modeling or analysis.")

    if st.session_state.df is None:
        st.warning("Please upload a file first.")
        return

    df = st.session_state.df

    # ADDED the new "Text-to-Number" tab as the first and most specialized option
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📝 Text-to-Number",
        "🔎 Find & Replace", 
        "📊 Normalize Categories", 
        "🔡 Text Cleaning", 
        "🔢 Numeric Scaling", 
        "📅 Datetime Feature Extraction"
    ])

    # =========================================================================
    # --- YOUR NEW FEATURE: Text-to-Number Conversion Tab ---
    # =========================================================================
    with tab1:
        st.subheader("Convert Number Words to Digits")
        st.markdown("Automatically convert text representations of numbers (e.g., 'thirty', 'five hundred') into their digit form (e.g., 30, 500). This is perfect for columns like 'Age' or 'Quantity' with inconsistent data entry.")
        
        # Helper function to safely convert a single value
        def convert_word_to_number(value):
            try:
                # Attempt to convert the word to a number
                return w2n.word_to_num(str(value))
            except ValueError:
                # If it fails (e.g., it's already a number, or it's non-numeric text like 'N/A'),
                # return the original value untouched.
                return value

        candidate_cols = get_categorical_columns(df) + get_numeric_columns(df)
        if not candidate_cols:
            st.info("No suitable columns found for this operation.")
        else:
            selected_col = st.selectbox(
                "1. Select a column to convert:",
                options=candidate_cols,
                key="w2n_col_select",
                help="Best for columns that mix numbers (56) and number-words ('thirty')."
            )

            if selected_col:
                # --- Live Preview ---
                st.markdown("#### 2. Preview of Conversion")
                preview_df = pd.DataFrame()
                # Create a temporary series for the preview without changing the state
                temp_series = df[selected_col].apply(convert_word_to_number)
                # Coerce to numeric for a clean final preview, turning errors into NaT/NaN
                temp_series_numeric = pd.to_numeric(temp_series, errors='coerce')

                preview_df[f"Original Value"] = df[selected_col].head(20)
                preview_df[f"Converted Value"] = temp_series_numeric.head(20)
                st.dataframe(preview_df.dropna(subset=[f"Original Value"]), use_container_width=True)

                if st.button("✅ Apply Text-to-Number Conversion", type="primary"):
                    # Apply the conversion to the actual dataframe in session state
                    converted_series = st.session_state.df[selected_col].apply(convert_word_to_number)
                    # The final step is to convert the entire column to a numeric type,
                    # which will handle original numbers and newly converted numbers correctly.
                    # 'coerce' will turn any remaining non-numeric text into 'NaN' (Not a Number).
                    st.session_state.df[selected_col] = pd.to_numeric(converted_series, errors='coerce')

                    log_action(f"Converted number-words to digits in column '{selected_col}'.")
                    st.success(f"Successfully processed '{selected_col}'. Non-numeric words were converted to numbers.")
                    st.rerun()
    
    # =========================================================================
    # --- All Other Transformation Features ---
    # =========================================================================
    with tab2:
        st.subheader("Find and Replace Values in a Column")
        categorical_cols = get_categorical_columns(df)
        if not categorical_cols:
            st.info("No categorical/text columns found to perform replacements on.")
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
            st.info("No categorical/text columns found in the dataset.")
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
                    # Chained operations for cleaner code
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
            scaler_type = st.radio("Select Scaler Type:", ["Min-Max Scaler (to [0, 1])", "Standard Scaler (zero mean, unit variance)"])
            cols_to_scale = st.multiselect("Select numeric columns to scale:", numeric_cols)
            if st.button("Apply Scaler"):
                if not cols_to_scale:
                    st.warning("Please select at least one column to scale.")
                else:
                    scaler = MinMaxScaler() if scaler_type.startswith("Min-Max") else StandardScaler()
                    st.session_state.df[cols_to_scale] = scaler.fit_transform(st.session_state.df[cols_to_scale])
                    log_action(f"Applied {scaler.__class__.__name__} to: {', '.join(cols_to_scale)}.")

    with tab6:
        st.subheader("Extract Features from Datetime Columns")
        datetime_cols = get_datetime_columns(df)
        if not datetime_cols:
            st.info("No datetime columns found. Use 'Column Operations' to change types first.")
        else:
            selected_col_dt = st.selectbox("Select a datetime column:", datetime_cols, key="dt_col")
            features_to_extract = st.multiselect("Select features:", ["Year", "Month", "Day", "Day of Week", "Hour"])
            if st.button("Extract Datetime Features"):
                if not features_to_extract:
                    st.warning("Please select features to extract.")
                else:
                    for feature in features_to_extract:
                        new_col_name = f"{selected_col_dt}_{feature.lower().replace(' ', '_')}"
                        if feature == "Year": st.session_state.df[new_col_name] = df[selected_col_dt].dt.year
                        if feature == "Month": st.session_state.df[new_col_name] = df[selected_col_dt].dt.month
                        if feature == "Day": st.session_state.df[new_col_name] = df[selected_col_dt].dt.day
                        if feature == "Day of Week": st.session_state.df[new_col_name] = df[selected_col_dt].dt.dayofweek
                        if feature == "Hour": st.session_state.df[new_col_name] = df[selected_col_dt].dt.hour
                    log_action(f"Extracted datetime features from '{selected_col_dt}'.")
                    st.rerun()
# ---------------------------------- 5.8 ACTION HISTORY PAGE -------------------------------------
def render_history_page():
    """Renders the page showing a log of all cleaning actions taken."""
    st.header("📜 Action History")
    st.markdown("A complete log of all cleaning and transformation steps applied to the dataset.")
    
    if not st.session_state.history:
        st.info("No actions have been performed yet.")
    else:
        for i, action in enumerate(reversed(st.session_state.history)):
            with st.expander(f"**Step {len(st.session_state.history) - i}:** {action['description']}"):
                if action['code']:
                    st.code(action['code'], language='python')
                else:
                    st.write("No code snippet available for this action.")

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
