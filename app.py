# ==============================================================================
# Comprehensive CSV Data Cleaning Streamlit Application
#
# Author: Gemini Advanced (Google)
# Date: May 2024
#
# Description:
# A full-featured, robust, and user-friendly Streamlit application designed for
# comprehensive CSV file cleaning. This application provides a multi-page
# interface to guide the user through the entire data cleaning pipeline, from
# loading and initial inspection to advanced cleaning tasks and final export.
#
# Features:
# 1.  File Upload and Session State Management.
# 2.  Data Overview: Head, Tail, Info, Description, Value Counts.
# 3.  Missing Value Analysis and Handling (Drop/Impute).
# 4.  Column Management: Drop, Rename, Change Data Type.
# 5.  Duplicate Record Handling.
# 6.  Outlier Detection and Removal (IQR, Z-Score).
# 7.  Text Data Cleaning Tools.
# 8.  Interactive Data Filtering and Querying.
# 9.  Download Cleaned Data.
#
# The code is designed to be modular, robust, and well-documented to meet
# the challenge requirements.
# ==============================================================================

# --- Core Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import io
import base64

# --- Visualization Imports ---
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================================================================
# 1. APPLICATION SETUP AND STATE MANAGEMENT
# ==============================================================================

def initialize_session_state():
    """
    Initializes the Streamlit session state variables.
    This function is called once at the start of the script to ensure all
    necessary keys are present in st.session_state.
    """
    # --- DataFrame Storage ---
    if 'df' not in st.session_state:
        st.session_state.df = None  # Holds the current working dataframe
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None  # Holds a backup of the original df

    # --- Control Flags ---
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0  # Used to reset the file uploader
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False # Flag to check if a file has been processed

    # --- UI State for specific pages ---
    if 'active_page' not in st.session_state:
        st.session_state.active_page = "Home"


def reset_app_state():
    """
    Resets the application to its initial state.
    This involves clearing the DataFrames and resetting control flags.
    It's triggered by the 'Upload a new file' button.
    """
    st.session_state.df = None
    st.session_state.df_original = None
    st.session_state.file_uploaded = False
    st.session_state.file_uploader_key += 1 # Increment key to force re-render of file_uploader
    st.session_state.active_page = "Home"  # <-- THE FIX: Reset the active page to a valid default.
    st.toast("Application has been reset. Please upload a new file.", icon="🔄")


# ==============================================================================
# 2. HELPER & UTILITY FUNCTIONS
# ==============================================================================

def get_dataframe_info(df: pd.DataFrame) -> str:
    """
    Captures the output of df.info() into a string.
    Standard df.info() prints to stdout, so we need to redirect it to capture
    it for display in Streamlit.

    Args:
        df (pd.DataFrame): The DataFrame to get info from.

    Returns:
        str: The string representation of the DataFrame's info.
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Identifies and returns a list of numeric column names from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list of column names that are of a numeric data type.
    """
    if df is None:
        return []
    return df.select_dtypes(include=np.number).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> list:
    """
    Identifies and returns a list of categorical/object column names.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list of column names that are of 'object', 'category', or 'boolean' type.
    """
    if df is None:
        return []
    return df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

def get_datetime_columns(df: pd.DataFrame) -> list:
    """
    Identifies and returns a list of datetime column names from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        list: A list of column names that are of a datetime data type.
    """
    if df is None:
        return []
    return df.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()


def generate_download_link(df: pd.DataFrame, filename: str) -> str:
    """
    Generates a link to download the given DataFrame as a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to download.
        filename (str): The desired filename for the downloaded file.

    Returns:
        str: An HTML anchor tag for the download link.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Click here to download</a>'


# ==============================================================================
# 3. UI RENDERING - PAGE COMPONENTS
# ==============================================================================

# ------------------------------------------------------------------------------
# 3.1. HOME / UPLOAD PAGE
# ------------------------------------------------------------------------------
def render_home_page():
    """
    Renders the content for the Home page, primarily handling file uploads.
    """
    st.title("🚀 Comprehensive CSV Data Cleaner")
    st.markdown("""
    Welcome to the ultimate data cleaning tool! This application is designed to
    help you systematically clean and prepare your CSV data for analysis.

    **Follow these steps to get started:**

    1.  **Upload your CSV file** using the uploader below.
    2.  Use the **navigation sidebar** on the left to move between cleaning tasks.
    3.  Each action you take will update the dataset in real-time.
    4.  Once you're satisfied, navigate to the **Download** page to get your cleaned file.

    Let's begin!
    """)
    st.subheader("1. Upload Your CSV File")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        key=f"file_uploader_{st.session_state.file_uploader_key}",
        help="Upload a CSV file to begin the cleaning process."
    )

    if uploaded_file is not None:
        try:
            # Process the uploaded file
            df = pd.read_csv(uploaded_file)
            st.session_state.df_original = df.copy()
            st.session_state.df = df.copy()
            st.session_state.file_uploaded = True
            st.success("File uploaded successfully! A preview is shown below.")
            st.write("### Quick Preview of Your Data")
            st.dataframe(st.session_state.df.head())
            st.info("You can now use the sidebar to navigate to different cleaning modules.")

        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
            st.warning("Please ensure the file is a valid CSV and try again.")
            st.session_state.file_uploaded = False

    elif st.session_state.file_uploaded:
        st.info("A file is already loaded. Use the sidebar to start cleaning.")
        st.write("### Quick Preview of Loaded Data")
        st.dataframe(st.session_state.df.head())


# ------------------------------------------------------------------------------
# 3.2. DATA OVERVIEW PAGE
# ------------------------------------------------------------------------------
def render_overview_page():
    """
    Renders the Data Overview page, showing detailed information about the
    loaded DataFrame.
    """
    st.title("📊 Data Overview")
    st.markdown("Get a high-level summary of your dataset. This helps in understanding the structure, data types, and basic statistics of your data before cleaning.")

    df = st.session_state.df

    with st.expander("▶️ DataFrame Head", expanded=True):
        st.write("Displaying the first 5 rows of your data.")
        st.dataframe(df.head())

    with st.expander("◀️ DataFrame Tail"):
        st.write("Displaying the last 5 rows of your data.")
        st.dataframe(df.tail())

    with st.expander("ℹ️ DataFrame Info"):
        st.write("A concise summary of the DataFrame, including data types and non-null values.")
        info_str = get_dataframe_info(df)
        st.text(info_str)

    with st.expander("🔢 DataFrame Description (Statistics)"):
        st.write("Descriptive statistics for all numeric columns in your dataset.")
        st.dataframe(df.describe())

    with st.expander("📋 Value Counts for Categorical Columns"):
        st.write("See the distribution of values in your categorical columns.")
        categorical_cols = get_categorical_columns(df)
        if categorical_cols:
            selected_col = st.selectbox(
                "Select a categorical column to see its value counts:",
                options=categorical_cols,
                help="Choose a column to inspect the frequency of each unique value."
            )
            if selected_col:
                value_counts = df[selected_col].value_counts().reset_index()
                value_counts.columns = [selected_col, 'Count']
                st.dataframe(value_counts)

                # Optional: Add a bar chart for visualization
                if st.checkbox(f"Show bar chart for '{selected_col}'?"):
                    try:
                        fig = px.bar(value_counts, x=selected_col, y='Count',
                                     title=f"Value Counts for {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate plot: {e}")
        else:
            st.info("No categorical columns found in the dataset.")

# ------------------------------------------------------------------------------
# 3.3. MISSING VALUE HANDLING PAGE
# ------------------------------------------------------------------------------
def render_missing_values_page():
    """
    Renders the page for analyzing and handling missing values.
    Provides options for visualization, dropping, and imputing.
    """
    st.title("❓ Missing Value Handling")
    st.markdown("Identify, visualize, and handle missing values (NaNs) in your dataset. Missing data can significantly impact analysis and model performance.")

    df = st.session_state.df
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    if missing_data.empty:
        st.success("🎉 Congratulations! No missing values found in your dataset.")
        return

    st.subheader("1. Missing Value Analysis")
    st.write("The following table and chart show the number of missing values per column.")

    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Missing Values per Column")
        st.dataframe(missing_data.rename("Number of Missing Values"))

    with col2:
        st.write("#### Missing Value Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
        ax.set_title("Heatmap of Missing Values")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("2. Handle Missing Values")

    # --- Option 1: Dropping Missing Values ---
    with st.expander("🚮 Option 1: Drop Missing Values"):
        st.write("Remove rows or columns containing missing values.")
        drop_option = st.radio(
            "How do you want to drop NaNs?",
            ('Drop rows with any missing values', 'Drop columns with any missing values'),
            key='drop_option'
        )

        if st.button("Apply Drop Operation", key='drop_button'):
            df_before_shape = df.shape
            if drop_option == 'Drop rows with any missing values':
                st.session_state.df = df.dropna(axis=0)
                st.success(f"Successfully dropped rows with missing values. Shape changed from {df_before_shape} to {st.session_state.df.shape}.")
            else: # Drop columns
                st.session_state.df = df.dropna(axis=1)
                st.success(f"Successfully dropped columns with missing values. Shape changed from {df_before_shape} to {st.session_state.df.shape}.")

            st.rerun() # Rerun to update the page with new state

    # --- Option 2: Imputing Missing Values ---
    with st.expander("✍️ Option 2: Fill (Impute) Missing Values"):
        st.write("Replace missing values with a calculated or specified value.")
        impute_cols = missing_data.index.tolist()
        selected_col_impute = st.selectbox(
            "Select a column to impute:",
            options=impute_cols,
            help="Choose the column where you want to fill missing values."
        )

        if selected_col_impute:
            imputation_method = st.selectbox(
                f"Select imputation method for '{selected_col_impute}':",
                ('Mean', 'Median', 'Mode', 'Custom Value'),
                help="""
                - **Mean**: Fills with the average value (for numeric columns).
                - **Median**: Fills with the middle value (for numeric columns).
                - **Mode**: Fills with the most frequent value (for all column types).
                - **Custom Value**: Fills with a value you specify.
                """
            )
            custom_value = None
            if imputation_method == 'Custom Value':
                custom_value = st.text_input("Enter the custom value to fill with:")

            if st.button("Apply Imputation", key='impute_button'):
                try:
                    fill_value = None
                    if imputation_method == 'Mean':
                        fill_value = df[selected_col_impute].mean()
                    elif imputation_method == 'Median':
                        fill_value = df[selected_col_impute].median()
                    elif imputation_method == 'Mode':
                        fill_value = df[selected_col_impute].mode()[0]
                    elif imputation_method == 'Custom Value':
                        # Attempt to cast custom value to the column's type
                        col_type = df[selected_col_impute].dtype
                        try:
                            fill_value = pd.Series([custom_value]).astype(col_type).iloc[0]
                        except Exception:
                           st.warning(f"Could not convert '{custom_value}' to type {col_type}. Using it as a string.")
                           fill_value = custom_value

                    if fill_value is not None:
                        st.session_state.df[selected_col_impute] = df[selected_col_impute].fillna(fill_value)
                        st.success(f"Successfully imputed missing values in '{selected_col_impute}' with '{fill_value}'.")
                        st.rerun() # Rerun to update the page state
                    else:
                        st.error("Could not determine a value for imputation. Please check your inputs.")

                except TypeError as te:
                    st.error(f"A type error occurred: {te}. The selected method might not be applicable for this column's data type (e.g., 'Mean' on a text column).")
                except Exception as e:
                    st.error(f"An error occurred during imputation: {e}")


# ------------------------------------------------------------------------------
# 3.4. COLUMN MANAGEMENT PAGE
# ------------------------------------------------------------------------------
def render_column_management_page():
    """
    Renders the page for managing columns: dropping, renaming, and changing types.
    """
    st.title("🏛️ Column Management")
    st.markdown("Perform essential column-level operations like dropping, renaming, or changing data types.")

    df = st.session_state.df
    all_cols = df.columns.tolist()

    # --- Section 1: Drop Columns ---
    st.subheader("1. Drop Columns")
    with st.expander("Select columns to drop", expanded=True):
        cols_to_drop = st.multiselect(
            "Select one or more columns to permanently remove from the dataset:",
            options=all_cols,
            help="Be careful, this action cannot be undone without resetting the app."
        )
        if st.button("Drop Selected Columns", type="primary"):
            if cols_to_drop:
                st.session_state.df = df.drop(columns=cols_to_drop)
                st.success(f"Successfully dropped columns: {', '.join(cols_to_drop)}")
                st.rerun()
            else:
                st.warning("Please select at least one column to drop.")

    st.markdown("---")

    # --- Section 2: Rename Columns ---
    st.subheader("2. Rename a Column")
    with st.expander("Select a column to rename"):
        col1, col2 = st.columns(2)
        with col1:
            col_to_rename = st.selectbox(
                "Select a column:",
                options=all_cols,
                key='rename_select'
            )
        with col2:
            new_col_name = st.text_input(
                "Enter the new name:",
                value=col_to_rename
            )

        if st.button("Rename Column"):
            if col_to_rename and new_col_name:
                if new_col_name in df.columns and new_col_name != col_to_rename:
                    st.error(f"A column named '{new_col_name}' already exists.")
                else:
                    st.session_state.df = df.rename(columns={col_to_rename: new_col_name})
                    st.success(f"Renamed column '{col_to_rename}' to '{new_col_name}'.")
                    st.rerun()
            else:
                st.warning("Please select a column and provide a new name.")

    st.markdown("---")

    # --- Section 3: Change Column Data Type ---
    st.subheader("3. Change Column Data Type")
    with st.expander("Select a column and a new data type"):
        col1, col2 = st.columns(2)
        with col1:
            col_to_change_type = st.selectbox(
                "Select a column:",
                options=all_cols,
                key='type_change_select'
            )

        with col2:
            new_type = st.selectbox(
                "Select the new data type:",
                ['object (string)', 'int64', 'float64', 'datetime64[ns]', 'category', 'bool']
            )

        if st.button("Change Data Type"):
            if col_to_change_type and new_type:
                try:
                    df_copy = df.copy() # Work on a copy to report errors
                    original_non_nulls = df_copy[col_to_change_type].notnull().sum()

                    if new_type == 'datetime64[ns]':
                        df_copy[col_to_change_type] = pd.to_datetime(df_copy[col_to_change_type], errors='coerce')
                    else:
                        df_copy[col_to_change_type] = df_copy[col_to_change_type].astype(new_type, errors='ignore')

                    st.session_state.df[col_to_change_type] = df_copy[col_to_change_type]

                    # Check for new NaNs created by coercion
                    new_nans = df_copy[col_to_change_type].isnull().sum() - df[col_to_change_type].isnull().sum()

                    st.success(f"Successfully changed data type of '{col_to_change_type}' to '{new_type}'.")
                    if new_nans > 0:
                        st.warning(f"Warning: {new_nans} values could not be converted and were set to NaN (Not a Number).")

                    st.rerun()

                except ValueError as ve:
                    st.error(f"ValueError: Could not convert column '{col_to_change_type}' to '{new_type}'. Check if all values in the column are compatible. Error: {ve}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")


# ------------------------------------------------------------------------------
# 3.5. DUPLICATE HANDLING PAGE
# ------------------------------------------------------------------------------
def render_duplicates_page():
    """
    Renders the page for identifying and removing duplicate records.
    """
    st.title("📑 Duplicate Record Handling")
    st.markdown("Find and remove duplicate rows from your dataset. Duplicates can skew analysis and machine learning model training.")

    df = st.session_state.df
    duplicates = df[df.duplicated(keep=False)]

    st.subheader("1. Identify Duplicates")
    if duplicates.empty:
        st.success("🎉 No duplicate rows found in the dataset.")
    else:
        st.warning(f"Found {len(duplicates)} rows that are part of a duplicate set. A preview is shown below.")
        st.dataframe(duplicates.sort_values(by=df.columns.tolist()))

        st.markdown("---")
        st.subheader("2. Remove Duplicates")
        st.write(f"There are **{df.duplicated().sum()}** duplicate rows that can be removed (keeping the first occurrence).")
        if st.button("Remove Duplicate Rows", type="primary"):
            df_before_shape = df.shape
            st.session_state.df = df.drop_duplicates(keep='first').reset_index(drop=True)
            df_after_shape = st.session_state.df.shape
            st.success(f"Successfully removed {df_before_shape[0] - df_after_shape[0]} duplicate rows.")
            st.info(f"DataFrame shape changed from {df_before_shape} to {df_after_shape}.")
            st.rerun()


# ------------------------------------------------------------------------------
# 3.6. OUTLIER HANDLING PAGE
# ------------------------------------------------------------------------------
def render_outlier_page():
    """
    Renders the page for outlier detection and removal using statistical methods.
    """
    st.title("📈 Outlier Handling")
    st.markdown("""
    Outliers are data points that differ significantly from other observations. They can be caused by measurement errors or represent genuine, but rare, variance in the data. This page helps you detect and remove them.
    """)

    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)

    if not numeric_cols:
        st.warning("No numeric columns found. Outlier detection requires numeric data.")
        return

    st.subheader("1. Select Column and Method")
    col1, col2 = st.columns(2)
    with col1:
        selected_col_outlier = st.selectbox(
            "Select a numeric column for outlier analysis:",
            options=numeric_cols
        )
    with col2:
        outlier_method = st.selectbox(
            "Select outlier detection method:",
            ("Inter-Quartile Range (IQR)", "Z-Score")
        )

    # --- Method Explanations ---
    if outlier_method == "Inter-Quartile Range (IQR)":
        st.info("""
        **IQR Method:** An outlier is a data point that falls outside the 1.5 * IQR range.
        - **IQR** = Q3 (75th percentile) - Q1 (25th percentile).
        - **Lower Bound** = Q1 - 1.5 * IQR
        - **Upper Bound** = Q3 + 1.5 * IQR
        This method is robust to outliers themselves.
        """)
        multiplier = st.slider("IQR Multiplier:", min_value=1.0, max_value=3.0, value=1.5, step=0.1, help="A higher multiplier is more lenient and keeps more data.")
    else: # Z-Score
        st.info("""
        **Z-Score Method:** An outlier is a data point with a Z-score greater than a certain threshold.
        - **Z-Score** = (Data Point - Mean) / Standard Deviation
        - It measures how many standard deviations a data point is from the mean.
        This method is sensitive to outliers as they influence the mean and std dev.
        """)
        threshold = st.slider("Z-Score Threshold:", min_value=1.0, max_value=5.0, value=3.0, step=0.1, help="A higher threshold is more lenient. 3 is a common choice.")

    st.markdown("---")
    st.subheader("2. Visualize and Remove Outliers")

    if selected_col_outlier:
        # --- Visualization ---
        st.write(f"#### Box Plot for '{selected_col_outlier}'")
        st.write("This plot helps visualize the distribution and identify potential outliers.")
        fig = px.box(df, y=selected_col_outlier, title=f"Distribution of '{selected_col_outlier}' Before Outlier Removal")
        st.plotly_chart(fig, use_container_width=True)

        # --- Detection and Removal ---
        if st.button(f"Remove Outliers from '{selected_col_outlier}'", type="primary"):
            df_before_shape = df.shape
            outliers_removed_count = 0

            if outlier_method == "Inter-Quartile Range (IQR)":
                Q1 = df[selected_col_outlier].quantile(0.25)
                Q3 = df[selected_col_outlier].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR

                df_cleaned = df[(df[selected_col_outlier] >= lower_bound) & (df[selected_col_outlier] <= upper_bound)]
                outliers_removed_count = df.shape[0] - df_cleaned.shape[0]
                st.session_state.df = df_cleaned

            else: # Z-Score
                from scipy.stats import zscore
                df_copy = df.copy()
                df_copy['zscore'] = zscore(df_copy[selected_col_outlier].dropna())
                df_cleaned = df_copy[df_copy['zscore'].abs() <= threshold].drop(columns=['zscore'])
                outliers_removed_count = df.shape[0] - df_cleaned.shape[0]
                st.session_state.df = df_cleaned


            st.success(f"Removed {outliers_removed_count} outliers from '{selected_col_outlier}'.")
            st.info(f"DataFrame shape changed from {df_before_shape} to {st.session_state.df.shape}.")

            st.write(f"#### Box Plot for '{selected_col_outlier}' After Removal")
            fig_after = px.box(st.session_state.df, y=selected_col_outlier, title=f"Distribution of '{selected_col_outlier}' After Outlier Removal")
            st.plotly_chart(fig_after, use_container_width=True)
            
            # Use st.rerun() carefully, only if necessary to fully reset the view
            # In this case, showing the 'after' plot is better before a full rerun.
            st.button("Confirm and Refresh Page")


# ------------------------------------------------------------------------------
# 3.7. TEXT CLEANING PAGE
# ------------------------------------------------------------------------------
def render_text_cleaning_page():
    """
    Renders the page for performing common text cleaning operations.
    """
    st.title("🔡 Text Data Cleaning")
    st.markdown("Clean your text (object/string) columns with a suite of common tools. This is crucial for natural language processing (NLP) tasks or standardizing categorical features.")

    df = st.session_state.df
    text_cols = get_categorical_columns(df)

    if not text_cols:
        st.warning("No text or categorical columns found to clean.")
        return

    selected_col_text = st.selectbox(
        "Select a text column to clean:",
        options=text_cols
    )

    if selected_col_text:
        st.subheader(f"Cleaning options for '{selected_col_text}':")
        # Ensure column is of string type for these operations
        df[selected_col_text] = df[selected_col_text].astype(str)

        cleaning_options = {
            "to_lowercase": st.checkbox("Convert to Lowercase"),
            "strip_whitespace": st.checkbox("Strip Leading/Trailing Whitespace"),
            "remove_punctuation": st.checkbox("Remove Punctuation"),
            "remove_stopwords": st.checkbox("Remove Common English Stopwords")
        }
        
        # Stopwords list for the feature
        stopwords = [
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        ]

        if st.button("Apply Text Cleaning Operations", type="primary"):
            cleaned_series = df[selected_col_text].copy()
            if cleaning_options["to_lowercase"]:
                cleaned_series = cleaned_series.str.lower()
                st.write("✅ Converted to lowercase.")
            if cleaning_options["strip_whitespace"]:
                cleaned_series = cleaned_series.str.strip()
                st.write("✅ Stripped whitespace.")
            if cleaning_options["remove_punctuation"]:
                cleaned_series = cleaned_series.str.replace(r'[^\w\s]', '', regex=True)
                st.write("✅ Removed punctuation.")
            if cleaning_options["remove_stopwords"]:
                # This is a basic implementation. More advanced ones would use libraries like NLTK.
                pat = r'\b(?:{})\b'.format('|'.join(stopwords))
                cleaned_series = cleaned_series.str.replace(pat, '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
                st.write("✅ Removed stopwords.")

            st.session_state.df[selected_col_text] = cleaned_series
            st.success(f"Text cleaning applied to column '{selected_col_text}'.")

            st.subheader("Preview of Changes")
            preview_df = pd.DataFrame({
                'Original': df[selected_col_text].head(),
                'Cleaned': st.session_state.df[selected_col_text].head()
            })
            st.dataframe(preview_df)


# ------------------------------------------------------------------------------
# 3.8. DATA FILTERING PAGE
# ------------------------------------------------------------------------------
def render_filtering_page():
    """
    Renders a page that allows the user to filter the DataFrame based on
    conditions.
    """
    st.title("🔍 Data Filtering and Querying")
    st.markdown("Filter your dataset based on specific conditions. This is useful for isolating subsets of your data for closer inspection or for creating a final dataset for a specific purpose.")

    df = st.session_state.df
    st.subheader("1. Build Your Filter")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_col = st.selectbox("Column to Filter On:", options=df.columns)
    
    # Dynamically select operator based on column type
    if pd.api.types.is_numeric_dtype(df[filter_col]):
        operators = ['==', '!=', '>', '<', '>=', '<=']
    else: # String/Object/Category
        operators = ['==', '!=', 'contains', 'not contains']

    with col2:
        operator = st.selectbox("Operator:", options=operators)

    with col3:
        # Use number_input for numeric types, text_input otherwise
        if pd.api.types.is_numeric_dtype(df[filter_col]):
            filter_value = st.number_input("Value:", value=0, format="%g")
        else:
            filter_value = st.text_input("Value:")

    # Construct the pandas query string
    try:
        if pd.api.types.is_numeric_dtype(df[filter_col]):
            query_string = f"`{filter_col}` {operator} {filter_value}"
        elif operator in ['contains', 'not contains']:
            # For string methods, query is a bit different
            if operator == 'contains':
                query_string = f"`{filter_col}`.str.contains('{filter_value}', na=False)"
            else: # not contains
                query_string = f"~`{filter_col}`.str.contains('{filter_value}', na=False)"
        else: # String equality/inequality
            query_string = f"`{filter_col}` {operator} '{filter_value}'"
            
        st.info(f"Generated Query: **{query_string}**")
        
        st.subheader("2. Apply Filter")
        if st.button("Apply Filter to DataFrame", type="primary"):
            try:
                df_before_shape = df.shape
                filtered_df = df.query(query_string)
                st.session_state.df = filtered_df.reset_index(drop=True)
                
                st.success(f"Filter applied successfully. {df_before_shape[0] - filtered_df.shape[0]} rows were removed.")
                st.info(f"DataFrame shape changed from {df_before_shape} to {st.session_state.df.shape}.")
                st.rerun()

            except Exception as e:
                st.error(f"Error applying filter: {e}")
                st.warning("Please check your query. String values must often be quoted. This app tries to do it for you, but complex cases may fail.")

    except Exception as e:
        st.error(f"Could not build filter. Error: {e}")

    st.subheader("Current DataFrame Preview")
    st.dataframe(df.head())


# ------------------------------------------------------------------------------
# 3.9. DOWNLOAD PAGE
# ------------------------------------------------------------------------------
def render_download_page():
    """
    Renders the final page for downloading the cleaned data.
    """
    st.title("📥 Download Cleaned Data")
    st.markdown("Your data has been processed! You can now download the cleaned version as a new CSV file.")
    st.balloons()

    df_cleaned = st.session_state.df

    st.subheader("Final Preview of Cleaned Data")
    st.dataframe(df_cleaned.head())
    st.info(f"The cleaned dataset has **{df_cleaned.shape[0]} rows** and **{df_cleaned.shape[1]} columns**.")

    # --- Download Button ---
    csv = df_cleaned.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv",
        type="primary",
        help="Click to save the cleaned data to your local machine."
    )


# ==============================================================================
# 4. MAIN APPLICATION LOGIC
# ==============================================================================

def main():
    """
    The main function that orchestrates the Streamlit application.
    It sets up the page configuration, initializes state, and handles navigation.
    """
    # --- Page Configuration ---
    st.set_page_config(
        page_title="Pro CSV Cleaner",
        page_icon="🧹",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Initialize Session State ---
    initialize_session_state()

    # --- Sidebar Navigation ---
    with st.sidebar:
        st.header("✨ Cleaning Workflow")

        # Conditional Reset Button
        if st.session_state.df is not None:
            if st.button("↩️ Reset to Original", help="Revert all changes and start over with the original data."):
                st.session_state.df = st.session_state.df_original.copy()
                st.toast("DataFrame has been reset to its original state!", icon="✅")
                # No rerun needed, will happen on next interaction

        page_options = ["Home"]
        if st.session_state.file_uploaded:
            page_options = [
                "🏠 Home",
                "📊 Data Overview",
                "❓ Missing Values",
                "🏛️ Column Management",
                "📑 Handle Duplicates",
                "📈 Outlier Handling",
                "🔡 Text Cleaning",
                "🔍 Data Filtering",
                "📥 Download"
            ]

        # Use a mapping to keep radio options clean but pages identifiable
        page_mapping = {
            "🏠 Home": "Home",
            "📊 Data Overview": "Data Overview",
            "❓ Missing Values": "Missing Values",
            "🏛️ Column Management": "Column Management",
            "📑 Handle Duplicates": "Handle Duplicates",
            "📈 Outlier Handling": "Outlier Handling",
            "🔡 Text Cleaning": "Text Cleaning",
            "🔍 Data Filtering": "Data Filtering",
            "📥 Download": "Download"
        }
        
        # Determine the default index for the radio button
        current_page_display = [k for k, v in page_mapping.items() if v == st.session_state.get('active_page', 'Home')]
        default_index = page_options.index(current_page_display[0]) if current_page_display else 0

        selected_page_display = st.radio(
            "Go to:",
            options=page_options,
            index=default_index,
            key='navigation_radio'
        )
        st.session_state.active_page = page_mapping.get(selected_page_display, "Home")

        st.markdown("---")
        st.info("Created with ❤️ by an AI Assistant.")
        
        # New file upload button in sidebar
        if st.session_state.file_uploaded:
             if st.button("⬆️ Upload a New File"):
                reset_app_state()
                st.rerun()


    # --- Page Routing ---
    active_page = st.session_state.active_page

    if not st.session_state.file_uploaded:
        render_home_page()
    else:
        if active_page == "Home":
            render_home_page()
        elif active_page == "Data Overview":
            render_overview_page()
        elif active_page == "Missing Values":
            render_missing_values_page()
        elif active_page == "Column Management":
            render_column_management_page()
        elif active_page == "Handle Duplicates":
            render_duplicates_page()
        elif active_page == "Outlier Handling":
            render_outlier_page()
        elif active_page == "Text Cleaning":
            render_text_cleaning_page()
        elif active_page == "Data Filtering":
            render_filtering_page()
        elif active_page == "Download":
            render_download_page()
        else:
            render_home_page() # Default fallback


if __name__ == "__main__":
    main()
