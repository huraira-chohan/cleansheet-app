import base64
import io
import streamlit as st
import numpy as np
import pandas as pd
import dateparser
from typing import Callable, Dict, List
from word2number import w2n
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            mean_squared_error, r2_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error)
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                             RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge, Lasso)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import zscore

# Application Configuration
APP_TITLE = "CSV Data Cleaner"
APP_ICON = "🧹"

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

# State Management
def initialize_session_state():
    defaults = {
        'df': None,
        'df_original': None,
        'file_uploader_key': 0,
        'file_uploaded': False,
        'active_page': list(PAGES.keys())[0],
        'file_name': None,
        'history': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_app_state():
    st.session_state.df = None
    st.session_state.df_original = None
    st.session_state.file_uploaded = False
    st.session_state.file_uploader_key += 1
    st.session_state.active_page = list(PAGES.keys())[0]
    st.session_state.history = []
    st.session_state.file_name = None
    st.toast("Application reset. Please upload a new file.", icon="🔄")
    st.rerun()

def log_action(description: str, code_snippet: str = None):
    st.session_state.history.append({"description": description, "code": code_snippet})
    st.toast(f"Action: {description}", icon="✅")

# Utility Functions
@st.cache_data
def get_dataframe_info(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer, verbose=True)
    return buffer.getvalue()

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=np.number).columns.tolist() if df is not None else []

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist() if df is not None else []

def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist() if df is not None else []

# Page Rendering Functions
def render_home_page():
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("""
    Welcome to the CSV Data Cleaner! This tool helps you clean, preprocess, and prepare CSV data for analysis or machine learning.

    **How to use:**
    1. Upload your CSV file below.
    2. Navigate through cleaning modules using the sidebar.
    3. Track changes in the 'Action History' page.
    4. Download your cleaned data from the 'Download & Export' page.
    """)

    st.subheader("Upload CSV File")
    with st.expander("Upload Options", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            key=f"uploader_{st.session_state.file_uploader_key}",
            help="Upload a CSV file to start cleaning."
        )
        st.markdown("---")
        st.subheader("Advanced Options")
        col1, col2 = st.columns(2)
        with col1:
            separator = st.text_input("Column Separator", value=",", help="Common separators: ',' or ';'")
        with col2:
            encoding = st.selectbox("File Encoding", ["utf-8", "latin1", "iso-8859-1", "cp1252"], help="Try different encodings if upload fails.")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=separator, encoding=encoding)
            st.session_state.df_original = df.copy()
            st.session_state.df = df.copy()
            st.session_state.file_uploaded = True
            st.session_state.file_name = uploaded_file.name
            log_action(f"Uploaded '{uploaded_file.name}'. Shape: {df.shape}", f"pd.read_csv(file, sep='{separator}', encoding='{encoding}')")
            st.success("File uploaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}. Check separator, encoding, or file format.")
            st.session_state.file_uploaded = False

    if st.session_state.file_uploaded:
        st.subheader("Quick Inspection")
        st.info(f"File: **{st.session_state.file_name}** | Shape: **{st.session_state.df.shape}**")
        st.dataframe(st.session_state.df.head())

def render_profiling_page():
    st.header("📊 Data Profiling & Overview")
    if st.session_state.df is None:
        st.warning("Please upload a file on the Home page.")
        return

    df = st.session_state.df
    tab1, tab2, tab3, tab4 = st.tabs(["DataFrame Info", "Statistical Summary", "Value Counts", "Column Correlations"])

    with tab1:
        st.subheader("DataFrame Structure")
        st.text(get_dataframe_info(df))

    with tab2:
        st.subheader("Descriptive Statistics")
        numeric_cols = get_numeric_columns(df)
        if numeric_cols:
            st.dataframe(df.describe(include=np.number))
        else:
            st.info("No numeric columns found.")
        categorical_cols = get_categorical_columns(df)
        if categorical_cols:
            st.dataframe(df.describe(include=['object', 'category']))
        else:
            st.info("No categorical columns found.")

    with tab3:
        st.subheader("Value Counts")
        categorical_cols = get_categorical_columns(df)
        if not categorical_cols:
            st.info("No categorical columns found.")
        else:
            selected_col = st.selectbox("Select column:", categorical_cols, help="View value distribution.")
            if selected_col:
                value_counts_df = df[selected_col].value_counts().reset_index()
                value_counts_df.columns = [selected_col, 'Count']
                st.dataframe(value_counts_df)
                if st.checkbox("Show bar chart?"):
                    fig = px.bar(value_counts_df, x=selected_col, y='Count', title=f"Value Counts for {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Correlation Analysis")
        numeric_cols = get_numeric_columns(df)
        if len(numeric_cols) < 2:
            st.info("Need at least two numeric columns for correlation.")
        else:
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

def render_missing_values_page():
    st.header("❓ Missing Value Manager")
    if st.session_state.df is None:
        st.warning("Please upload a file.")
        return

    df = st.session_state.df
    missing_data = df.isnull().sum()
    missing_data_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage (%)': missing_data_percent})
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)

    if missing_df.empty:
        st.success("No missing values found!")
        return

    st.subheader("Missing Value Analysis")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(missing_df)
    with col2:
        st.markdown("#### Missing Values Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
        st.pyplot(fig)

    st.subheader("Handle Missing Values")
    with st.expander("Drop Missing Values"):
        drop_choice = st.radio("Drop method:", ["Rows with any NaNs", "Columns with any NaNs", "Columns by threshold"])
        if drop_choice == "Columns by threshold":
            threshold = st.slider("Threshold (%)", 0, 100, 50)
            if st.button("Drop Columns"):
                cols_to_drops = missing_df[missing_df['Percentage (%)'] > threshold].index.tolist()
                if cols_to_drops:
                    st.session_state.df.drop(columns=cols_to_drops, inplace=True)
                    log_action(f"Dropped columns with >{threshold}% missing: {', '.join(cols_to_drops)}", 
                              f"df.drop(columns={cols_to_drops}, inplace=True)")
                    st.rerun()
                else:
                    st.warning("No columns meet the threshold.")
        elif st.button("Apply Drop"):
            if drop_choice == "Rows with any NaNs":
                st.session_state.df.dropna(axis=0, inplace=True)
                log_action("Dropped rows with missing values", "df.dropna(axis=0, inplace=True)")
            else:
                st.session_state.df.dropna(axis=1, inplace=True)
                log_action("Dropped columns with missing values", "df.dropna(axis=1, inplace=True)")
            st.rerun()

    with st.expander("Impute Missing Values"):
        impute_cols = missing_df.index.tolist()
        if impute_cols:
            selected_col = st.selectbox("Select column:", impute_cols)
            col_type = df[selected_col].dtype
            impute_methods = ['Mode', 'Custom Value'] if not pd.api.types.is_numeric_dtype(col_type) else \
                             ['Mean', 'Median', 'Mode', 'Interpolate', 'Custom Value']
            imputation_method = st.selectbox("Imputation method:", impute_methods)
            custom_value = st.text_input("Custom value:", "") if imputation_method == 'Custom Value' else None
            if st.button("Apply Imputation"):
                try:
                    if imputation_method == 'Mean':
                        fill_value = df[selected_col].mean()
                        st.session_state.df[selected_col].fillna(fill_value, inplace=True)
                        log_action(f"Imputed '{selected_col}' with mean: {fill_value}", 
                                  f"df['{selected_col}'].fillna(df['{selected_col}'].mean(), inplace=True)")
                    elif imputation_method == 'Median':
                        fill_value = df[selected_col].median()
                        st.session_state.df[selected_col].fillna(fill_value, inplace=True)
                        log_action(f"Imputed '{selected_col}' with median: {fill_value}", 
                                  f"df['{selected_col}'].fillna(df['{selected_col}'].median(), inplace=True)")
                    elif imputation_method == 'Mode':
                        fill_value = df[selected_col].mode()[0]
                        st.session_state.df[selected_col].fillna(fill_value, inplace=True)
                        log_action(f"Imputed '{selected_col}' with mode: {fill_value}", 
                                  f"df['{selected_col}'].fillna(df['{selected_col}'].mode()[0], inplace=True)")
                    elif imputation_method == 'Interpolate':
                        st.session_state.df[selected_col].interpolate(method='linear', inplace=True)
                        log_action(f"Interpolated '{selected_col}'", 
                                  f"df['{selected_col}'].interpolate(method='linear', inplace=True)")
                    elif imputation_method == 'Custom Value':
                        if custom_value:
                            st.session_state.df[selected_col].fillna(custom_value, inplace=True)
                            log_action(f"Imputed '{selected_col}' with custom value: {custom_value}", 
                                      f"df['{selected_col}'].fillna('{custom_value}', inplace=True)")
                        else:
                            st.error("Please provide a custom value.")
                            return
                    st.rerun()
                except Exception as e:
                    st.error(f"Imputation failed: {e}")

def render_column_management_page():
    st.header("🏛️ Column Operations")
    if st.session_state.df is None:
        st.warning("Please upload a file.")
        return

    df = st.session_state.df
    all_cols = df.columns.tolist()
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📅 Convert to Datetime", "📊 Analyze Types", "✂️ Split Column", 
        "🗑️ Drop Columns", "✏️ Rename Column", "🔄 Change Type"
    ])

    with tab1:
        st.subheader("Convert to Datetime")
        candidate_cols = get_categorical_columns(df)
        if not candidate_cols:
            st.info("No text columns found.")
        else:
            selected_col = st.selectbox("Select column:", candidate_cols, key="date_convert")
            parser_engine = st.radio("Parser:", ["Pandas (Fast)", "Dateparser (Robust)"], index=1)
            dayfirst_param = st.checkbox("Day first (DD/MM/YYYY)", help="For ambiguous dates.")
            st.markdown("#### Preview")
            try:
                def parse_with_dateparser(date_string):
                    if pd.isna(date_string): return pd.NaT
                    return dateparser.parse(str(date_string), settings={'PREFER_DAY_OF_MONTH': 'first' if dayfirst_param else 'last'})
                
                preview_series = pd.to_datetime(df[selected_col].astype(str).str.strip(), dayfirst=dayfirst_param, errors='coerce') if "Pandas" in parser_engine else \
                                df[selected_col].apply(parse_with_dateparser)
                preview_df = pd.DataFrame({
                    f"Original '{selected_col}'": df[selected_col].head(20),
                    "Parsed Datetime": preview_series.head(20)
                })
                st.dataframe(preview_df.dropna(subset=[f"Original '{selected_col}'"]), use_container_width=True)
                failed_parses = preview_series.isna().sum() - df[selected_col].isna().sum()
                if failed_parses > 0:
                    st.warning(f"{failed_parses} values could not be parsed (set to NaT).")
                if st.button("Apply Conversion", type="primary"):
                    st.session_state.df[selected_col] = preview_series
                    log_action(f"Converted '{selected_col}' to datetime using {parser_engine}", 
                              f"pd.to_datetime(df['{selected_col}'], dayfirst={dayfirst_param}, errors='coerce')" if "Pandas" in parser_engine else 
                              f"df['{selected_col}'].apply(dateparser.parse, settings={{'PREFER_DAY_OF_MONTH': '{'first' if dayfirst_param else 'last'}'}})")
                    st.rerun()
            except Exception as e:
                st.error(f"Preview failed: {e}")

    with tab2:
        st.subheader("Analyze Column Types")
        numeric_cols, categorical_cols, datetime_cols = get_numeric_columns(df), get_categorical_columns(df), get_datetime_columns(df)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Numeric Columns")
            st.dataframe(pd.DataFrame(numeric_cols, columns=["Column Name"]))
            st.markdown("#### Datetime Columns")
            st.dataframe(pd.DataFrame(datetime_cols, columns=["Column Name"]))
        with col2:
            st.markdown("#### Categorical Columns")
            st.dataframe(pd.DataFrame(categorical_cols, columns=["Column Name"]))

    with tab3:
        st.subheader("Split Column")
        candidate_cols = get_categorical_columns(df)
        if not candidate_cols:
            st.info("No text columns found.")
        else:
            with st.form("split_form"):
                source_col = st.selectbox("Select column:", candidate_cols)
                col1, col2 = st.columns(2)
                with col1: new_cat_col = st.text_input("Text Column Name", value=f"{source_col}_cat")
                with col2: new_num_col = st.text_input("Number Column Name", value=f"{source_col}_num")
                drop_original = st.checkbox("Drop original column?", value=True)
                if st.form_submit_button("Apply Split"):
                    if new_cat_col and new_num_col:
                        st.session_state.df[new_cat_col] = df[source_col].str.extract(r'([^\d]*)', expand=False).str.strip()
                        st.session_state.df[new_num_col] = pd.to_numeric(df[source_col].str.extract(r'(\d+)', expand=False), errors='coerce')
                        log_desc = f"Split '{source_col}' into '{new_cat_col}' and '{new_num_col}'"
                        if drop_original:
                            st.session_state.df.drop(columns=[source_col], inplace=True)
                            log_desc += "; dropped original"
                        log_action(log_desc, f"df['{new_cat_col}'] = df['{source_col}'].str.extract(r'([^\d]*)').str.strip(); df['{new_num_col}'] = pd.to_numeric(df['{source_col}'].str.extract(r'(\d+)'), errors='coerce')")
                        st.rerun()
                    else:
                        st.error("Please provide valid column names.")

    with tab4:
        st.subheader("Drop Columns")
        cols_to_drop = st.multiselect("Select columns:", all_cols)
        if st.button("Drop Columns"):
            if cols_to_drop:
                st.session_state.df.drop(columns=cols_to_drop, inplace=True)
                log_action(f"Dropped columns: {', '.join(cols_to_drop)}", f"df.drop(columns={cols_to_drop}, inplace=True)")
                st.rerun()
            else:
                st.warning("Select at least one column.")

    with tab5:
        st.subheader("Rename Column")
        col_to_rename = st.selectbox("Select column:", all_cols, key="rename_col")
        new_name = st.text_input("New name:", value=col_to_rename)
        if st.button("Rename"):
            if new_name:
                st.session_state.df.rename(columns={col_to_rename: new_name}, inplace=True)
                log_action(f"Renamed '{col_to_rename}' to '{new_name}'", f"df.rename(columns={{'{col_to_rename}': '{new_name}'}}, inplace=True)")
                st.rerun()
            else:
                st.error("Enter a valid column name.")

    with tab6:
        st.subheader("Change Column Type")
        col_to_change = st.selectbox("Select column:", all_cols, key="type_change")
        new_type = st.selectbox("New type:", ['object', 'int64', 'float64', 'category', 'bool'])
        if st.button("Apply Type Change"):
            try:
                st.session_state.df[col_to_change] = st.session_state.df[col_to_change].astype(new_type)
                log_action(f"Changed '{col_to_change}' to {new_type}", f"df['{col_to_change}'].astype('{new_type}')")
                st.rerun()
            except Exception as e:
                st.error(f"Type conversion failed: {e}")

def render_duplicate_handling_page():
    st.header("📑 Row & Duplicate Manager")
    if st.session_state.df is None:
        st.warning("Please upload a file.")
        return

    df = st.session_state.df
    tab1, tab2 = st.tabs(["Duplicate Rows", "Filter Rows"])

    with tab1:
        st.subheader("Handle Duplicates")
        duplicates = df[df.duplicated(keep=False)]
        num_duplicates = df.duplicated().sum()
        if duplicates.empty:
            st.success("No duplicate rows found.")
        else:
            st.warning(f"Found {num_duplicates} duplicate rows.")
            st.dataframe(duplicates.sort_values(by=df.columns.tolist()))
            if st.button("Remove Duplicates (keep first)", type="primary"):
                st.session_state.df.drop_duplicates(keep='first', inplace=True)
                log_action(f"Removed {num_duplicates} duplicates", "df.drop_duplicates(keep='first', inplace=True)")
                st.rerun()

    with tab2:
        st.subheader("Filter Rows")
        st.info("Use pandas query syntax, e.g., `Age > 30` or `Country == 'USA'`.")
        query_string = st.text_area("Enter query:", height=100)
        if st.button("Apply Filter"):
            if query_string:
                try:
                    filtered_df = df.query(query_string)
                    rows_removed = df.shape[0] - filtered_df.shape[0]
                    st.session_state.df = filtered_df
                    log_action(f"Filtered with '{query_string}'; removed {rows_removed} rows", f"df.query('{query_string}')")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid query: {e}")
            else:
                st.warning("Enter a query string.")

def render_outlier_page():
    st.header("📈 Outlier Detection & Handling")
    if st.session_state.df is None:
        st.warning("Please upload a file.")
        return

    df = st.session_state.df
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        st.info("No numeric columns found.")
        return

    col1, col2 = st.columns(2)
    with col1: selected_col = st.selectbox("Select column:", numeric_cols)
    with col2: method = st.selectbox("Method:", ["IQR", "Z-Score"])

    st.subheader("Visualize Distribution")
    fig = px.box(df, y=selected_col, title=f"Box Plot for {selected_col}", points="all")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detect & Remove Outliers")
    if method == "IQR":
        multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
        st.warning(f"Detected {len(outliers)} outliers in '{selected_col}'.")
        if not outliers.empty and st.checkbox("Show outliers?"):
            st.dataframe(outliers)
        if st.button("Remove Outliers", type="primary"):
            st.session_state.df = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)]
            log_action(f"Removed {len(outliers)} outliers from '{selected_col}' (IQR, multiplier={multiplier})", 
                      f"df = df[(df['{selected_col}'] >= {lower_bound}) & (df['{selected_col}'] <= {upper_bound})]")
            st.rerun()
    else:
        threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
        z_scores = zscore(df[selected_col].dropna())
        outlier_indices = df[selected_col].dropna()[np.abs(z_scores) > threshold].index
        outliers = df.loc[outlier_indices]
        st.warning(f"Detected {len(outliers)} outliers in '{selected_col}'.")
        if not outliers.empty and st.checkbox("Show outliers?"):
            st.dataframe(outliers)
        if st.button("Remove Outliers", type="primary"):
            st.session_state.df = df.drop(outlier_indices)
            log_action(f"Removed {len(outliers)} outliers from '{selected_col}' (Z-Score, threshold={threshold})", 
                      f"df = df.drop(df[np.abs(zscore(df['{selected_col}'].dropna())) > {threshold}].index)")
            st.rerun()

def render_transformation_page():
    st.header("🔬 Data Transformation")
    if st.session_state.df is None:
        st.warning("Please upload a file.")
        return

    df = st.session_state.df
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Text-to-Number", "Find & Replace", "Normalize Categories", 
        "Text Cleaning", "Numeric Scaling", "Datetime Features"
    ])

    with tab1:
        st.subheader("Text-to-Number")
        def convert_word_to_number(value):
            try: return w2n.word_to_num(str(value))
            except ValueError: return value
        candidate_cols = get_categorical_columns(df) + get_numeric_columns(df)
        if not candidate_cols:
            st.info("No suitable columns found.")
        else:
            selected_col = st.selectbox("Select column:", candidate_cols, key="w2n")
            temp_series = df[selected_col].apply(convert_word_to_number)
            temp_series_numeric = pd.to_numeric(temp_series, errors='coerce')
            st.dataframe(pd.DataFrame({"Original": df[selected_col].head(20), "Converted": temp_series_numeric.head(20)}))
            if st.button("Apply Conversion", type="primary"):
                st.session_state.df[selected_col] = temp_series_numeric
                log_action(f"Converted text to numbers in '{selected_col}'", 
                          f"df['{selected_col}'] = df['{selected_col}'].apply(w2n.word_to_num)")
                st.rerun()

    with tab2:
        st.subheader("Find & Replace")
        categorical_cols = get_categorical_columns(df)
        if not categorical_cols:
            st.info("No text columns found.")
        else:
            with st.form("find_replace"):
                selected_col = st.selectbox("Select column:", categorical_cols, key="fr")
                match_case = st.checkbox("Match Case")
                rules_df = pd.DataFrame([{"Value to Find": "", "Replace With": ""}])
                edited_rules = st.data_editor(rules_df, num_rows="dynamic", key="fr_editor")
                if st.form_submit_button("Apply"):
                    valid_rules = edited_rules.dropna(subset=["Value to Find"]).loc[edited_rules["Value to Find"] != ""]
                    if valid_rules.empty:
                        st.warning("No valid rules defined.")
                    else:
                        temp_col = df[selected_col].astype(str)
                        if match_case:
                            replace_dict = dict(zip(valid_rules["Value to Find"], valid_rules["Replace With"]))
                            temp_col.replace(replace_dict, inplace=True)
                        else:
                            for _, rule in valid_rules.iterrows():
                                temp_col = temp_col.str.replace(f'^{rule["Value to Find"]}$', rule["Replace With"], case=False, regex=True)
                        st.session_state.df[selected_col] = temp_col
                        log_action(f"Applied find/replace in '{selected_col}'", 
                                  f"df['{selected_col}'].replace({replace_dict})" if match_case else 
                                  f"df['{selected_col}'].str.replace(...case=False)")
                        st.rerun()

    with tab3:
        st.subheader("Normalize Categories")
        categorical_cols = get_categorical_columns(df)
        if not categorical_cols:
            st.info("No categorical columns found.")
        else:
            selected_col = st.selectbox("Select column:", categorical_cols, key="norm")
            with st.form("norm_form"):
                unique_values = df[selected_col].dropna().unique()
                mapping_df = pd.DataFrame({"Original": unique_values, "New": unique_values})
                edited_mapping = st.data_editor(mapping_df, key=f"editor_{selected_col}")
                if st.form_submit_button("Apply"):
                    mapping_dict = dict(zip(edited_mapping["Original"], edited_mapping["New"]))
                    st.session_state.df[selected_col] = df[selected_col].replace(mapping_dict)
                    log_action(f"Normalized '{selected_col}'", f"df['{selected_col}'].replace({mapping_dict})")
                    st.rerun()

    with tab4:
        st.subheader("Text Cleaning")
        text_cols = get_categorical_columns(df)
        if not text_cols:
            st.info("No text columns found.")
        else:
            selected_col = st.selectbox("Select column:", text_cols, key="clean")
            with st.form("clean_form"):
                to_lowercase = st.checkbox("Convert to lowercase")
                strip_whitespace = st.checkbox("Strip whitespace")
                remove_punctuation = st.checkbox("Remove punctuation")
                if st.form_submit_button("Apply"):
                    cleaned_series = df[selected_col].astype(str)
                    log_items = []
                    if to_lowercase:
                        cleaned_series = cleaned_series.str.lower()
                        log_items.append("lowercase")
                    if strip_whitespace:
                        cleaned_series = cleaned_series.str.strip()
                        log_items.append("strip whitespace")
                    if remove_punctuation:
                        cleaned_series = cleaned_series.str.replace(r'[^\w\s]', '', regex=True)
                        log_items.append("remove punctuation")
                    st.session_state.df[selected_col] = cleaned_series
                    if log_items:
                        log_action(f"Cleaned '{selected_col}' ({', '.join(log_items)})", 
                                  f"df['{selected_col}'].str.lower/strip/replace...")
                        st.rerun()

    with tab5:
        st.subheader("Scale Numeric Columns")
        numeric_cols = get_numeric_columns(df)
        if not numeric_cols:
            st.info("No numeric columns found.")
        else:
            scaler_type = st.radio("Scaler:", ["Min-Max", "Standard"])
            cols_to_scale = st.multiselect("Select columns:", numeric_cols)
            if st.button("Apply Scaling"):
                if cols_to_scale:
                    scaler = MinMaxScaler() if scaler_type == "Min-Max" else StandardScaler()
                    st.session_state.df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                    log_action(f"Scaled {', '.join(cols_to_scale)} with {scaler.__class__.__name__}", 
                              f"scaler.fit_transform(df[{cols_to_scale}])")
                    st.rerun()
                else:
                    st.warning("Select at least one column.")

    with tab6:
        st.subheader("Extract Datetime Features")
        datetime_cols = get_datetime_columns(df)
        if not datetime_cols:
            st.info("No datetime columns found.")
        else:
            selected_col = st.selectbox("Select column:", datetime_cols, key="dt")
            features = st.multiselect("Features:", ["Year", "Month", "Day", "Day of Week", "Hour"])
            if st.button("Extract Features"):
                if features:
                    for feature in features:
                        new_col = f"{selected_col}_{feature.lower().replace(' ', '_')}"
                        if feature == "Year": st.session_state.df[new_col] = df[selected_col].dt.year
                        elif feature == "Month": st.session_state.df[new_col] = df[selected_col].dt.month
                        elif feature == "Day": st.session_state.df[new_col] = df[selected_col].dt.day
                        elif feature == "Day of Week": st.session_state.df[new_col] = df[selected_col].dt.dayofweek
                        elif feature == "Hour": st.session_state.df[new_col] = df[selected_col].dt.hour
                    log_action(f"Extracted features from '{selected_col}'", 
                              f"df['{new_col}'] = df['{selected_col}'].dt.{feature.lower()}")
                    st.rerun()

def render_ml_modeler_page():
    st.header("🤖 ML Modeler")
    if st.session_state.df is None:
        st.warning("Please upload and prepare data.")
        return

    df = st.session_state.df.copy()
    with st.sidebar:
        st.header("Model Configuration")
        target = st.selectbox("Target Column:", df.columns)
        df.dropna(subset=[target], inplace=True)
        if df.empty:
            st.error("No valid data after removing missing targets.")
            return
        problem_type = "Regression" if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 25 else "Classification"
        st.info(f"Problem: {problem_type}")
        
        model_options = ["--Select Algorithm--"] + (
            ["Logistic Regression", "Random Forest Classifier", "Gradient Boosting", "XGBoost Classifier", 
             "LightGBM Classifier", "SVC", "KNeighbors Classifier", "DecisionTreeClassifier"] if problem_type == "Classification" else
            ["Linear Regression", "Ridge", "Lasso", "Random Forest Regressor", "Gradient Boosting Regressor", 
             "XGBoost Regressor", "LightGBM Regressor", "SVR"]
        )
        selected_model = st.selectbox("Algorithm:", model_options)
        
        params = {}
        if selected_model != "--Select Algorithm--":
            st.header("Hyperparameters")
            if selected_model == "Logistic Regression":
                params['C'] = st.slider("C", 0.01, 10.0, 1.0, 0.01)
                params['solver'] = st.selectbox("Solver", ['liblinear', 'lbfgs', 'saga'])
                params['max_iter'] = st.slider("Max Iterations", 100, 1000, 100, 50)
            elif "Random Forest" in selected_model:
                params['n_estimators'] = st.slider("Trees", 10, 500, 100, 10)
                params['max_depth'] = st.slider("Max Depth", 2, 50, 10, 1)
                params['min_samples_leaf'] = st.slider("Min Samples Leaf", 1, 20, 1, 1)
            elif "Gradient Boosting" in selected_model:
                params['n_estimators'] = st.slider("Estimators", 10, 500, 100, 10)
                params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
                params['max_depth'] = st.slider("Max Depth", 2, 15, 3, 1)
            elif "XGBoost" in selected_model:
                params['n_estimators'] = st.slider("Estimators", 10, 500, 100, 10)
                params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
                params['max_depth'] = st.slider("Max Depth", 2, 15, 3, 1)
                params['subsample'] = st.slider("Subsample", 0.5, 1.0, 1.0, 0.1)
            elif selected_model == "SVC":
                params['C'] = st.slider("C", 0.01, 10.0, 1.0)
                params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
                params['probability'] = True
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random Seed", value=42)

    if selected_model == "--Select Algorithm--":
        st.info("Select an algorithm to proceed.")
        return

    if st.button(f"Train {selected_model}", type="primary"):
        with st.spinner("Training model..."):
            X = df.drop(columns=[target])
            y = df[target]
            numeric_features = get_numeric_columns(X)
            categorical_features = get_categorical_columns(X)
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
                    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
                ])
            model_class_map = {
                "Logistic Regression": LogisticRegression, "Random Forest Classifier": RandomForestClassifier, 
                "Gradient Boosting": GradientBoostingClassifier, "XGBoost Classifier": xgb.XGBClassifier, 
                "LightGBM Classifier": lgb.LGBMClassifier, "SVC": SVC, "KNeighbors Classifier": KNeighborsClassifier, 
                "DecisionTreeClassifier": DecisionTreeClassifier, "Linear Regression": LinearRegression, 
                "Ridge": Ridge, "Lasso": Lasso, "Random Forest Regressor": RandomForestRegressor, 
                "Gradient Boosting Regressor": GradientBoostingRegressor, "XGBoost Regressor": xgb.XGBRegressor, 
                "LightGBM Regressor": lgb.LGBMRegressor, "SVR": SVR
            }
            model = model_class_map[selected_model](**params)
            if 'random_state' in model.get_params():
                model.set_params(random_state=random_state)
            if "XGBoost" in selected_model:
                model.set_params(eval_metric='logloss' if problem_type == "Classification" else 'rmse')
            pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=(y if problem_type == "Classification" else None))
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

        st.success("Model trained!")
        tab1, tab2 = st.tabs(["Metrics", "Visualizations"])
        with tab1:
            if problem_type == "Classification":
                y_proba = pipeline.predict_proba(X_test)
                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    "MCC": matthews_corrcoef(y_test, y_pred),
                    "AUC": roc_auc_score(y_test, y_proba, multi_class='ovr') if len(y.unique()) > 2 else roc_auc_score(y_test, y_proba[:, 1])
                }
                col1, col2, col3 = st.columns(3)
                for i, (k, v) in enumerate(metrics.items()):
                    (col1 if i < 2 else col2 if i < 4 else col3).metric(k, f"{v:.3f}")
                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose())
            else:
                metrics = {
                    "R²": r2_score(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "MSE": mean_squared_error(y_test, y_pred)
                }
                col1, col2 = st.columns(2)
                for i, (k, v) in enumerate(metrics.items()):
                    (col1 if i < 2 else col2).metric(k, f"{v:.3f}")

        with tab2:
            if problem_type == "Classification":
                if hasattr(pipeline, "predict_proba"):
                    y_proba = pipeline.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC = {auc(fpr, tpr):.2f})', 
                                     labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig_roc, use_container_width=True)
                fig_cm, ax_cm = plt.subplots()
                cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                           xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
                st.pyplot(fig_cm)
            else:
                fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, 
                                    title="Actual vs. Predicted")
                fig_pred.add_shape(type='line', line=dict(dash='dash'), x0=y_test.min(), y0=y_test.min(), 
                                  x1=y_test.max(), y1=y_test.max())
                st.plotly_chart(fig_pred, use_container_width=True)
                residuals = y_test - y_pred
                fig_res = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'}, 
                                   title="Residuals vs. Predicted")
                fig_res.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig_res, use_container_width=True)
            model = pipeline.named_steps['model']
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_[0]
                feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(importances)}).sort_values(by='Importance', ascending=False).head(20)
                fig_imp = px.bar(feature_df, x='Importance', y='Feature', orientation='h', title="Top 20 Features")
                st.plotly_chart(fig_imp, use_container_width=True)

def render_history_page():
    st.header("📜 Action History")
    if not st.session_state.history:
        st.info("No actions recorded yet.")
    else:
        for idx, action in enumerate(st.session_state.history):
            st.markdown(f"**Action {idx + 1}:** {action['description']}")
            if action['code']:
                st.code(action['code'], language='python')

def render_download_page():
    st.header("📥 Download & Export")
    if st.session_state.df is None:
        st.warning("No data to download.")
        return

    df = st.session_state.df
    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.info(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    csv = df.to_csv(index=False).encode('utf-8')
    filename = f"cleaned_{st.session_state.file_name or 'data.csv'}"
    st.download_button("Download CSV", csv, filename, "text/csv", type="primary")

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")
    initialize_session_state()
    with st.sidebar:
        st.header("⚙️ Workflow")
        available_pages = list(PAGES.keys()) if st.session_state.file_uploaded else [list(PAGES.keys())[0]]
        if st.session_state.active_page not in available_pages:
            st.session_state.active_page = available_pages[0]
        st.session_state.active_page = st.radio("Go to:", available_pages, index=available_pages.index(st.session_state.active_page))
        if st.session_state.file_uploaded:
            if st.button("Reset Changes"):
                st.session_state.df = st.session_state.df_original.copy()
                st.session_state.history = []
                log_action("Reset to original file")
                st.rerun()
            if st.button("Upload New File"):
                reset_app_state()
        st.markdown("---")
        st.info("Created by Chohan.")

    page_function = globals().get(PAGES.get(st.session_state.active_page, "render_home_page"))
    page_function()

if __name__ == "__main__":
    main()
