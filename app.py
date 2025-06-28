import streamlit as st
import pandas as pd
import numpy as np
import re
from dateutil import parser
from urllib.parse import urlparse
from scipy import stats
import requests
from io import StringIO

# Setup
st.set_page_config("🧹 CleanSheet v10", layout="wide")
st.title("🧹 CleanSheet v10 - Smartest CSV Data Cleaner")

NULL_VALUES = ["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"]

# Cleaning helpers
def clean_text(x): return str(x).strip().title() if pd.notnull(x) else x
def convert_to_numeric(x): 
    try: return float(re.sub(r"[^0-9.]+", "", str(x)))
    except: return np.nan
def parse_any_date(x): 
    try: return parser.parse(str(x), fuzzy=True)
    except: return np.nan
def normalize_gender(x): 
    x = str(x).strip().lower()
    if x in ["m", "male"]: return "Male"
    if x in ["f", "female"]: return "Female"
    return "Other"
def is_valid_email(x): 
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$', str(x)))
def is_valid_url(x): 
    try: r = urlparse(x); return all([r.scheme, r.netloc])
    except: return False
def remove_outliers(series, threshold=3):
    if pd.api.types.is_numeric_dtype(series):
        z = np.abs(stats.zscore(series.dropna()))
        return series.where(z < threshold)
    return series

# Load dataset
st.sidebar.header("📦 Load CSV")
use_sample = st.sidebar.button("Use Titanic Sample")
uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

# Load data once into session_state
if "df" not in st.session_state:
    st.session_state.df = None

if use_sample:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    sample_data = StringIO(requests.get(url).text)
    st.session_state.df = pd.read_csv(sample_data)
    st.session_state.df.replace(NULL_VALUES, np.nan, inplace=True)

df = st.session_state.df


if uploaded:
    df = pd.read_csv(uploaded)
    df.replace(NULL_VALUES, np.nan, inplace=True)

    st.subheader("📊 Data Preview & Filtering")
    
    view_option = st.radio("View Rows", ["Top 5", "Top 50", "All"], horizontal=True)
    if view_option == "Top 5": st.dataframe(df.head())
    elif view_option == "Top 50": st.dataframe(df.head(50))
    else: st.dataframe(df)

    st.markdown("---")
    st.subheader("🔎 Filter Rows")
    col = st.selectbox("Select Column to Filter", df.columns)
    val = st.text_input("Filter by value (exact match)")
    if val:
        df = df[df[col].astype(str) == val]
        st.success(f"Filtered {len(df)} rows.")

    st.markdown("---")
    st.subheader("🚮 Drop Duplicate Rows")
    if st.checkbox("Drop duplicates?"):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        st.success(f"Removed {before - after} duplicate rows.")

    st.markdown("---")
    st.subheader("🪄 Column Cleaning Options")
    config = {}
    with st.form("cleaning_form"):
        for col in df.columns:
            with st.expander(f"⚙️ {col}"):
                clean = st.selectbox(f"Cleaning method for {col}", 
                    ["none", "text_normalize", "numeric", "date", "gender", "email_validate", "url_validate", "drop"], key=f"clean_{col}")
                fill = st.selectbox("Handle missing values", 
                    ["none", "drop_rows", "fill_mean", "fill_median", "fill_mode"], key=f"null_{col}")
                outliers = st.checkbox("Remove outliers (Z-score)", key=f"outlier_{col}") if pd.api.types.is_numeric_dtype(df[col]) else False
                config[col] = (clean, fill, outliers)
        st.form_submit_button("🧼 Apply Cleaning")

    # Apply cleaning
    for col, (clean, fill, outliers) in config.items():
        if clean == "drop":
            df.drop(columns=[col], inplace=True)
            continue
        if clean == "text_normalize": df[col] = df[col].apply(clean_text)
        elif clean == "numeric": df[col] = df[col].apply(convert_to_numeric)
        elif clean == "date": df[col] = df[col].apply(parse_any_date)
        elif clean == "gender": df[col] = df[col].apply(normalize_gender)
        elif clean == "email_validate": df[f"{col}_valid"] = df[col].apply(is_valid_email)
        elif clean == "url_validate": df[f"{col}_valid"] = df[col].apply(is_valid_url)

        if fill == "drop_rows": df = df[df[col].notna()]
        elif fill == "fill_mean" and pd.api.types.is_numeric_dtype(df[col]): df[col].fillna(df[col].mean(), inplace=True)
        elif fill == "fill_median" and pd.api.types.is_numeric_dtype(df[col]): df[col].fillna(df[col].median(), inplace=True)
        elif fill == "fill_mode": 
            mode = df[col].mode().iloc[0] if not df[col].mode().empty else None
            df[col].fillna(mode, inplace=True)

        if outliers: df[col] = remove_outliers(df[col])

    st.success("✅ Cleaning Complete!")

    st.markdown("---")
    st.subheader("🔁 Column Merging")
    merge_cols = st.multiselect("Select columns to merge", df.columns)
    merge_name = st.text_input("New column name after merge")
    if merge_cols and merge_name and st.button("Merge Columns"):
        df[merge_name] = df[merge_cols].astype(str).agg(" ".join, axis=1)
        st.success(f"Merged into `{merge_name}`")

    st.subheader("✏️ Column Renaming")
    col_to_rename = st.selectbox("Select column to rename", df.columns)
    new_name = st.text_input("New name for the column")
    if new_name and st.button("Rename Column"):
        df.rename(columns={col_to_rename: new_name}, inplace=True)
        st.success(f"Renamed `{col_to_rename}` to `{new_name}`")

    st.subheader("📊 Sorting")
    sort_col = st.selectbox("Sort by column", df.columns)
    sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
    if sort_col:
        df.sort_values(by=sort_col, ascending=(sort_order == "Ascending"), inplace=True)
        st.success(f"Sorted by `{sort_col}`")

    st.subheader("✅ Final Cleaned Data")
    st.dataframe(df)

    st.download_button("📥 Download Cleaned CSV", df.to_csv(index=False), file_name="cleansheet_cleaned.csv", mime="text/csv")

else:
    st.warning("Please upload a CSV file or load sample data.")
