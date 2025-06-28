import streamlit as st
import pandas as pd
import numpy as np
import re
from dateutil import parser
from urllib.parse import urlparse
from scipy import stats
from io import StringIO
import requests

# --- App config ---
st.set_page_config(page_title="CleanSheet", layout="wide")
st.title("🧹 CleanSheet: Smart CSV Cleaner")

# --- Helpers ---
NULL_VALUES = ["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"]

def clean_text(x): return str(x).strip().title() if pd.notnull(x) else x
def normalize_gender(g):
    g = str(g).strip().lower()
    if g in ["m", "male"]: return "Male"
    if g in ["f", "female"]: return "Female"
    return "Other"
def convert_to_numeric(x):
    try: return float(re.sub(r"[^\d.]+", "", str(x)))
    except: return np.nan
def parse_any_date(x):
    try: return parser.parse(str(x), fuzzy=True)
    except: return np.nan
def is_valid_email(email): return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$', str(email)))
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except: return False
def remove_outliers_zscore(series, threshold=3):
    if pd.api.types.is_numeric_dtype(series):
        z = np.abs(stats.zscore(series.dropna()))
        return series.where(z < threshold)
    return series

# --- Load dataset into session state ---
if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "df_cleaned" not in st.session_state:
    st.session_state.df_cleaned = None

# --- Load CSV ---
st.sidebar.header("📂 Upload or Load Sample")
uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if st.sidebar.button("Use Sample Titanic Dataset"):
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    st.session_state.df_original = pd.read_csv(url)
    st.session_state.df_cleaned = st.session_state.df_original.copy()

elif uploaded:
    df = pd.read_csv(uploaded)
    df.replace(NULL_VALUES, np.nan, inplace=True)
    st.session_state.df_original = df
    st.session_state.df_cleaned = df.copy()

# --- Display dataset if loaded ---
if st.session_state.df_cleaned is not None:
    df = st.session_state.df_cleaned

    st.subheader("📊 Dataset Preview")
    view = st.radio("Show", ["Top 5", "Top 50", "All"], horizontal=True)
    if view == "Top 5":
        st.dataframe(df.head())
    elif view == "Top 50":
        st.dataframe(df.head(50))
    else:
        st.dataframe(df)

    # --- Column Actions ---
    st.subheader("🧪 Column Actions")

    with st.expander("📉 Drop Columns"):
        drop_cols = st.multiselect("Select columns to drop", df.columns)
        if st.button("Drop Selected Columns"):
            df.drop(columns=drop_cols, inplace=True)
            st.session_state.df_cleaned = df
            st.success("Dropped selected columns.")

    with st.expander("✏️ Rename Columns"):
        col_to_rename = st.selectbox("Select column to rename", df.columns)
        new_name = st.text_input("New column name")
        if st.button("Rename Column") and new_name:
            df.rename(columns={col_to_rename: new_name}, inplace=True)
            st.session_state.df_cleaned = df
            st.success(f"Renamed `{col_to_rename}` to `{new_name}`.")

    with st.expander("🔗 Merge Columns"):
        col1 = st.selectbox("Column 1", df.columns, key="merge1")
        col2 = st.selectbox("Column 2", df.columns, key="merge2")
        new_col_name = st.text_input("Name of merged column")
        sep = st.text_input("Separator", " ")
        if st.button("Merge Columns") and new_col_name:
            df[new_col_name] = df[col1].astype(str) + sep + df[col2].astype(str)
            st.session_state.df_cleaned = df
            st.success(f"Merged into `{new_col_name}`.")

    # --- Row Actions ---
    st.subheader("🧹 Row Actions")

    with st.expander("🧽 Remove Duplicate Rows"):
        if st.button("Remove Duplicates"):
            before = len(df)
            df.drop_duplicates(inplace=True)
            after = len(df)
            st.session_state.df_cleaned = df
            st.success(f"Removed {before - after} duplicate rows.")

    with st.expander("🔍 Filter Rows by Column Value"):
        col = st.selectbox("Select column to filter", df.columns, key="filter_col")
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val, max_val = st.slider("Select range", float(df[col].min()), float(df[col].max()), (float(df[col].min()), float(df[col].max())))
            if st.button("Apply Numeric Filter"):
                df = df[(df[col] >= min_val) & (df[col] <= max_val)]
                st.session_state.df_cleaned = df
        else:
            unique_vals = df[col].dropna().unique().tolist()
            selected = st.multiselect("Select values", unique_vals)
            if st.button("Apply Text Filter") and selected:
                df = df[df[col].isin(selected)]
                st.session_state.df_cleaned = df

    # --- Download ---
    st.subheader("⬇️ Download Cleaned CSV")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")

else:
    st.info("📁 Upload or load a dataset to begin.")

