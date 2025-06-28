import streamlit as st
import pandas as pd
import numpy as np
import re
from dateutil import parser
from urllib.parse import urlparse
from scipy import stats
from io import StringIO
import requests

st.set_page_config("CleanSheet Final", layout="wide")
st.title("🧹 CleanSheet v10 – Final Version")
st.caption("A no-error, fully-featured CSV data cleaner (no AI needed)")

# Predefined nulls
NULL_VALUES = ["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"]

# Functions
def clean_text(x): return str(x).strip().title() if pd.notnull(x) else x
def normalize_gender(g):
    g = str(g).strip().lower()
    return "Male" if g in ["m", "male"] else "Female" if g in ["f", "female"] else "Other"
def convert_to_numeric(x): 
    try: return float(re.sub(r"[^0-9.]+", "", str(x)))
    except: return np.nan
def parse_any_date(x):
    try: return parser.parse(str(x), fuzzy=True)
    except: return np.nan
def is_valid_email(x): return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$', str(x)))
def is_valid_url(x):
    try: return all([urlparse(x).scheme, urlparse(x).netloc])
    except: return False
def remove_outliers_zscore(series, threshold=3):
    if pd.api.types.is_numeric_dtype(series):
        z = np.abs(stats.zscore(series.dropna()))
        return series.where(z < threshold)
    return series

# Upload file
st.sidebar.markdown("### 📂 Load CSV")
use_sample = st.sidebar.button("📥 Load Titanic Sample")
uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if use_sample:
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    uploaded = StringIO(requests.get(url).text)

if uploaded:
    df = pd.read_csv(uploaded)
    df.replace(NULL_VALUES, np.nan, inplace=True)
    original_df = df.copy()

    st.subheader("🔍 Filter & Sort Rows")
    col1, col2 = st.columns(2)
    with col1:
        filter_col = st.selectbox("Choose a column to filter", df.columns)
        filter_val = st.text_input("Value to keep (exact match)")
        if filter_val:
            df = df[df[filter_col].astype(str) == filter_val]

    with col2:
        sort_col = st.selectbox("Sort by column", df.columns)
        asc = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
        df = df.sort_values(by=sort_col, ascending=asc == "Ascending")

    st.subheader("✂️ Drop or Rename Columns")
    with st.expander("🔻 Drop Columns"):
        drop_cols = st.multiselect("Select columns to drop", df.columns)
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

    with st.expander("✏️ Rename Columns"):
        rename_map = {}
        for col in df.columns:
            new_name = st.text_input(f"Rename `{col}` to", col)
            if new_name and new_name != col:
                rename_map[col] = new_name
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

    st.subheader("➕ Merge Columns")
    with st.expander("🧬 Combine Columns"):
        merge_cols = st.multiselect("Select columns to merge", df.columns)
        new_col_name = st.text_input("New column name", "merged_column")
        separator = st.text_input("Separator", " ")
        if merge_cols and new_col_name:
            df[new_col_name] = df[merge_cols].astype(str).apply(lambda row: separator.join(row), axis=1)

    st.subheader("🧼 Clean Individual Columns")
    for col in df.columns:
        st.markdown(f"**🔧 `{col}`**")
        clean_option = st.selectbox(f"Cleaning for `{col}`",
            ["None", "Text Normalize", "Numeric Convert", "Date Parse", "Normalize Gender", "Validate Email", "Validate URL", "Remove Outliers"],
            key=f"clean_{col}")
        
        if clean_option == "Text Normalize":
            df[col] = df[col].apply(clean_text)
        elif clean_option == "Numeric Convert":
            df[col] = df[col].apply(convert_to_numeric)
        elif clean_option == "Date Parse":
            df[col] = df[col].apply(parse_any_date)
        elif clean_option == "Normalize Gender":
            df[col] = df[col].apply(normalize_gender)
        elif clean_option == "Validate Email":
            df[f"{col}_valid"] = df[col].apply(is_valid_email)
        elif clean_option == "Validate URL":
            df[f"{col}_valid"] = df[col].apply(is_valid_url)
        elif clean_option == "Remove Outliers":
            df[col] = remove_outliers_zscore(df[col])

    st.subheader("🧾 Final Preview & Download")
    view_opt = st.radio("Rows to view", ["Top 5", "Top 50", "All"], horizontal=True)
    if view_opt == "Top 5":
        st.dataframe(df.head())
    elif view_opt == "Top 50":
        st.dataframe(df.head(50))
    else:
        st.dataframe(df)

    st.download_button("📥 Download Cleaned CSV", data=df.to_csv(index=False), file_name="cleaned_data.csv", mime="text/csv")
else:
    st.warning("📄 Please upload a CSV file or load the Titanic dataset to begin.")
