
import streamlit as st
import pandas as pd
import numpy as np
import re
from dateutil import parser
from scipy import stats
from io import StringIO
import requests

# ---- SETUP ----
st.set_page_config(page_title="CleanSheet", layout="wide")
st.title("🧹 CleanSheet - Simple CSV Cleaner")
st.markdown("Upload a dataset and choose cleaning steps. No AI or API keys needed.")

NULL_VALUES = ["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"]

# ---- CLEANING HELPERS ----
def clean_text(x):
    return str(x).strip().title() if pd.notnull(x) else x

def normalize_gender(g):
    g = str(g).strip().lower()
    return "Male" if g in ["m", "male"] else "Female" if g in ["f", "female"] else "Other"

def convert_to_numeric(x):
    try: return float(re.sub(r"[^0-9.]+", "", str(x)))
    except: return np.nan

def parse_any_date(date_str):
    try: return parser.parse(str(date_str), fuzzy=True)
    except: return np.nan

def is_valid_email(email):
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$', str(email)))

def is_valid_url(url):
    try:
        from urllib.parse import urlparse
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except: return False

def remove_outliers_zscore(series, threshold=3):
    if pd.api.types.is_numeric_dtype(series):
        non_na = series.dropna()
        z = np.abs(stats.zscore(non_na))
        return series.where(series.isin(non_na[z < threshold]))
    return series

# ---- FILE UPLOAD ----
st.sidebar.subheader("📂 Load Dataset")
if st.sidebar.button("Use Titanic Sample"):
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    uploaded_file = StringIO(requests.get(url).text)
    uploaded_file.name = "titanic.csv"
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.replace(NULL_VALUES, np.nan, inplace=True)

    # ---- FILTER ROWS ----
    st.sidebar.subheader("🔎 Filter Rows")
    filter_column = st.sidebar.selectbox("Select column to filter", ["None"] + list(df.columns))
    if filter_column != "None":
        unique_vals = df[filter_column].dropna().unique()
        selected_vals = st.sidebar.multiselect("Choose values to keep", unique_vals)
        if selected_vals:
            df = df[df[filter_column].isin(selected_vals)]

    # ---- COLUMN DROPPING ----
    st.sidebar.subheader("🗑️ Drop Columns")
    cols_to_drop = st.sidebar.multiselect("Columns to drop", df.columns)
    df.drop(columns=cols_to_drop, inplace=True)

    # ---- COLUMN RENAMING ----
    st.sidebar.subheader("✏️ Rename Columns")
    rename_dict = {}
    for col in df.columns:
        new_name = st.sidebar.text_input(f"Rename '{col}' to:", value=col, key=f"rename_{col}")
        rename_dict[col] = new_name
    df.rename(columns=rename_dict, inplace=True)

    # ---- COLUMN MERGING ----
    st.sidebar.subheader("🔗 Merge Columns")
    merge_cols = st.sidebar.multiselect("Select columns to merge (into one)", df.columns)
    separator = st.sidebar.text_input("Separator for merged column", " | ")
    new_col_name = st.sidebar.text_input("Name of new merged column", "MergedColumn")
    if st.sidebar.button("Merge Columns"):
        df[new_col_name] = df[merge_cols].astype(str).agg(separator.join, axis=1)

    st.subheader("📄 Data Preview")
    view_option = st.radio("View", ["Top 5", "Top 50", "All"], horizontal=True)
    if view_option == "Top 5":
        st.dataframe(df.head())
    elif view_option == "Top 50":
        st.dataframe(df.head(50))
    else:
        st.dataframe(df)

    # ---- COLUMN CLEANING ----
    st.subheader("🛠️ Column Cleaning Options")
    col_config = {}
    with st.form("cleaning_form"):
        for col in df.columns:
            with st.expander(f"⚙️ {col}"):
                action = st.selectbox(
                    "Cleaning method",
                    ["none", "clean_text", "numeric", "date", "gender", "email_validate", "url_validate"],
                    key=f"action_{col}"
                )
                fill = st.selectbox("Missing value strategy", ["none", "drop", "fill_mean", "fill_median", "fill_mode"], key=f"fill_{col}")
                outliers = st.checkbox("Remove outliers (Z-score)", value=False, key=f"outlier_{col}")
                col_config[col] = (action, fill, outliers)
        submitted = st.form_submit_button("🧼 Clean My Data")

    if submitted:
        for col, (action, fill, outliers) in col_config.items():
            if action == "clean_text":
                df[col] = df[col].apply(clean_text)
            elif action == "numeric":
                df[col] = df[col].apply(convert_to_numeric)
            elif action == "date":
                df[col] = df[col].apply(parse_any_date)
            elif action == "gender":
                df[col] = df[col].apply(normalize_gender)
            elif action == "email_validate":
                df[col + "_valid"] = df[col].apply(is_valid_email)
            elif action == "url_validate":
                df[col + "_valid"] = df[col].apply(is_valid_url)

            if fill == "drop":
                df = df[df[col].notna()]
            elif fill == "fill_mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            elif fill == "fill_median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif fill == "fill_mode":
                mode = df[col].mode().iloc[0] if not df[col].mode().empty else None
                df[col].fillna(mode, inplace=True)

            if outliers:
                df[col] = remove_outliers_zscore(df[col])

        st.success("✅ Data cleaned!")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned CSV", data=csv, file_name="cleansheet_cleaned.csv", mime="text/csv")
else:
    st.info("📌 Upload a CSV file or load a sample from the sidebar to get started.")
