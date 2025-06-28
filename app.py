import streamlit as st
import pandas as pd
import numpy as np
import re
from dateutil import parser
from urllib.parse import urlparse
from scipy import stats

# Page settings
st.set_page_config("🧹 CleanSheet - CSV Cleaner", layout="wide")
st.title("🧹 CleanSheet - CSV Cleaner")
st.caption("A simple, powerful CSV cleaner with full control. No AI or API required.")

NULL_VALUES = ["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"]

# Cleaning functions
def clean_text(x):
    return str(x).strip().title() if pd.notnull(x) else x

def normalize_gender(g):
    g = str(g).strip().lower()
    if g in ["m", "male"]: return "Male"
    elif g in ["f", "female"]: return "Female"
    return "Other"

def convert_to_numeric(x):
    try:
        return float(re.sub(r"[^\d.]", "", str(x)))
    except: return np.nan

def parse_any_date(date_str):
    try: return parser.parse(str(date_str), fuzzy=True)
    except: return np.nan

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except: return False

def remove_outliers_zscore(series, threshold=3):
    if pd.api.types.is_numeric_dtype(series):
        z = np.abs(stats.zscore(series.dropna()))
        mask = (z < threshold)
        return series.where(series.isin(series.dropna()[mask]))
    return series

def profile_column(col_data):
    return {
        "type": str(col_data.dtype),
        "% missing": round(col_data.isnull().mean() * 100, 2),
        "unique": col_data.nunique(),
        "top": col_data.value_counts().index[0] if not col_data.dropna().empty else "",
        "top freq": col_data.value_counts().iloc[0] if not col_data.dropna().empty else 0
    }

# --- File upload
st.sidebar.subheader("📤 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.replace(NULL_VALUES, np.nan, inplace=True)

    st.subheader("✅ Data Preview")
    view = st.radio("How many rows to display?", ["Top 5", "Top 50", "All"], horizontal=True)
    if view == "Top 5": st.dataframe(df.head(), use_container_width=True)
    elif view == "Top 50": st.dataframe(df.head(50), use_container_width=True)
    else: st.dataframe(df, use_container_width=True)

    st.subheader("🧪 Column Profiling")
    profile = pd.DataFrame({col: profile_column(df[col]) for col in df.columns}).T
    st.dataframe(profile)

    st.subheader("🧰 Column Operations")
    with st.expander("🧺 Rename Columns"):
        new_names = {}
        for col in df.columns:
            new_name = st.text_input(f"Rename '{col}'", value=col)
            new_names[col] = new_name
        df.rename(columns=new_names, inplace=True)

    with st.expander("➕ Merge Columns"):
        merge_cols = st.multiselect("Select columns to merge", df.columns)
        if merge_cols:
            merge_name = st.text_input("Name for merged column", value="_".join(merge_cols))
            sep = st.text_input("Separator", value=" ")
            if st.button("🔗 Merge Columns"):
                df[merge_name] = df[merge_cols].astype(str).agg(sep.join, axis=1)
                st.success(f"Merged into column '{merge_name}'")

    with st.expander("🗑️ Drop Columns"):
        drop_cols = st.multiselect("Select columns to drop", df.columns)
        if st.button("🧹 Drop Selected Columns"):
            df.drop(columns=drop_cols, inplace=True)
            st.success(f"Dropped: {', '.join(drop_cols)}")

    st.subheader("🧼 Column Cleaning Options")
    for col in df.columns:
        with st.expander(f"⚙️ Clean `{col}`"):
            clean_type = st.selectbox("Cleaning method", 
                ["none", "text_normalize", "numeric", "date", "gender", "email_validate", "url_validate", "drop"], key=col)
            
            fill_strategy = st.radio("Missing value strategy", 
                ["none", "drop rows", "fill mean", "fill median", "fill mode"], horizontal=True, key=f"fill_{col}")
            
            outlier_removal = st.checkbox("Remove outliers (numeric only)", value=False, key=f"outlier_{col}")

            # Apply
            if clean_type == "drop":
                df.drop(columns=[col], inplace=True)
                st.warning(f"Column `{col}` dropped.")
                continue
            elif clean_type == "text_normalize":
                df[col] = df[col].apply(clean_text)
            elif clean_type == "numeric":
                df[col] = df[col].apply(convert_to_numeric)
            elif clean_type == "date":
                df[col] = df[col].apply(parse_any_date)
            elif clean_type == "gender":
                df[col] = df[col].apply(normalize_gender)
            elif clean_type == "email_validate":
                df[f"{col}_valid"] = df[col].apply(is_valid_email)
            elif clean_type == "url_validate":
                df[f"{col}_valid"] = df[col].apply(is_valid_url)

            if fill_strategy == "drop rows":
                df = df[df[col].notna()]
            elif fill_strategy == "fill mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
            elif fill_strategy == "fill median":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
            elif fill_strategy == "fill mode":
                if not df[col].mode().empty:
                    df[col].fillna(df[col].mode().iloc[0], inplace=True)

            if outlier_removal:
                df[col] = remove_outliers_zscore(df[col])

    st.subheader("📊 Final Cleaned Dataset")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Cleaned CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")
