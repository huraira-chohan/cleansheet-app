import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import requests
from io import StringIO

st.set_page_config(page_title="CleanSheet - All-in-One CSV Cleaner", layout="wide")
st.title("🧹 CleanSheet")
st.caption("An all-in-one, no-code data cleaning assistant")
st.sidebar.markdown("### 📦 Load Dataset")

load_sample = st.sidebar.button("Load Sample Titanic Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if load_sample:
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    response = requests.get(titanic_url)
    df = pd.read_csv(StringIO(response.text))
    st.session_state.df_clean = df.copy()
    st.success("✅ Sample Titanic dataset loaded successfully!")

elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df_clean = df.copy()
    st.success("✅ Your dataset was uploaded successfully!")

# --- Helpers ---
NULL_VALUES = ["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"]

def clean_text(x):
    return str(x).strip().title() if pd.notnull(x) else x

def convert_to_numeric(x):
    try:
        return float(re.sub(r"[^0-9.]+", "", str(x)))
    except:
        return np.nan

# --- Dataset loading ---
st.sidebar.header("📤 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.replace(NULL_VALUES, np.nan, inplace=True)
    st.session_state.df_original = df.copy()
    st.session_state.df_clean = df.copy()
elif "df_clean" in st.session_state:
    df = st.session_state.df_clean
else:
    st.info("📎 Please upload a CSV file to get started.")
    st.stop()

# --- Tabs for navigation ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Preview", "🧹 Clean", "🧮 Columns", "🔍 Filter", "📈 Sort", "⬇️ Export"
])

# --- Preview Tab ---
with tab1:
    st.subheader("🔎 Dataset Preview")
    view_opt = st.radio("How much data to show?", ["Top 5", "Top 50", "All"], horizontal=True)
    if view_opt == "Top 5":
        st.dataframe(df.head(), use_container_width=True)
    elif view_opt == "Top 50":
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

    st.write("#### ℹ️ Column Summary")
    st.dataframe(df.describe(include='all').T.fillna("N/A"))

# --- Clean Tab ---
with tab2:
    st.subheader("🧼 Clean Columns")
    columns = df.columns.tolist()

    for col in columns:
        with st.expander(f"⚙️ {col}"):
            clean_opt = st.selectbox(f"Cleaning for `{col}`", [
                "None", "Text Normalize", "Convert to Numeric"
            ], key=f"clean_{col}")

            fill_na = st.selectbox(f"Missing values in `{col}`", [
                "None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"
            ], key=f"na_{col}")

            if st.button(f"✅ Apply to `{col}`", key=f"apply_{col}"):
                if clean_opt == "Text Normalize":
                    df[col] = df[col].apply(clean_text)
                elif clean_opt == "Convert to Numeric":
                    df[col] = df[col].apply(convert_to_numeric)

                if fill_na == "Drop Rows":
                    df = df[df[col].notna()]
                elif fill_na == "Fill with Mean" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                elif fill_na == "Fill with Median" and pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                elif fill_na == "Fill with Mode":
                    df[col].fillna(df[col].mode().iloc[0], inplace=True)

                st.success(f"✅ Cleaning applied to `{col}`")
                st.session_state.df_clean = df

# --- Column Tab ---
with tab3:
    st.subheader("🧮 Manage Columns")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🗑 Drop Columns")
        drop_cols = st.multiselect("Select columns to drop", df.columns.tolist())
        if st.button("Drop Selected Columns"):
            df.drop(columns=drop_cols, inplace=True)
            st.success("✅ Dropped selected columns")
            st.session_state.df_clean = df

    with col2:
        st.write("### ✏️ Rename Column")
        old_col = st.selectbox("Column to rename", df.columns.tolist())
        new_col = st.text_input("New column name")
        if st.button("Rename"):
            if new_col.strip():
                df.rename(columns={old_col: new_col}, inplace=True)
                st.success(f"✅ Renamed `{old_col}` to `{new_col}`")
                st.session_state.df_clean = df

    st.write("### ➕ Merge Columns")
    merge_cols = st.multiselect("Select 2 or more columns to merge", df.columns.tolist(), key="merge_cols")
    merge_name = st.text_input("Merged column name", key="merge_name")
    sep = st.text_input("Separator (e.g. space, comma)", value=" ")
    if st.button("Merge Columns"):
        if len(merge_cols) >= 2 and merge_name:
            df[merge_name] = df[merge_cols].astype(str).agg(sep.join, axis=1)
            st.success(f"✅ Created merged column `{merge_name}`")
            st.session_state.df_clean = df

    st.write("### 🔤 Split Alphanumeric Column")
    split_col = st.selectbox("Select alphanumeric column to split", df.columns.tolist(), key="split_col")
    new_alpha = st.text_input("New column for alphabets", key="alpha_part")
    new_num = st.text_input("New column for numbers", key="num_part")

    if st.button("Split Alphanumeric"):
        if new_alpha and new_num:
            df[new_alpha] = df[split_col].astype(str).apply(lambda x: ''.join(re.findall(r'[A-Za-z]+', x)))
            df[new_num] = df[split_col].astype(str).apply(lambda x: ''.join(re.findall(r'\d+', x)))
            st.success(f"✅ Split `{split_col}` into `{new_alpha}` and `{new_num}`")
            st.session_state.df_clean = df


# --- Filter Tab ---
with tab4:
    st.subheader("🔍 Filter Rows")
    col_to_filter = st.selectbox("Choose column", df.columns.tolist())
    unique_vals = df[col_to_filter].dropna().unique().tolist()

    selected_vals = st.multiselect(f"Show rows where `{col_to_filter}` is:", unique_vals)
    if selected_vals:
        df = df[df[col_to_filter].isin(selected_vals)]
        st.session_state.df_clean = df
        st.success("✅ Filter applied")

# --- Sort Tab ---
with tab5:
    st.subheader("📈 Sort Data")
    sort_col = st.selectbox("Column to sort by", df.columns.tolist())
    ascending = st.checkbox("Sort ascending", value=True)
    if st.button("Sort"):
        df = df.sort_values(by=sort_col, ascending=ascending)
        st.success(f"✅ Sorted by `{sort_col}`")
        st.session_state.df_clean = df

    st.write("### 🧬 Remove Duplicate Rows")
    if st.button("Remove Duplicates"):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        st.success(f"✅ Removed {before - after} duplicate rows")
        st.session_state.df_clean = df

# --- Export Tab ---
with tab6:
    st.subheader("⬇️ Export Cleaned CSV")

    export_view = st.radio("How much data to preview?", ["Top 5", "Top 50", "All"], horizontal=True, key="export_view")
    if export_view == "Top 5":
        st.dataframe(df.head(), use_container_width=True)
    elif export_view == "Top 50":
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")


