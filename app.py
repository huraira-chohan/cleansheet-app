import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import requests
import dateutil.parser

# --- Session State Initialization ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Preview"
if "adv_filter_reset_key" not in st.session_state:
    st.session_state.adv_filter_reset_key = 0
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "df_original" not in st.session_state:
    st.session_state.df_original = None

# --- Page Config ---
st.set_page_config(page_title="CleanSheet - All-in-One CSV Cleaner", layout="wide")
st.title("🧹 CleanSheet")
st.caption("An all-in-one, no-code data cleaning assistant")
st.sidebar.markdown("### 📦 Load Dataset")

# --- Data Loading ---
load_sample = st.sidebar.button("📂 Load Sample Titanic Dataset")
uploaded_file = st.sidebar.file_uploader("📤 Or Upload your CSV file", type=["csv"])

if load_sample:
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        response = requests.get(titanic_url)
        df = pd.read_csv(StringIO(response.text))
        st.session_state.df_original = df.copy()
        st.session_state.df_clean = df.copy()
        st.success("✅ Sample Titanic dataset loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load sample data: {e}")

elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.replace(["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"], np.nan, inplace=True)
        st.session_state.df_original = df.copy()
        st.session_state.df_clean = df.copy()
        st.success("✅ Your dataset was uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")

if st.session_state.df_clean is None:
    st.info("📎 Please upload a CSV file or load the sample dataset to get started.")
    st.stop()

df = st.session_state.df_clean

# --- Helper Functions ---
def clean_text(x):
    return str(x).strip().title() if pd.notnull(x) else x

def convert_to_numeric(x):
    try:
        return float(re.sub(r"[^0-9.]+", "", str(x)))
    except:
        return np.nan

def auto_clean_column(series):
    try:
        numeric_series = series.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        if numeric_series.notna().sum() >= series.notna().sum() * 0.8:
            return numeric_series
    except: pass
    try:
        date_series = series.apply(lambda x: pd.to_datetime(x, errors='coerce', infer_datetime_format=True))
        if date_series.notna().sum() >= series.notna().sum() * 0.8:
            return date_series
    except: pass
    unique_vals = series.dropna().astype(str).str.strip().str.lower().unique()
    if len(unique_vals) <= 10:
        return series.astype(str).str.strip().str.lower().replace({
            "yes": "Yes", "y": "Yes", "1": "Yes", "no": "No", "n": "No", "0": "No",
            "m": "Male", "f": "Female"
        }).str.title()
    return series

# --- Sidebar Navigation ---
tab_labels = ["📊 Preview", "🧹 Clean", "🧮 Columns", "🔍 Filter", "📈 Sort", "🧠 Advanced Filter", "⬇️ Export"]
st.radio("Navigation", options=tab_labels, key="active_tab", horizontal=True, label_visibility="collapsed")

# --- Preview Tab ---
if st.session_state.active_tab == "📊 Preview":
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
    if st.button("🔄 Reset Dataset"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.rerun()

# --- Clean Tab ---
elif st.session_state.active_tab == "🧹 Clean":
    st.subheader("🧹 Clean Column Values")
    df_clean_tab = st.session_state.get("df_clean", pd.DataFrame()).copy()
    if df_clean_tab.empty:
        st.warning("⚠️ No dataset loaded.")
        st.stop()
    col = st.selectbox("Select a column to clean", df_clean_tab.columns)
    cleaning_action = st.selectbox(
        "Choose cleaning operation",
        ["-- Select --", "Remove NaNs", "Fill NaNs with 0", "To lowercase", "To title case", "Auto Clean"]
    )
    preview_col1, preview_col2 = st.columns(2)
    preview_col1.markdown("**Before Cleaning**")
    preview_col1.write(st.session_state.df_clean[[col]].head(10))
    cleaned_df = df_clean_tab.copy()
    if cleaning_action == "Remove NaNs":
        cleaned_df = df_clean_tab[df_clean_tab[col].notna()]
        st.success("✅ NaN rows removed.")
    elif cleaning_action == "Fill NaNs with 0":
        cleaned_df[col] = df_clean_tab[col].fillna(0)
        st.success("✅ NaNs filled with 0.")
    elif cleaning_action == "To lowercase":
        cleaned_df[col] = df_clean_tab[col].astype(str).str.lower()
        st.success("✅ Converted to lowercase.")
    elif cleaning_action == "To title case":
        cleaned_df[col] = df_clean_tab[col].astype(str).str.title()
        st.success("✅ Converted to title case.")
    elif cleaning_action == "Auto Clean":
        cleaned_df[col] = auto_clean_column(df_clean_tab[col])
        st.success("✅ Auto-cleaning applied to column.")
    preview_col2.markdown("**After Cleaning**")
    preview_col2.write(cleaned_df[[col]].head(10))
    if st.button("Apply Cleaning to Dataset"):
        st.session_state.df_clean = cleaned_df.copy()
        st.rerun()
    if st.button("🔄 Reset Clean Tab"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.rerun()

# --- Columns Tab ---
elif st.session_state.active_tab == "🧮 Columns":
    st.subheader("🧮 Manage Columns")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🗑 Drop Columns")
        drop_cols = st.multiselect("Select columns to drop", df.columns.tolist())
        if st.button("Drop Selected Columns"):
            df.drop(columns=drop_cols, inplace=True)
            st.session_state.df_clean = df
            st.success("✅ Dropped selected columns")
            st.rerun()
    with col2:
        st.write("### ✏️ Rename Column")
        old_col = st.selectbox("Column to rename", df.columns.tolist())
        new_col = st.text_input("New column name")
        if st.button("Rename"):
            if new_col.strip():
                df.rename(columns={old_col: new_col}, inplace=True)
                st.session_state.df_clean = df
                st.success(f"✅ Renamed `{old_col}` to `{new_col}`")
                st.rerun()
    st.write("### ➕ Merge Columns")
    merge_cols = st.multiselect("Select 2 or more columns to merge", df.columns.tolist(), key="merge_cols")
    merge_name = st.text_input("Merged column name", key="merge_name")
    sep = st.text_input("Separator (e.g. space, comma)", value=" ")
    if st.button("Merge Columns"):
        if len(merge_cols) >= 2 and merge_name:
            df[merge_name] = df[merge_cols].astype(str).agg(sep.join, axis=1)
            st.session_state.df_clean = df
            st.success(f"✅ Created merged column `{merge_name}`")
            st.rerun()
    if st.button("🔄 Reset Columns Tab"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.rerun()

# --- Filter Tab ---
elif st.session_state.active_tab == "🔍 Filter":
    st.subheader("🔍 Filter Rows (temporary view only)")
    df_to_filter = st.session_state.get("df_clean", pd.DataFrame()).copy()
    col_to_filter = st.selectbox("Choose column to filter", df_to_filter.columns.tolist())
    filtered_df = df_to_filter.copy()
    if pd.api.types.is_numeric_dtype(df_to_filter[col_to_filter]):
        st.write(f"📏 Numeric Range Filter for `{col_to_filter}`")
        min_val, max_val = float(df_to_filter[col_to_filter].min()), float(df_to_filter[col_to_filter].max())
        if min_val == max_val:
            st.warning(f"⚠️ All values in `{col_to_filter}` are the same: {min_val}")
        else:
            step_val = max((max_val - min_val) / 100, 0.01)
            start, end = st.slider("Select value range", min_value=min_val, max_value=max_val, value=(min_val, max_val), step=step_val)
            filtered_df = df_to_filter[df_to_filter[col_to_filter].between(start, end)]
    elif pd.api.types.is_datetime64_any_dtype(df_to_filter[col_to_filter]) or "date" in col_to_filter.lower():
        st.write(f"🗓 Date Range Filter for `{col_to_filter}`")
        df_to_filter[col_to_filter] = pd.to_datetime(df_to_filter[col_to_filter], errors='coerce')
        min_date = df_to_filter[col_to_filter].min()
        max_date = df_to_filter[col_to_filter].max()
        if pd.isnull(min_date) or pd.isnull(max_date):
            st.warning(f"⚠️ Could not convert `{col_to_filter}` to datetime.")
        else:
            date_start, date_end = st.date_input("Select date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            filtered_df = df_to_filter[df_to_filter[col_to_filter].between(date_start, date_end)]
    else:
        st.write(f"🔠 Categorical Filter for `{col_to_filter}`")
        unique_vals = df_to_filter[col_to_filter].dropna().unique().tolist()
        selected_vals = st.multiselect("Select values to include from `{col_to_filter}`:", options=unique_vals, default=unique_vals)
        filtered_df = df_to_filter[df_to_filter[col_to_filter].isin(selected_vals)] if selected_vals else df_to_filter.head(0)
    st.dataframe(filtered_df, use_container_width=True)
    if st.button("✅ Apply This Filter to Dataset"):
        st.session_state.df_clean = filtered_df.copy()
        st.rerun()
    if st.button("🔄 Reset Filters Tab"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.rerun()

# --- Sort Tab ---
elif st.session_state.active_tab == "📈 Sort":
    st.subheader("📈 Sort Data")
    sort_col = st.selectbox("Column to sort by", df.columns.tolist())
    ascending = st.checkbox("Sort ascending", value=True)
    if st.button("Sort Data"):
        df = df.sort_values(by=sort_col, ascending=ascending)
        st.session_state.df_clean = df
        st.success(f"✅ Sorted by `{sort_col}`")
        st.rerun()
    if st.button("Remove Duplicates"):
        df = df.drop_duplicates()
        st.session_state.df_clean = df
        st.success("✅ Duplicate rows removed")
        st.rerun()
    if st.button("🔄 Reset Sort Tab"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.rerun()

# --- Advanced Filter Tab ---
elif st.session_state.active_tab == "🧠 Advanced Filter":
    st.subheader("🧠 Advanced Multi-Column Filtering")
    adv_df = st.session_state.get("df_clean", pd.DataFrame()).copy()
    adv_filter_key_prefix = f"adv_filter_{st.session_state.adv_filter_reset_key}"
    num_conditions = st.number_input("How many filter conditions?", min_value=1, max_value=5, value=1, key=f"{adv_filter_key_prefix}_num_conditions")
    logic = st.radio("Combine filters using:", ["AND", "OR"], horizontal=True, key=f"{adv_filter_key_prefix}_logic")
    conditions = []
    for i in range(int(num_conditions)):
        col = st.selectbox(f"Column for condition #{i+1}", adv_df.columns, key=f"{adv_filter_key_prefix}_col_{i}")
        if pd.api.types.is_numeric_dtype(adv_df[col]):
            min_val, max_val = float(adv_df[col].min()), float(adv_df[col].max())
            range_val = st.slider(f"Range for `{col}`", min_val, max_val, (min_val, max_val), key=f"{adv_filter_key_prefix}_range_{i}")
            cond = adv_df[col].between(range_val[0], range_val[1])
        elif pd.api.types.is_datetime64_any_dtype(adv_df[col]) or "date" in col.lower():
            adv_df[col] = pd.to_datetime(adv_df[col], errors="coerce")
            min_date, max_date = adv_df[col].min(), adv_df[col].max()
            start_date, end_date = st.date_input(f"Date range for `{col}`", (min_date.date(), max_date.date()), key=f"{adv_filter_key_prefix}_date_{i}")
            cond = adv_df[col].dt.date.between(start_date, end_date)
        else:
            values = adv_df[col].dropna().unique().tolist()
            selected = st.multiselect(f"Values for `{col}`", values, key=f"{adv_filter_key_prefix}_cat_{i}", default=values)
            cond = adv_df[col].isin(selected) if selected else pd.Series([False] * len(adv_df), index=adv_df.index)
        conditions.append(cond)
    final_mask = conditions[0]
    for c in conditions[1:]:
        final_mask = final_mask & c if logic == "AND" else final_mask | c
    adv_filtered_df = adv_df[final_mask]
    st.dataframe(adv_filtered_df, use_container_width=True)
    if st.button("✅ Apply These Advanced Filters"):
        st.session_state.df_clean = adv_filtered_df.copy()
        st.rerun()
    if st.button("🔄 Reset Advanced Filter Tab"):
        st.session_state.adv_filter_reset_key += 1
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.rerun()

# --- Export Tab ---
elif st.session_state.active_tab == "⬇️ Export":
    st.subheader("⬇️ Export Cleaned CSV")
    export_df = st.session_state.get("df_clean", pd.DataFrame())
    st.dataframe(export_df, use_container_width=True)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )
    if st.button("🔄 Reset Export Tab"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.rerun()
