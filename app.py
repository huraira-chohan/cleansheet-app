import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import requests

# --- Page and State Configuration ---
st.set_page_config(page_title="CleanSheet - All-in-One CSV Cleaner", layout="wide")

# Centralized session state initialization
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Preview"
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "adv_filter_reset_key" not in st.session_state:
    st.session_state.adv_filter_reset_key = 0

# --- UI Header ---
st.title("🧹 CleanSheet")
st.caption("An all-in-one, no-code data cleaning assistant")

# --- Sidebar for Data Loading ---
st.sidebar.markdown("### 📦 Load Dataset")
load_sample = st.sidebar.button("📂 Load Sample Titanic Dataset")
uploaded_file = st.sidebar.file_uploader("📤 Or Upload your CSV file", type=["csv"])

# --- Data Loading Logic ---
if load_sample:
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        df = pd.read_csv(StringIO(response.text))
        st.session_state.df_original = df.copy()
        st.session_state.df_clean = df.copy()
        st.session_state.adv_filter_reset_key += 1
        st.success("✅ Sample Titanic dataset loaded successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to load sample data: {e}")

elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = df.replace(["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"], np.nan)
        st.session_state.df_original = df.copy()
        st.session_state.df_clean = df.copy()
        st.session_state.adv_filter_reset_key += 1
        st.success("✅ Your dataset was uploaded successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")

# Exit if no data is loaded
if st.session_state.df_clean is None:
    st.info("📎 Please upload a CSV file or load the sample dataset to get started.")
    st.stop()

# df is the WORKING copy. df_original is the STATIC copy.
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
            "yes": "Yes", "y": "Yes", "1": "Yes",
            "no": "No", "n": "No", "0": "No",
            "m": "Male", "f": "Female"
        }).str.title()
    return series

# --- Main App with Tabs ---
tab_labels = ["📊 Preview", "🧹 Clean", "🧮 Columns", "🔍 Filter", "📈 Sort", "🧠 Advanced Filter", "⬇️ Export"]
tabs = st.tabs(tab_labels)

# --- Preview Tab (MODIFIED) ---
with tabs[0]:
    st.subheader("🔎 Original Dataset Preview")
    st.info("This tab always shows the original, unmodified dataset that you first loaded.")
    
    df_original = st.session_state.df_original
    
    view_opt = st.radio("How much data to show?", ["Top 5", "Top 50", "All"], horizontal=True, key="preview_radio")
    
    if view_opt == "Top 5":
        st.dataframe(df_original.head(), use_container_width=True)
    elif view_opt == "Top 50":
        st.dataframe(df_original.head(50), use_container_width=True)
    else:
        st.dataframe(df_original, use_container_width=True)

    st.write("#### ℹ️ Column Summary (Original Data)")
    st.dataframe(df_original.describe(include='all').T.fillna("N/A"))


# --- Clean Tab ---
with tabs[1]:
    st.subheader("🧹 Clean Column Values")
    col = st.selectbox("Select a column to clean", df.columns)
    cleaning_action = st.selectbox(
        "Choose cleaning operation",
        ["-- Select --", "Remove NaNs", "Fill NaNs with 0", "To lowercase", "To title case", "Auto Clean"]
    )

    preview_col1, preview_col2 = st.columns(2)
    preview_col1.markdown("**Before Cleaning (Current State)**")
    preview_col1.write(df[[col]].head(10))

    cleaned_df = df.copy()
    if cleaning_action == "Remove NaNs":
        cleaned_df = df[df[col].notna()]
    elif cleaning_action == "Fill NaNs with 0":
        cleaned_df[col] = df[col].fillna(0)
    elif cleaning_action == "To lowercase":
        cleaned_df[col] = df[col].astype(str).str.lower()
    elif cleaning_action == "To title case":
        cleaned_df[col] = df[col].astype(str).str.title()
    elif cleaning_action == "Auto Clean":
        cleaned_df[col] = auto_clean_column(df[col])

    preview_col2.markdown("**After Cleaning (Preview)**")
    preview_col2.write(cleaned_df[[col]].head(10))

    if st.button("Apply Cleaning to Dataset"):
        st.session_state.df_clean = cleaned_df.copy()
        st.success("✅ Cleaning applied!")
        st.rerun()

    st.markdown("---")
    st.subheader("🔎 Clean All Columns (Comprehensive)")
    for col_to_clean in df.columns:
        with st.expander(f"⚙️ Options for `{col_to_clean}`"):
            clean_opt = st.selectbox(f"Cleaning for `{col_to_clean}`", ["None", "Text Normalize", "Convert to Numeric"], key=f"clean_{col_to_clean}")
            fill_na = st.selectbox(f"Missing values in `{col_to_clean}`", ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"], key=f"na_{col_to_clean}")

            if st.button(f"✅ Apply to `{col_to_clean}`", key=f"apply_{col_to_clean}"):
                temp_df = st.session_state.df_clean.copy()
                if clean_opt == "Text Normalize":
                    temp_df[col_to_clean] = temp_df[col_to_clean].apply(clean_text)
                elif clean_opt == "Convert to Numeric":
                    temp_df[col_to_clean] = temp_df[col_to_clean].apply(convert_to_numeric)

                if fill_na == "Drop Rows":
                    temp_df = temp_df.dropna(subset=[col_to_clean])
                elif fill_na == "Fill with Mean" and pd.api.types.is_numeric_dtype(temp_df[col_to_clean]):
                    temp_df[col_to_clean] = temp_df[col_to_clean].fillna(temp_df[col_to_clean].mean())
                elif fill_na == "Fill with Median" and pd.api.types.is_numeric_dtype(temp_df[col_to_clean]):
                    temp_df[col_to_clean] = temp_df[col_to_clean].fillna(temp_df[col_to_clean].median())
                elif fill_na == "Fill with Mode":
                    mode_val = temp_df[col_to_clean].mode()
                    if not mode_val.empty:
                        temp_df[col_to_clean] = temp_df[col_to_clean].fillna(mode_val.iloc[0])
                    else:
                        st.warning(f"Column `{col_to_clean}` has no mode. No values were filled.")
                
                st.session_state.df_clean = temp_df
                st.success(f"✅ Cleaning applied to `{col_to_clean}`")
                st.rerun()

# --- Column Tab ---
with tabs[2]:
    st.subheader("🧮 Manage Columns")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🗑 Drop Columns")
        drop_cols = st.multiselect("Select columns to drop", df.columns.tolist())
        if st.button("Drop Selected Columns"):
            st.session_state.df_clean = df.drop(columns=drop_cols)
            st.success("✅ Dropped selected columns")
            st.rerun()

    with col2:
        st.write("### ✏️ Rename Column")
        old_col = st.selectbox("Column to rename", df.columns.tolist())
        new_col = st.text_input("New column name")
        if st.button("Rename"):
            if new_col.strip():
                st.session_state.df_clean = df.rename(columns={old_col: new_col})
                st.success(f"✅ Renamed `{old_col}` to `{new_col}`")
                st.rerun()

    st.markdown("---")
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

    st.markdown("---")
    st.write("### 🔤 Split Alphanumeric Column")
    split_col = st.selectbox("Select alphanumeric column to split", df.columns.tolist(), key="split_col")
    new_alpha = st.text_input("New column for alphabets", key="alpha_part")
    new_num = st.text_input("New column for numbers", key="num_part")

    if st.button("Split Alphanumeric"):
        if new_alpha and new_num:
            df[new_alpha] = df[split_col].astype(str).str.findall(r'[A-Za-z]').str.join('')
            df[new_num] = df[split_col].astype(str).str.findall(r'\d').str.join('')
            st.session_state.df_clean = df
            st.success(f"✅ Split `{split_col}` into `{new_alpha}` and `{new_num}`")
            st.rerun()

# --- Filter Tab ---
with tabs[3]:
    st.subheader("🔍 Filter Rows")
    st.info("This filter creates a temporary view. Press 'Apply' to make changes permanent.")
    col_to_filter = st.selectbox("Choose column to filter", df.columns.tolist())
    
    filtered_df = df.copy()
    
    if pd.api.types.is_numeric_dtype(df[col_to_filter]):
        min_val, max_val = float(df[col_to_filter].min()), float(df[col_to_filter].max())
        if min_val == max_val:
            st.warning(f"⚠️ All values in `{col_to_filter}` are the same: {min_val}")
        else:
            start, end = st.slider("Select value range", min_val, max_val, (min_val, max_val))
            filtered_df = df[df[col_to_filter].between(start, end)]
    
    elif pd.api.types.is_datetime64_any_dtype(df[col_to_filter]):
        temp_col = pd.to_datetime(df[col_to_filter], errors='coerce')
        min_date, max_date = temp_col.min(), temp_col.max()
        if pd.isna(min_date) or pd.isna(max_date):
            st.warning(f"⚠️ Could not parse `{col_to_filter}` as datetime.")
        else:
            date_start, date_end = st.date_input("Select date range", (min_date.date(), max_date.date()), min_date.date(), max_date.date())
            if date_start and date_end:
                 filtered_df = df[temp_col.dt.date.between(date_start, date_end)]
    else:
        unique_vals = df[col_to_filter].dropna().unique().tolist()
        if not unique_vals:
            st.warning("⚠️ No valid values to filter.")
        else:
            selected_vals = st.multiselect(f"Select values from `{col_to_filter}`:", options=unique_vals, default=unique_vals)
            filtered_df = df[df[col_to_filter].isin(selected_vals)]
            
    st.dataframe(filtered_df, use_container_width=True)

    c1, c2, _ = st.columns([1,2,3])
    with c1:
        if st.button("✅ Apply This Filter"):
            st.session_state.df_clean = filtered_df.copy()
            st.success("✅ Filter applied to the dataset.")
            st.rerun()
    with c2:
        if st.button("🔄 Reset All Cleaning & Filters"):
            st.session_state.df_clean = st.session_state.df_original.copy()
            st.session_state.adv_filter_reset_key += 1
            st.success("✅ Dataset has been reset to its original state.")
            st.rerun()

# --- Sort Tab ---
with tabs[4]:
    st.subheader("📈 Sort Data")
    sort_col = st.selectbox("Column to sort by", df.columns.tolist())
    ascending = st.checkbox("Sort ascending", value=True)
    if st.button("Sort Data"):
        st.session_state.df_clean = df.sort_values(by=sort_col, ascending=ascending)
        st.success(f"✅ Sorted by `{sort_col}`")
        st.rerun()

    st.markdown("---")
    st.write("### 🧬 Remove Duplicate Rows")
    if st.button("Remove Duplicates"):
        before = len(df)
        st.session_state.df_clean = df.drop_duplicates()
        after = len(st.session_state.df_clean)
        st.success(f"✅ Removed {before - after} duplicate rows")
        st.rerun()

# --- Advanced Filter Tab ---
with tabs[5]:
    st.subheader("🧠 Advanced Multi-Column Filtering")
    st.info("This filter is applied to the current dataset and will modify it for export.")

    adv_df = df.copy()
    adv_filter_key_prefix = f"adv_filter_{st.session_state.adv_filter_reset_key}"

    num_conditions = st.number_input("How many filter conditions?", min_value=1, max_value=5, value=1, key=f"{adv_filter_key_prefix}_num_conditions")
    logic = st.radio("Combine filters using:", ["AND", "OR"], horizontal=True, key=f"{adv_filter_key_prefix}_logic")

    conditions = []
    for i in range(int(num_conditions)):
        st.markdown(f"--- \n ### ➕ Condition #{i+1}")
        col = st.selectbox(f"Choose column", adv_df.columns, key=f"{adv_filter_key_prefix}_col_{i}")
        dtype = adv_df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            min_val, max_val = float(adv_df[col].min()), float(adv_df[col].max())
            if min_val == max_val:
                st.warning(f"⚠️ All values in `{col}` are the same: {min_val}")
                cond = pd.Series([True] * len(adv_df), index=adv_df.index)
            else:
                range_val = st.slider(f"Range for `{col}`", min_val, max_val, (min_val, max_val), key=f"{adv_filter_key_prefix}_range_{i}")
                cond = adv_df[col].between(range_val[0], range_val[1])

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            temp_col = pd.to_datetime(adv_df[col], errors="coerce")
            min_date, max_date = temp_col.min(), temp_col.max()
            if pd.isna(min_date) or pd.isna(max_date):
                st.warning(f"⚠️ Cannot parse `{col}` as datetime or it contains no valid dates.")
                cond = pd.Series([True] * len(adv_df), index=adv_df.index)
            else:
                start_date, end_date = st.date_input(f"Date range for `{col}`", (min_date.date(), max_date.date()), key=f"{adv_filter_key_prefix}_date_{i}")
                if start_date and end_date:
                    cond = temp_col.dt.date.between(start_date, end_date)
                else:
                    cond = pd.Series([True] * len(adv_df), index=adv_df.index)

        else: # Categorical
            values = adv_df[col].dropna().unique().tolist()
            default_selection = values if values else []
            selected = st.multiselect(f"Select values for `{col}`", values, key=f"{adv_filter_key_prefix}_cat_{i}", default=default_selection)
            cond = adv_df[col].isin(selected)

        conditions.append(cond)

    adv_filtered_df = adv_df.copy()
    if conditions:
        combined_mask = conditions[0]
        for c in conditions[1:]:
            combined_mask = (combined_mask & c) if logic == "AND" else (combined_mask | c)
        adv_filtered_df = adv_df[combined_mask]

    st.markdown("---")
    st.subheader("Filtered Data Preview")
    st.dataframe(adv_filtered_df, use_container_width=True)
    st.success(f"✅ Previewing {len(adv_filtered_df)} rows that match your filters.")

    col1, col2, _ = st.columns([1.5, 1.5, 4])
    with col1:
        if st.button("✅ Apply These Advanced Filters"):
            st.session_state.df_clean = adv_filtered_df.copy()
            st.success("✅ Advanced filters applied to the dataset.")
            st.rerun()
    with col2:
        if st.button("🔄 Reset Conditions"):
            st.session_state.adv_filter_reset_key += 1
            st.rerun()

# --- Export Tab ---
with tabs[6]:
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

