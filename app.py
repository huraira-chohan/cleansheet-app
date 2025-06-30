import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import requests
import dateutil.parser

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Preview"

# ADDED: Initialize the key for the advanced filter reset functionality.
if "adv_filter_reset_key" not in st.session_state:
    st.session_state.adv_filter_reset_key = 0

st.set_page_config(page_title="CleanSheet - All-in-One CSV Cleaner", layout="wide")
st.title("🧹 CleanSheet")
st.caption("An all-in-one, no-code data cleaning assistant")
st.sidebar.markdown("### 📦 Load Dataset")

# File uploader + sample loader
load_sample = st.sidebar.button("📂 Load Sample Titanic Dataset")
uploaded_file = st.sidebar.file_uploader("📤 Or Upload your CSV file", type=["csv"])

# Load dataset into session state
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "df_original" not in st.session_state:
    st.session_state.df_original = None

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

# Ensure data is available
if st.session_state.df_clean is None:
    st.info("📎 Please upload a CSV file or load the sample dataset to get started.")
    st.stop()

# Use the session state dataframe going forward
df = st.session_state.df_clean


# --- Helpers ---
def clean_text(x):
    return str(x).strip().title() if pd.notnull(x) else x

def convert_to_numeric(x):
    try:
        return float(re.sub(r"[^0-9.]+", "", str(x)))
    except:
        return np.nan

def auto_clean_column(series):
    # Try to convert to numeric
    try:
        numeric_series = series.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        if numeric_series.notna().sum() >= series.notna().sum() * 0.8:
            return numeric_series
    except:
        pass

    # Try to parse as datetime
    try:
        date_series = series.apply(lambda x: pd.to_datetime(x, errors='coerce', infer_datetime_format=True))
        if date_series.notna().sum() >= series.notna().sum() * 0.8:
            return date_series
    except:
        pass

    # Normalize categorical values (like gender, yes/no)
    unique_vals = series.dropna().astype(str).str.strip().str.lower().unique()
    if len(unique_vals) <= 10:
        return series.astype(str).str.strip().str.lower().replace({
            "yes": "Yes", "y": "Yes", "1": "Yes",
            "no": "No", "n": "No", "0": "No",
            "m": "Male", "f": "Female"
        }).str.title()

    return series

# --- Tabs for navigation ---
tab_labels = ["📊 Preview", "🧹 Clean", "🧮 Columns", "🔍 Filter", "📈 Sort", "🧠 Advanced Filter", "⬇️ Export"]
tabs = st.tabs(tab_labels)
selected_index = next(i for i, tab in enumerate(tab_labels) if tab == st.session_state.active_tab)
tabs[selected_index].select()

# --- Preview Tab ---
with tabs[0]:
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
with tabs[1]:
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

    cleaned_df = df_clean_tab.copy() # Work on a copy for preview

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

    # Show cleaned preview
    preview_col2.markdown("**After Cleaning**")
    preview_col2.write(cleaned_df[[col]].head(10))

    if st.button("Apply Cleaning to Dataset"):
        st.session_state.df_clean = cleaned_df.copy()
        st.rerun()

    st.markdown("---")
    st.subheader("🔎 Clean All Columns")
    columns = st.session_state.df_clean.columns.tolist()

    for col_to_clean in columns:
        with st.expander(f"⚙️ Options for `{col_to_clean}`"):
            clean_opt = st.selectbox(f"Cleaning for `{col_to_clean}`", [
                "None", "Text Normalize", "Convert to Numeric"
            ], key=f"clean_{col_to_clean}")

            fill_na = st.selectbox(f"Missing values in `{col_to_clean}`", [
                "None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"
            ], key=f"na_{col_to_clean}")

            if st.button(f"✅ Apply to `{col_to_clean}`", key=f"apply_{col_to_clean}"):
                temp_df = st.session_state.df_clean.copy()
                if clean_opt == "Text Normalize":
                    temp_df[col_to_clean] = temp_df[col_to_clean].apply(clean_text)
                elif clean_opt == "Convert to Numeric":
                    temp_df[col_to_clean] = temp_df[col_to_clean].apply(convert_to_numeric)

                if fill_na == "Drop Rows":
                    temp_df = temp_df[temp_df[col_to_clean].notna()]
                elif fill_na == "Fill with Mean" and pd.api.types.is_numeric_dtype(temp_df[col_to_clean]):
                    temp_df[col_to_clean].fillna(temp_df[col_to_clean].mean(), inplace=True)
                elif fill_na == "Fill with Median" and pd.api.types.is_numeric_dtype(temp_df[col_to_clean]):
                    temp_df[col_to_clean].fillna(temp_df[col_to_clean].median(), inplace=True)
                elif fill_na == "Fill with Mode":
                    temp_df[col_to_clean].fillna(temp_df[col_to_clean].mode().iloc[0], inplace=True)

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

    st.write("### 🔤 Split Alphanumeric Column")
    split_col = st.selectbox("Select alphanumeric column to split", df.columns.tolist(), key="split_col")
    new_alpha = st.text_input("New column for alphabets", key="alpha_part")
    new_num = st.text_input("New column for numbers", key="num_part")

    if st.button("Split Alphanumeric"):
        if new_alpha and new_num:
            df[new_alpha] = df[split_col].astype(str).apply(lambda x: ''.join(re.findall(r'[A-Za-z]+', x)))
            df[new_num] = df[split_col].astype(str).apply(lambda x: ''.join(re.findall(r'\d+', x)))
            st.session_state.df_clean = df
            st.success(f"✅ Split `{split_col}` into `{new_alpha}` and `{new_num}`")
            st.rerun()


# --- Filter Tab ---
with tabs[3]:
    st.subheader("🔍 Filter Rows (temporary view only)")

    df_to_filter = st.session_state.get("df_clean", pd.DataFrame()).copy()
    col_to_filter = st.selectbox("Choose column to filter", df_to_filter.columns.tolist())
    filtered_df = df_to_filter.copy()

    # Numeric filtering
    if pd.api.types.is_numeric_dtype(df_to_filter[col_to_filter]):
        st.write(f"📏 Numeric Range Filter for `{col_to_filter}`")
        min_val, max_val = float(df_to_filter[col_to_filter].min()), float(df_to_filter[col_to_filter].max())

        if min_val == max_val:
            st.warning(f"⚠️ All values in `{col_to_filter}` are the same: {min_val}")
        else:
            step_val = max((max_val - min_val) / 100, 0.01)
            start, end = st.slider(
                "Select value range",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                step=step_val
            )
            filtered_df = df_to_filter[df_to_filter[col_to_filter].between(start, end)]
            st.info(f"Showing rows where `{col_to_filter}` is between {start:.2f} and {end:.2f}.")

    # Datetime filtering
    elif pd.api.types.is_datetime64_any_dtype(df_to_filter[col_to_filter]) or "date" in col_to_filter.lower():
        st.write(f"🗓 Date Range Filter for `{col_to_filter}`")
        df_to_filter[col_to_filter] = pd.to_datetime(df_to_filter[col_to_filter], errors='coerce')
        min_date = df_to_filter[col_to_filter].min()
        max_date = df_to_filter[col_to_filter].max()

        if pd.isnull(min_date) or pd.isnull(max_date):
            st.warning(f"⚠️ Could not convert `{col_to_filter}` to datetime.")
        else:
            date_start, date_end = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            filtered_df = df_to_filter[df_to_filter[col_to_filter].between(date_start, date_end)]
            st.info(f"Showing rows where `{col_to_filter}` is between {date_start} and {date_end}.")

    # Categorical filtering
    else:
        st.write(f"🔠 Categorical Filter for `{col_to_filter}`")
        unique_vals = df_to_filter[col_to_filter].dropna().unique().tolist()
        if not unique_vals:
            st.warning("⚠️ No valid values to filter.")
        else:
            selected_vals = st.multiselect(
                f"Select values to include from `{col_to_filter}`:",
                options=unique_vals,
                default=unique_vals
            )
            if selected_vals:
                filtered_df = df_to_filter[df_to_filter[col_to_filter].isin(selected_vals)]
                st.info(f"Filtered by selected values in `{col_to_filter}`.")
            else:
                filtered_df = df_to_filter.head(0) # Show empty if nothing selected
                st.caption("No filter applied — showing full dataset.")

    st.dataframe(filtered_df, use_container_width=True)

    if st.button("✅ Apply This Filter to Dataset"):
        st.session_state.df_clean = filtered_df.copy()
        st.success("✅ Filter applied to the dataset.")
        st.rerun()

    if st.button("🔄 Reset All Filters & Cleaning"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ Dataset has been reset to its original state.")
        st.rerun()


# --- Sort Tab ---
with tabs[4]:
    st.subheader("📈 Sort Data")
    st.info("ℹ️ Sorting is applied to the current state of the dataset.")
    sort_col = st.selectbox("Column to sort by", df.columns.tolist())
    ascending = st.checkbox("Sort ascending", value=True)
    if st.button("Sort Data"):
        df = df.sort_values(by=sort_col, ascending=ascending)
        st.session_state.df_clean = df
        st.success(f"✅ Sorted by `{sort_col}`")
        st.rerun()

    st.markdown("---")
    st.write("### 🧬 Remove Duplicate Rows")
    if st.button("Remove Duplicates"):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        st.session_state.df_clean = df
        st.success(f"✅ Removed {before - after} duplicate rows")
        st.rerun()

# --- Advanced Filter Tab (MODIFIED FOR RESET BUTTON) ---
with tabs[5]:
    st.subheader("🧠 Advanced Multi-Column Filtering")
    st.info("This filter is applied to the current dataset and will modify it for export.")

    adv_df = st.session_state.get("df_clean", pd.DataFrame()).copy()
    if adv_df.empty:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    # ADDED: Create a dynamic key prefix using the reset counter
    adv_filter_key_prefix = f"adv_filter_{st.session_state.adv_filter_reset_key}"

    # MODIFIED: Added keys to these widgets
    num_conditions = st.number_input("How many filter conditions?", min_value=1, max_value=5, value=1, key=f"{adv_filter_key_prefix}_num_conditions")
    logic = st.radio("Combine filters using:", ["AND", "OR"], horizontal=True, key=f"{adv_filter_key_prefix}_logic")

    conditions = []
    for i in range(int(num_conditions)):
        st.markdown(f"### ➕ Condition #{i+1}")
        # MODIFIED: Added key prefix to all widgets in the loop
        col = st.selectbox(f"Choose column", adv_df.columns, key=f"{adv_filter_key_prefix}_adv_col_{i}")
        dtype = adv_df[col].dtype

        # Numeric
        if pd.api.types.is_numeric_dtype(dtype):
            min_val, max_val = float(adv_df[col].min()), float(adv_df[col].max())
            if min_val == max_val:
                st.warning(f"⚠️ All values in `{col}` are the same: {min_val}")
                cond = pd.Series([True] * len(adv_df), index=adv_df.index)
            else:
                range_val = st.slider(f"Range for `{col}`", min_val, max_val, (min_val, max_val), key=f"{adv_filter_key_prefix}_adv_range_{i}")
                cond = adv_df[col].between(range_val[0], range_val[1])

        # Datetime
        elif pd.api.types.is_datetime64_any_dtype(dtype) or "date" in col.lower():
            adv_df[col] = pd.to_datetime(adv_df[col], errors="coerce")
            min_date, max_date = adv_df[col].min(), adv_df[col].max()
            if pd.isnull(min_date) or pd.isnull(max_date):
                st.warning(f"⚠️ Cannot parse `{col}` as datetime.")
                cond = pd.Series([True] * len(adv_df), index=adv_df.index)
            else:
                start_date, end_date = st.date_input(f"Date range for `{col}`", (min_date.date(), max_date.date()), key=f"{adv_filter_key_prefix}_adv_date_{i}")
                cond = adv_df[col].dt.date.between(start_date, end_date)

        # Categorical
        else:
            values = adv_df[col].dropna().unique().tolist()
            selected = st.multiselect(f"Select values for `{col}`", values, key=f"{adv_filter_key_prefix}_adv_cat_{i}", default=values)
            cond = adv_df[col].isin(selected) if selected else pd.Series([False] * len(adv_df), index=adv_df.index)

        conditions.append(cond)

    adv_filtered_df = adv_df.copy()
    if conditions:
        combined_mask = conditions[0]
        for c in conditions[1:]:
            combined_mask = combined_mask & c if logic == "AND" else combined_mask | c
        adv_filtered_df = adv_df[combined_mask]

    st.dataframe(adv_filtered_df, use_container_width=True)
    st.success(f"✅ Previewing {len(adv_filtered_df)} rows that match your filters.")

    # MODIFIED: Using columns for better button layout
    col1, col2, _ = st.columns([1.5, 1.5, 4])
    with col1:
        if st.button("✅ Apply These Advanced Filters"):
            st.session_state.df_clean = adv_filtered_df.copy()
            st.success("✅ Filters applied to the dataset.")
            st.rerun()
    with col2:
        # ADDED: The Reset Conditions button and its logic
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
