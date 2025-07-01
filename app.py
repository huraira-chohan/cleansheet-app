import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from io import StringIO
import dateutil.parser

# --- Streamlit Page Config ---
st.set_page_config(page_title="CleanSheet - All-in-One CSV Cleaner", layout="wide")
st.title("🧹 CleanSheet")
st.caption("An all-in-one, no-code data cleaning assistant")

# --- Sidebar Controls ---
st.sidebar.markdown("### 📦 Load Dataset")
load_sample = st.sidebar.button("📂 Load Sample Titanic Dataset")
uploaded_file = st.sidebar.file_uploader("📤 Or Upload your CSV file", type=["csv"])

# --- Initialize Session State ---
st.session_state.setdefault("active_tab", "📊 Preview")
st.session_state.setdefault("adv_filter_reset_key", 0)
st.session_state.setdefault("df_clean", None)
st.session_state.setdefault("df_original", None)

# --- Load Dataset ---
def load_titanic_sample():
    try:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))
        st.success("✅ Sample Titanic dataset loaded successfully!")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load sample data: {e}")
        return None

def load_uploaded_file(file):
    try:
        df = pd.read_csv(file)
        df.replace(["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"], np.nan, inplace=True)
        st.success("✅ Your dataset was uploaded successfully!")
        return df
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")
        return None

if load_sample:
    df = load_titanic_sample()
    if df is not None:
        st.session_state.df_original = df.copy()
        st.session_state.df_clean = df.copy()

elif uploaded_file:
    df = load_uploaded_file(uploaded_file)
    if df is not None:
        st.session_state.df_original = df.copy()
        st.session_state.df_clean = df.copy()

# --- Guard Clause: No Data Loaded ---
if st.session_state.df_clean is None:
    st.info("📎 Please upload a CSV file or load the sample dataset to get started.")
    st.stop()

df = st.session_state.df_clean.copy()

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
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().sum() >= series.notna().sum() * 0.8:
            return numeric
    except: pass
    try:
        dates = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        if dates.notna().sum() >= series.notna().sum() * 0.8:
            return dates
    except: pass

    cleaned = series.astype(str).str.strip().str.lower()
    unique_vals = cleaned.dropna().unique()
    if len(unique_vals) <= 10:
        return cleaned.replace({
            "yes": "Yes", "y": "Yes", "1": "Yes",
            "no": "No", "n": "No", "0": "No",
            "m": "Male", "f": "Female"
        }).str.title()
    return series

# --- Navigation ---
tab_labels = ["📊 Preview", "🧹 Clean", "🧮 Columns", "🔍 Filter", "📈 Sort", "🧠 Advanced Filter", "⬇️ Export"]
st.radio("Navigation", options=tab_labels, key="active_tab", horizontal=True, label_visibility="collapsed")

# --- 📊 Preview Tab ---
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


# --- 🧹 Clean Tab ---
elif st.session_state.active_tab == "🧹 Clean":
    st.subheader("🧹 Clean Your Dataset")

    df_clean_tab = st.session_state.df_clean.copy()
    if df_clean_tab.empty:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    # --- Column-wise Cleaning ---
    st.markdown("### 🔧 Column-wise Cleaning")
    col = st.selectbox("Select column to clean", df_clean_tab.columns)
    actions = st.multiselect(
        "Select cleaning steps (applied in order)",
        [
            "Remove NaNs", "Fill NaNs with 0", "Fill NaNs with Mean", "Fill NaNs with Median", "Custom Fill",
            "To Lowercase", "To Title Case", "Convert to Numeric", "Auto Clean", "Strip Whitespace",
            "Replace Values", "Remove Duplicates", "Remove Outliers"
        ]
    )

    custom_fill_value = st.text_input("Custom value for NaNs (if selected)", key="custom_fill_val")
    replace_dict = {}
    if "Replace Values" in actions:
        with st.expander("🔄 Replace Values"):
            old_vals = st.text_input("Old values (comma-separated)", key="old_vals")
            new_val = st.text_input("New value to replace them with", key="new_val")
            if old_vals:
                replace_dict = {v.strip(): new_val for v in old_vals.split(",")}

    col1, col2 = st.columns(2)
    col1.write("**Before Cleaning**")
    col1.dataframe(df_clean_tab[[col]].head(10), use_container_width=True)

    # Start from full DataFrame
    cleaned_df = df_clean_tab.copy()

    # Perform selected actions
    for action in actions:# --- 🧹 Clean Tab ---
elif st.session_state.active_tab == "🧹 Clean":
    st.subheader("🧹 Clean Your Dataset")

    if st.session_state.df_clean is None or st.session_state.df_clean.empty:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    df_clean_tab = st.session_state.df_clean.copy()

    # --- Column-wise Cleaning ---
    st.markdown("### 🔧 Column-wise Cleaning")
    col = st.selectbox("Select column to clean", df_clean_tab.columns)

    actions = st.multiselect(
        "Select cleaning steps (applied in order)",
        [
            "Remove NaNs", "Fill NaNs with 0", "Fill NaNs with Mean", "Fill NaNs with Median", "Custom Fill",
            "To Lowercase", "To Title Case", "Convert to Numeric", "Auto Clean", "Strip Whitespace",
            "Replace Values", "Remove Duplicates", "Remove Outliers"
        ]
    )

    custom_fill_value = st.text_input("Custom value for NaNs (if selected)", key="custom_fill_val")

    replace_dict = {}
    if "Replace Values" in actions:
        with st.expander("🔄 Replace Values"):
            old_vals = st.text_input("Old values (comma-separated)", key="old_vals")
            new_val = st.text_input("New value to replace them with", key="new_val")
            if old_vals:
                replace_dict = {v.strip(): new_val for v in old_vals.split(",")}

    col1, col2 = st.columns(2)
    col1.write("**Before Cleaning**")
    col1.dataframe(df_clean_tab[[col]].head(10), use_container_width=True)

    # Work on the original df_clean
    working_df = st.session_state.df_clean.copy()

    for action in actions:
        if action == "Remove NaNs":
            working_df = working_df[working_df[col].notna()]
        elif action == "Fill NaNs with 0":
            working_df[col] = working_df[col].fillna(0)
        elif action == "Fill NaNs with Mean" and pd.api.types.is_numeric_dtype(working_df[col]):
            working_df[col] = working_df[col].fillna(working_df[col].mean())
        elif action == "Fill NaNs with Median" and pd.api.types.is_numeric_dtype(working_df[col]):
            working_df[col] = working_df[col].fillna(working_df[col].median())
        elif action == "Custom Fill":
            working_df[col] = working_df[col].fillna(custom_fill_value)
        elif action == "To Lowercase":
            working_df[col] = working_df[col].astype(str).str.lower()
        elif action == "To Title Case":
            working_df[col] = working_df[col].astype(str).str.title()
        elif action == "Convert to Numeric":
            working_df[col] = working_df[col].apply(convert_to_numeric)
        elif action == "Strip Whitespace":
            working_df[col] = working_df[col].astype(str).str.strip()
        elif action == "Auto Clean":
            working_df[col] = auto_clean_column(working_df[col])
        elif action == "Replace Values" and replace_dict:
            working_df[col] = working_df[col].replace(replace_dict)
        elif action == "Remove Duplicates":
            working_df = working_df.drop_duplicates(subset=[col])
        elif action == "Remove Outliers" and pd.api.types.is_numeric_dtype(working_df[col]):
            mean_val = working_df[col].mean()
            std_val = working_df[col].std()
            working_df = working_df[(working_df[col] - mean_val).abs() <= 3 * std_val]

    col2.write("**After Cleaning**")
    col2.dataframe(working_df[[col]].head(10), use_container_width=True)

    if st.button("✅ Apply Cleaning"):
        st.session_state.df_clean = working_df.copy()
        st.success("✅ Cleaning applied and stored in session.")
        st.rerun()

    if st.button("🔄 Reset Column Cleaning"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ Dataset reset to original.")
        st.rerun()

    # --- Bulk Cleaning ---
    st.markdown("---")
    st.subheader("🧹 Bulk Column Cleaning")
    for c in df_clean_tab.columns:
        with st.expander(f"Column: `{c}`"):
            clean_opt = st.selectbox(
                f"Cleaning for `{c}`",
                ["None", "Text Normalize", "Convert to Numeric"],
                key=f"clean_{c}"
            )
            fill_opt = st.selectbox(
                f"NaN Handling for `{c}`",
                ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"],
                key=f"fill_{c}"
            )

            if st.button(f"Apply to `{c}`", key=f"apply_{c}"):
                bulk_df = st.session_state.df_clean.copy()

                if clean_opt == "Text Normalize":
                    bulk_df[c] = bulk_df[c].apply(clean_text)
                elif clean_opt == "Convert to Numeric":
                    bulk_df[c] = bulk_df[c].apply(convert_to_numeric)

                if fill_opt == "Drop Rows":
                    bulk_df = bulk_df[bulk_df[c].notna()]
                elif fill_opt == "Fill with Mean" and pd.api.types.is_numeric_dtype(bulk_df[c]):
                    bulk_df[c] = bulk_df[c].fillna(bulk_df[c].mean())
                elif fill_opt == "Fill with Median" and pd.api.types.is_numeric_dtype(bulk_df[c]):
                    bulk_df[c] = bulk_df[c].fillna(bulk_df[c].median())
                elif fill_opt == "Fill with Mode":
                    try:
                        bulk_df[c] = bulk_df[c].fillna(bulk_df[c].mode().iloc[0])
                    except:
                        pass

                st.session_state.df_clean = bulk_df.copy()
                st.success(f"✅ Bulk cleaning applied to `{c}`")
                st.rerun()

    st.markdown("---")
    if st.button("🔄 Reset Bulk Cleaning"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ Dataset reset to original.")
        st.rerun()


# --- 🧮 Columns Tab ---
elif st.session_state.active_tab == "🧮 Columns":
    st.subheader("🧮 Manage Columns")

    col1, col2 = st.columns(2)

    # --- Drop Columns ---
    with col1:
        st.write("### 🗑 Drop Columns")
        drop_cols = st.multiselect("Select columns to drop", df.columns.tolist())

        if st.button("Drop Selected Columns"):
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
                st.session_state.df_clean = df
                st.success("✅ Dropped selected columns")
                st.rerun()
            else:
                st.warning("⚠️ Please select columns to drop.")

    # --- Rename Column ---
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
            else:
                st.warning("⚠️ Please enter a new column name.")

    # --- Merge Columns ---
    st.markdown("### ➕ Merge Columns")
    merge_cols = st.multiselect("Select 2 or more columns to merge", df.columns.tolist(), key="merge_cols")
    merge_name = st.text_input("Merged column name", key="merge_name")
    sep = st.text_input("Separator (e.g. space, comma)", value=" ")

    if st.button("Merge Columns"):
        if len(merge_cols) >= 2 and merge_name.strip():
            df[merge_name] = df[merge_cols].astype(str).agg(sep.join, axis=1)
            st.session_state.df_clean = df
            st.success(f"✅ Created merged column `{merge_name}`")
            st.rerun()
        else:
            st.warning("⚠️ Please select at least 2 columns and enter a new column name.")

    # --- Reset Columns Tab ---
    st.markdown("---")
    if st.button("🔄 Reset Columns Tab"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ Dataset reset to original.")
        st.rerun()


# --- 🔍 Filter Tab ---
elif st.session_state.active_tab == "🔍 Filter":
    st.subheader("🔍 Filter Rows (temporary view only)")

    df_to_filter = st.session_state.get("df_clean", pd.DataFrame()).copy()
    if df_to_filter.empty:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    col_to_filter = st.selectbox("Choose column to filter", df_to_filter.columns.tolist())

    filtered_df = df_to_filter.copy()

    # --- Numeric Filtering ---
    if pd.api.types.is_numeric_dtype(df_to_filter[col_to_filter]):
        st.write(f"📏 Numeric Range Filter for `{col_to_filter}`")

        min_val = float(df_to_filter[col_to_filter].min())
        max_val = float(df_to_filter[col_to_filter].max())

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

    # --- Date Filtering ---
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

    # --- Categorical Filtering ---
    else:
        st.write(f"🔠 Categorical Filter for `{col_to_filter}`")

        unique_vals = df_to_filter[col_to_filter].dropna().unique().tolist()
        selected_vals = st.multiselect(
            f"Select values to include from `{col_to_filter}`:",
            options=unique_vals,
            default=unique_vals
        )

        if selected_vals:
            filtered_df = df_to_filter[df_to_filter[col_to_filter].isin(selected_vals)]
        else:
            filtered_df = df_to_filter.head(0)  # Empty DataFrame if nothing selected

    # --- Display Result ---
    st.markdown("### 👁 Filtered View")
    st.dataframe(filtered_df, use_container_width=True)

    # --- Apply Filter ---
    if st.button("✅ Apply This Filter to Dataset"):
        st.session_state.df_clean = filtered_df.copy()
        st.success("✅ Filter applied to working dataset.")
        st.rerun()

    # --- Reset Filter Tab ---
    st.markdown("---")
    if st.button("🔄 Reset Filters Tab"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ Dataset reset to original.")
        st.rerun()


# --- 📈 Sort Tab ---
elif st.session_state.active_tab == "📈 Sort":
    st.subheader("📈 Sort Data")

    if df.empty:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    # --- Sorting ---
    sort_col = st.selectbox("Column to sort by", df.columns.tolist())
    ascending = st.checkbox("Sort ascending", value=True)

    if st.button("Sort Data"):
        df_sorted = df.sort_values(by=sort_col, ascending=ascending)
        st.session_state.df_clean = df_sorted
        st.success(f"✅ Sorted by `{sort_col}` ({'ascending' if ascending else 'descending'})")
        st.rerun()

    # --- Remove Duplicates ---
    if st.button("Remove Duplicates"):
        df_no_duplicates = df.drop_duplicates()
        st.session_state.df_clean = df_no_duplicates
        st.success("✅ Duplicate rows removed")
        st.rerun()

    # --- Reset Sort Tab ---
    st.markdown("---")
    if st.button("🔄 Reset Sort Tab"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ Dataset reset to original.")
        st.rerun()

# --- 🧠 Advanced Filter Tab ---
elif st.session_state.active_tab == "🧠 Advanced Filter":
    st.subheader("🧠 Advanced Multi-Column Filtering")

    adv_df = st.session_state.get("df_clean", pd.DataFrame()).copy()
    if adv_df.empty:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    # Unique key prefix for component state resets
    filter_key_prefix = f"adv_filter_{st.session_state.adv_filter_reset_key}"

    # --- Number of Conditions and Logic Type ---
    num_conditions = st.number_input(
        "How many filter conditions?",
        min_value=1,
        max_value=5,
        value=1,
        key=f"{filter_key_prefix}_num_conditions"
    )

    logic = st.radio(
        "Combine filters using:",
        ["AND", "OR"],
        horizontal=True,
        key=f"{filter_key_prefix}_logic"
    )

    conditions = []

    # --- Build Conditions ---
    for i in range(int(num_conditions)):
        col_key = f"{filter_key_prefix}_col_{i}"
        col = st.selectbox(f"Column for condition #{i+1}", adv_df.columns, key=col_key)

        if pd.api.types.is_numeric_dtype(adv_df[col]):
            min_val = float(adv_df[col].min())
            max_val = float(adv_df[col].max())
            range_val = st.slider(
                f"Range for `{col}`",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                key=f"{filter_key_prefix}_range_{i}"
            )
            cond = adv_df[col].between(range_val[0], range_val[1])

        elif pd.api.types.is_datetime64_any_dtype(adv_df[col]) or "date" in col.lower():
            adv_df[col] = pd.to_datetime(adv_df[col], errors="coerce")
            min_date = adv_df[col].min()
            max_date = adv_df[col].max()
            if pd.isnull(min_date) or pd.isnull(max_date):
                st.warning(f"⚠️ Could not convert `{col}` to datetime.")
                cond = pd.Series([True] * len(adv_df), index=adv_df.index)  # No-op filter
            else:
                start_date, end_date = st.date_input(
                    f"Date range for `{col}`",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key=f"{filter_key_prefix}_date_{i}"
                )
                cond = adv_df[col].dt.date.between(start_date, end_date)

        else:
            values = adv_df[col].dropna().unique().tolist()
            selected_vals = st.multiselect(
                f"Values for `{col}`",
                options=values,
                default=values,
                key=f"{filter_key_prefix}_cat_{i}"
            )
            cond = adv_df[col].isin(selected_vals) if selected_vals else pd.Series([False] * len(adv_df), index=adv_df.index)

        conditions.append(cond)

    # --- Combine Conditions ---
    final_mask = conditions[0]
    for cond in conditions[1:]:
        final_mask = final_mask & cond if logic == "AND" else final_mask | cond

    adv_filtered_df = adv_df[final_mask]

    # --- Display and Actions ---
    st.markdown("### 👁 Filtered View")
    st.dataframe(adv_filtered_df, use_container_width=True)

    if st.button("✅ Apply These Advanced Filters"):
        st.session_state.df_clean = adv_filtered_df.copy()
        st.success("✅ Advanced filters applied to working dataset.")
        st.rerun()

    st.markdown("---")
    if st.button("🔄 Reset Advanced Filter Tab"):
        st.session_state.adv_filter_reset_key += 1  # forces rerender of components with new keys
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ Dataset reset to original.")
        st.rerun()

# --- ⬇️ Export Tab ---
elif st.session_state.active_tab == "⬇️ Export":
    st.subheader("⬇️ Export Cleaned CSV")

    export_df = st.session_state.get("df_clean", pd.DataFrame()).copy()

    if export_df.empty:
        st.warning("⚠️ No cleaned data available to export.")
        st.stop()

    # ✅ Show column names
    st.markdown("**🧪 Current Cleaned Columns:**")
    st.write(export_df.columns.tolist())

    # ✅ Show data preview
    st.markdown("### 👁 Preview of Cleaned Data")
    st.dataframe(export_df, use_container_width=True)

    # ✅ Download button
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv_data,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    # ✅ Reset button
    st.markdown("---")
    if st.button("🔄 Reset Export Tab"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ Dataset reset to original.")
        st.rerun()

