# --- 1. IMPORTS ---
# Group imports: standard library first, followed by third-party packages.
from io import StringIO
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st
# dateutil.parser is imported below for future use if needed

# --- 2. CONSTANTS ---
# Centralize configuration for easier maintenance.
APP_TITLE = "CleanSheet - All-in-One CSV Cleaner"
TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

NULL_VALUE_REPLACEMENTS = [
    "", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"
]

TAB_LABELS = [
    "📊 Preview", "🧹 Clean", "🧮 Columns", "🔍 Filter", "📈 Sort",
    "🧠 Advanced Filter", "⬇️ Export"
]

st.radio(
    "Navigation", options=TAB_LABELS, key="active_tab",
    horizontal=True, label_visibility="collapsed"
)


# --- 3. STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("🧹 CleanSheet")
st.caption("An all-in-one, no-code data cleaning assistant")

# --- 4. SESSION STATE INITIALIZATION ---
def initialize_session_state():
    """Initializes all required session state variables if they don't already exist."""
    if "df_clean" not in st.session_state:
        st.session_state.df_clean = None
        st.session_state.df_original = None
        st.session_state.active_tab = "📊 Preview"
        st.session_state.adv_filter_reset_key = 0
        st.session_state.staged_ops = {
            "clean": [],
            "columns": {"drop": [], "rename": {}},
            "filter": {"simple": [], "advanced": ""},
            "sort": {},
        }

initialize_session_state()

# --- 5. DATA LOADING FUNCTIONS ---
@st.cache_data(show_spinner="Downloading sample data...")
def load_titanic_sample():
    """Downloads and caches the Titanic dataset from a specified URL."""
    try:
        response = requests.get(TITANIC_URL)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df.replace(NULL_VALUE_REPLACEMENTS, np.nan, inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error loading sample data: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Failed to process sample data: {e}")
        return None

def load_uploaded_file(file):
    """Loads a user-uploaded CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file)
        df.replace(NULL_VALUE_REPLACEMENTS, np.nan, inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Failed to read or process the uploaded file: {e}")
        return None

# --- 6. SIDEBAR AND DATA LOADING LOGIC ---

# --- Sidebar Controls ---
st.sidebar.markdown("### 📦 Load Dataset")
load_sample_clicked = st.sidebar.button("📂 Load Sample Titanic Dataset")
uploaded_file = st.sidebar.file_uploader("📤 Or Upload your CSV file", type=["csv"])

# --- Handle Data Loading ---
newly_loaded_df = None
if load_sample_clicked:
    newly_loaded_df = load_titanic_sample()
    if newly_loaded_df is not None:
        st.success("✅ Sample Titanic dataset loaded successfully!")

elif uploaded_file is not None:
    newly_loaded_df = load_uploaded_file(uploaded_file)
    if newly_loaded_df is not None:
        st.success("✅ Your dataset was uploaded successfully!")

# If a new dataframe was loaded, update the session state.
if newly_loaded_df is not None:
    st.session_state.df_original = newly_loaded_df.copy()
    st.session_state.df_clean = newly_loaded_df.copy()
    st.session_state.active_tab = "📊 Preview"
    
    # When new data is loaded, reset all staged operations.
    st.session_state.staged_ops = {
        "clean": [], "columns": {"drop": [], "rename": {}},
        "filter": {"simple": [], "advanced": ""}, "sort": {},
    }
    st.rerun()

# --- 7. DATA AVAILABILITY GUARD ---
if st.session_state.df_clean is None:
    st.info("📎 Please upload a CSV file or load the sample dataset to get started.")
    st.stop()

df_display = st.session_state.df_clean.copy()

# --- Navigation UI ---
st.radio(
    "Navigation",
    options=TAB_LABELS,
    key="active_tab",
    horizontal=True,
    label_visibility="collapsed"
)

# --- 8. HELPER FUNCTIONS ---
def convert_to_numeric(x):
    """Attempt to convert a value to numeric, removing non-numeric characters."""
    if pd.isnull(x):
        return np.nan
    try:
        return float(re.sub(r"[^0-9.]+", "", str(x)))
    except (ValueError, TypeError):
        return np.nan

# --- 9. UI: NAVIGATION AND TABS ---
st.radio(
    "Navigation", options=TAB_LABELS, key="active_tab",
    horizontal=True, label_visibility="collapsed"
)

if st.session_state.active_tab == "📊 Preview":
    st.subheader("🔎 Dataset Preview")

    VIEW_OPTIONS = {"Top 5": 5, "Top 50": 50, "All": None}
    view_opt = st.radio("How much data to show?", VIEW_OPTIONS.keys(), horizontal=True)

    rows_to_show = VIEW_OPTIONS[view_opt]
    if rows_to_show is None:
        st.dataframe(df_display, use_container_width=True)
    else:
        st.dataframe(df_display.head(rows_to_show), use_container_width=True)

    st.write("#### ℹ️ Column Summary")
    summary_df = df_display.describe(include='all').T.fillna("N/A")
    st.dataframe(summary_df, use_container_width=True)

    if st.button("🔄 Reset Dataset to Original"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("🔄 Dataset has been reset!")
        import time
        time.sleep(1)
        st.rerun()

        
# --- 🧹 Clean Tab ---
elif st.session_state.active_tab == "🧹 Clean":
    st.subheader("🧹 Clean Your Dataset")

    if st.session_state.df_clean is None or st.session_state.df_clean.empty:
        st.warning("⚠️ Please load a dataset to begin cleaning.")
        st.stop()

    def apply_action(df, column, action, params=None):
    if params is None:
        params = {}

        df_out = df.copy()
        if column not in df_out.columns:
            st.error(f"Column '{column}' not found. It might have been dropped or renamed.")
            return df_out

        if action == "Remove NaNs":
            df_out.dropna(subset=[column], inplace=True)
        elif action == "Fill NaNs with 0":
            df_out[column].fillna(0, inplace=True)
        elif action == "Fill NaNs with Mean":
            if pd.api.types.is_numeric_dtype(df_out[column]):
                df_out[column].fillna(df_out[column].mean(), inplace=True)
            else:
                st.warning(f"Column '{column}' is not numeric. Cannot fill with mean.")
        elif action == "Fill NaNs with Median":
            if pd.api.types.is_numeric_dtype(df_out[column]):
                df_out[column].fillna(df_out[column].median(), inplace=True)
            else:
                st.warning(f"Column '{column}' is not numeric. Cannot fill with median.")
        elif action == "Custom Fill":
            df_out[column].fillna(params.get('custom_value', ''), inplace=True)
        elif action == "To Lowercase":
            df_out[column] = df_out[column].astype(str).str.lower()
        elif action == "To Title Case":
            df_out[column] = df_out[column].apply(clean_text)
        elif action == "Convert to Numeric":
            df_out[column] = df_out[column].apply(convert_to_numeric)
        elif action == "Strip Whitespace":
            df_out[column] = df_out[column].astype(str).str.strip()
        elif action == "Auto Clean":
            df_out[column] = auto_clean_column(df_out[column])
        elif action == "Replace Values" and params.get('replace_dict'):
            df_out[column].replace(params['replace_dict'], inplace=True)
        elif action == "Remove Duplicates":
            df_out.drop_duplicates(subset=[column], inplace=True)
        elif action == "Remove Outliers":
            if pd.api.types.is_numeric_dtype(df_out[column]):
                mean = df_out[column].mean()
                std = df_out[column].std()
                df_out = df_out[df_out[column].between(mean - 3 * std, mean + 3 * std)]
            else:
                st.warning(f"Column '{column}' is not numeric. Cannot remove outliers.")
        return df_out

    st.markdown("### 🔧 Interactive Column Cleaning")
    df_preview = st.session_state.df_clean.copy()
    col_to_clean = st.selectbox("Select column to clean", df_preview.columns, key="interactive_clean_col")

    actions = st.multiselect(
        "Select cleaning steps (applied in order):",
        [
            "Remove NaNs", "Fill NaNs with 0", "Fill NaNs with Mean", "Fill NaNs with Median", "Custom Fill",
            "To Lowercase", "To Title Case", "Convert to Numeric", "Auto Clean", "Strip Whitespace",
            "Replace Values", "Remove Duplicates", "Remove Outliers"
        ],
        key="interactive_clean_actions"
    )

    action_params = {}
    if "Custom Fill" in actions:
        action_params['custom_value'] = st.text_input("Value for 'Custom Fill'", key="custom_fill_val")
    if "Replace Values" in actions:
        with st.expander("🔄 Configure 'Replace Values'"):
            old_vals = st.text_input("Values to replace (comma-separated)", key="old_vals")
            new_val = st.text_input("New value to replace them with", key="new_val")
            if old_vals:
                action_params['replace_dict'] = {v.strip(): new_val for v in old_vals.split(",")}

    df_after_preview = df_preview.copy()
    if actions:
        for action in actions:
            df_after_preview = apply_action(df_after_preview, col_to_clean, action, action_params)

    col1, col2 = st.columns(2)
    col1.write("**Before Cleaning** (Top 10 rows)")
    col1.dataframe(df_preview[[col_to_clean]].head(10), use_container_width=True)
    col2.write("**After Cleaning Preview** (Top 10 rows)")
    col2.dataframe(df_after_preview[[col_to_clean]].head(10), use_container_width=True)

    if st.button("✅ Apply These Changes", disabled=not actions):
        st.session_state.df_clean = df_after_preview.copy()
        st.success(f"✅ Changes applied successfully to column '{col_to_clean}'.")
        import time
        time.sleep(1)
        st.rerun()

  # --- 2. BULK COLUMN CLEANING ---
    st.markdown("---")
    st.markdown("### 🧹 Bulk Column Cleaning")

    for col in st.session_state.df_clean.columns:
        with st.expander(f"Clean Column: `{col}`"):
            col1, col2, col3 = st.columns([2, 2, 1])

            clean_opt = col1.selectbox(
                f"Transformation for `{col}`",
                ["None", "Text Normalize", "Convert to Numeric"],
                key=f"clean_{col}"
            )
            fill_opt = col2.selectbox(
                f"NaN Handling for `{col}`",
                ["None", "Drop Rows with NaNs", "Fill with Mean", "Fill with Median", "Fill with Mode"],
                key=f"fill_{col}"
            )

            if col3.button(f"Apply to `{col}`", key=f"apply_bulk_{col}"):
                df_bulk_cleaned = st.session_state.df_clean.copy()

                if clean_opt == "Text Normalize":
                    df_bulk_cleaned = apply_action(df_bulk_cleaned, col, "To Title Case", {})
                elif clean_opt == "Convert to Numeric":
                    df_bulk_cleaned = apply_action(df_bulk_cleaned, col, "Convert to Numeric", {})

                if fill_opt == "Drop Rows with NaNs":
                    df_bulk_cleaned = apply_action(df_bulk_cleaned, col, "Remove NaNs", {})
                elif fill_opt == "Fill with Mean":
                    df_bulk_cleaned = apply_action(df_bulk_cleaned, col, "Fill NaNs with Mean", {})
                elif fill_opt == "Fill with Median":
                    df_bulk_cleaned = apply_action(df_bulk_cleaned, col, "Fill NaNs with Median", {})
                elif fill_opt == "Fill with Mode":
                    if not df_bulk_cleaned[col].empty and df_bulk_cleaned[col].notna().any():
                        try:
                            mode_val = df_bulk_cleaned[col].mode().iloc[0]
                            df_bulk_cleaned[col].fillna(mode_val, inplace=True)
                        except IndexError:
                            st.warning(f"Could not calculate a mode for column `{col}`.")
                    else:
                        st.warning(f"Column `{col}` is empty or all NaN; cannot fill with mode.")

                st.session_state.df_clean = df_bulk_cleaned.copy()
                st.success(f"✅ Bulk cleaning applied to `{col}`.")
                import time
                time.sleep(1)
                st.rerun()

    # --- 3. RESET DATASET ---
    st.markdown("---")
    st.markdown("### 🔄 Reset")
    if st.button("Reset ALL Cleaning to Original Dataset", use_container_width=True, type="primary"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ Dataset has been fully reset to its original state.")
        import time
        time.sleep(2)
        st.rerun()

# --- 🧮 Columns Tab ---
elif st.session_state.active_tab == "🧮 Columns":
    st.subheader("🧮 Manage Columns (Drop, Rename, Merge)")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please load a dataset to begin.")
        st.stop()

    working_df = st.session_state.df_clean.copy()
    available_columns = working_df.columns.tolist()

    col1, col2 = st.columns(2)

    # --- 1. Drop Columns ---
    with col1:
        st.markdown("### 🗑️ Drop Columns")
        cols_to_drop = st.multiselect("Select columns to drop:", options=available_columns, key="cols_to_drop")

        if st.button("Drop Selected Columns", key="drop_btn"):
            if not cols_to_drop:
                st.warning("⚠️ Please select one or more columns to drop.")
            else:
                df_after_drop = working_df.drop(columns=cols_to_drop)
                st.session_state.df_clean = df_after_drop
                st.success(f"✅ Dropped columns: {', '.join(cols_to_drop)}")
                import time
                time.sleep(1)
                st.rerun()

    # --- 2. Rename Column ---
    with col2:
        st.markdown("### ✏️ Rename Column")
        col_to_rename = st.selectbox("Select column to rename:", options=available_columns, key="col_to_rename")
        new_col_name = st.text_input("Enter new column name:", key="new_col_name").strip()

        if st.button("Rename Column", key="rename_btn"):
            if not new_col_name:
                st.warning("⚠️ New column name cannot be empty.")
            elif new_col_name == col_to_rename:
                st.info("ℹ️ The new name is the same as the old name. No changes made.")
            elif new_col_name in available_columns:
                st.error(f"❌ A column named '{new_col_name}' already exists. Please choose a unique name.")
            else:
                working_df.rename(columns={col_to_rename: new_col_name}, inplace=True)
                st.session_state.df_clean = working_df
                st.success(f"✅ Renamed '{col_to_rename}' to '{new_col_name}'.")
                import time
                time.sleep(1)
                st.rerun()

    # --- 3. Merge Columns ---
    st.markdown("---")
    st.markdown("### ➕ Merge Columns")

    cols_to_merge = st.multiselect("Select 2 or more columns to merge:", options=available_columns, key="cols_to_merge")
    merged_col_name = st.text_input("Enter name for the new merged column:", key="merged_col_name").strip()
    separator = st.text_input("Separator to use between values:", value=" ", key="separator")

    if st.button("Merge Selected Columns", key="merge_btn"):
        if len(cols_to_merge) < 2:
            st.warning("⚠️ Please select at least two columns to merge.")
        elif not merged_col_name:
            st.warning("⚠️ Please provide a name for the new merged column.")
        elif merged_col_name in available_columns:
            st.error(f"❌ A column named '{merged_col_name}' already exists. Please choose a unique name or drop the existing one first.")
        else:
            working_df[merged_col_name] = working_df[cols_to_merge].astype(str).agg(separator.join, axis=1)
            st.session_state.df_clean = working_df
            st.success(f"✅ Merged {len(cols_to_merge)} columns into '{merged_col_name}'.")
            import time
            time.sleep(1)
            st.rerun()

    # --- 4. Reset Logic ---
    st.markdown("---")
    st.markdown("### 🔄 Reset")
    if st.button("Reset All Column Changes to Original", use_container_width=True, type="primary"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ All column changes have been reverted. Dataset is reset to its original structure.")
        import time
        time.sleep(2)
        st.rerun()

# --- 🔍 Filter Tab ---
elif st.session_state.active_tab == "🔍 Filter":
    st.subheader("🔍 Filter and Preview Data")
    st.info(
        "Use the controls below to temporarily filter your data. "
        "The view will update instantly. You can then choose to permanently apply this "
        "filtered view to your working dataset."
    )

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please load a dataset to begin.")
        st.stop()

    working_df = st.session_state.df_clean.copy()
    col_to_filter = st.selectbox("1. Choose a column to filter by:", working_df.columns)

    def render_numeric_filter(df, column):
        st.write(f"**2. Set numeric range for `{column}`**")
        series = df[column].dropna()
        if series.empty:
            st.warning(f"Column `{column}` contains no numeric data to filter.")
            return df
        min_val, max_val = float(series.min()), float(series.max())
        if min_val == max_val:
            st.info(f"ℹ️ All values in `{column}` are the same: {min_val}")
            return df
        step = (max_val - min_val) / 100
        start, end = st.slider(
            "Select value range:",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=step if step > 0 else 0.01
        )
        return df[df[column].between(start, end)]

    def render_date_filter(df, column):
        st.write(f"**2. Set date range for `{column}`**")
        date_series = pd.to_datetime(df[column], errors='coerce')
        if date_series.isna().all():
            st.error(f"❌ Could not convert `{column}` to a valid date format.")
            return df
        min_date, max_date = date_series.min().to_pydatetime(), date_series.max().to_pydatetime()
        selected_start, selected_end = st.date_input(
            "Select date range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        start_ts = pd.to_datetime(selected_start)
        end_ts = pd.to_datetime(selected_end)
        return df[date_series.between(start_ts, end_ts)]

    def render_categorical_filter(df, column):
        st.write(f"**2. Select categories from `{column}`**")
        unique_vals = sorted(df[column].dropna().unique().tolist())
        if not unique_vals:
            st.warning(f"Column `{column}` contains no values to filter.")
            return df
        selected_vals = st.multiselect(
            "Select values to include:",
            options=unique_vals,
            default=unique_vals
        )
        if not selected_vals:
            return df.head(0)
        return df[df[column].isin(selected_vals)]

    column_type = working_df[col_to_filter].dtype

    if pd.api.types.is_numeric_dtype(column_type):
        filtered_df = render_numeric_filter(working_df, col_to_filter)
    elif pd.api.types.is_datetime64_any_dtype(column_type) or "date" in col_to_filter.lower():
        filtered_df = render_date_filter(working_df, col_to_filter)
    else:
        filtered_df = render_categorical_filter(working_df, col_to_filter)

    st.markdown("### 👁️ Filtered View")
    st.dataframe(filtered_df, use_container_width=True)
    st.write(f"Showing **{len(filtered_df)}** of **{len(working_df)}** rows.")

    col1, col2 = st.columns(2)

    is_filtered = len(filtered_df) != len(working_df)
    col1.button(
        "✅ Apply This Filter to Dataset",
        key="apply_filter_btn",
        disabled=not is_filtered,
        help="Applies the current filtered view to the main dataset. Disabled if view is unchanged."
    )

    col2.button(
        "🔄 Reset All Filters and Data to Original",
        key="reset_filter_btn",
        type="primary",
        help="Reverts the entire dataset back to its original state, discarding all changes."
    )

    if st.session_state.apply_filter_btn and is_filtered:
        st.session_state.df_clean = filtered_df.copy()
        st.success("✅ Filter has been permanently applied to the dataset.")
        import time
        time.sleep(1)
        st.rerun()

    if st.session_state.reset_filter_btn:
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ All filters cleared. Dataset reset to original state.")
        import time
        time.sleep(2)
        st.rerun()


# --- 📈 Sort Tab ---
elif st.session_state.active_tab == "📈 Sort":
    st.subheader("📈 Sort Data and Manage Duplicates")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please load a dataset to begin.")
        st.stop()

    working_df = st.session_state.df_clean.copy()
    available_columns = working_df.columns.tolist()

    st.markdown("### 📊 Sort by One or More Columns")
    st.info(
        "You can sort by multiple columns. The data will be sorted by the first column, "
        "then the second, and so on."
    )

    with st.container(border=True):
        cols_to_sort = st.multiselect(
            "Select column(s) to sort by (in order of priority):",
            options=available_columns,
            key="sort_cols"
        )

        ascending_flags = []
        if cols_to_sort:
            st.write("**Sort Order:**")
            cols = st.columns(len(cols_to_sort))
            for i, col in enumerate(cols_to_sort):
                with cols[i]:
                    ascending_flags.append(
                        st.checkbox(f"`{col}` Ascending", value=True, key=f"sort_asc_{col}")
                    )

    if st.button("Apply Sort", key="apply_sort_btn", disabled=not cols_to_sort):
        df_sorted = working_df.sort_values(by=cols_to_sort, ascending=ascending_flags)
        sort_desc = ", ".join([f"`{c}` ({'ASC' if a else 'DESC'})" for c, a in zip(cols_to_sort, ascending_flags)])
        st.session_state.df_clean = df_sorted
        st.success(f"✅ Dataset sorted by: {sort_desc}")
        import time
        time.sleep(1)
        st.rerun()

    st.markdown("---")
    st.markdown("### 🗑️ Remove Duplicate Rows")

    with st.container(border=True):
        st.write(
            "This will remove rows where **all** column values are identical to another row. "
            "To remove duplicates based on a specific column, use the 'Remove Duplicates' "
            "action in the **🧹 Clean** tab."
        )

        num_duplicates = working_df.duplicated().sum()

        if num_duplicates > 0:
            st.info(f"ℹ️ Found **{num_duplicates}** duplicate row(s).")
            if st.button("Remove All Duplicate Rows", key="remove_duplicates_btn"):
                df_no_duplicates = working_df.drop_duplicates()
                st.session_state.df_clean = df_no_duplicates
                st.success(f"✅ Removed {num_duplicates} duplicate row(s).")
                import time
                time.sleep(1)
                st.rerun()
        else:
            st.info("✅ No duplicate rows found in the dataset.")

    st.markdown("---")
    st.markdown("### 🔄 Reset")
    if st.button("Reset All Sorting to Original Order", use_container_width=True, type="primary"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ All sorting and duplicate removal has been reverted. Dataset is reset to its original order.")
        import time
        time.sleep(2)
        st.rerun()

# --- 🧠 Advanced Filter Tab ---
elif st.session_state.active_tab == "🧠 Advanced Filter":
    st.subheader("🧠 Advanced Multi-Condition Filtering")
    st.info(
        "Build a set of conditions to filter your data. The view below will update live. "
        "You can then choose to permanently apply this view to your dataset."
    )

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please load a dataset to begin.")
        st.stop()

    adv_df = st.session_state.df_clean.copy()

    if "adv_filter_reset_key" not in st.session_state:
        st.session_state.adv_filter_reset_key = 0
    filter_key_prefix = f"adv_filter_{st.session_state.adv_filter_reset_key}"

    def create_numeric_condition(df, column, key):
        series = df[column].dropna()
        if series.empty:
            st.warning(f"Column `{column}` has no numeric data to filter.")
            return None
        min_val, max_val = float(series.min()), float(series.max())
        if min_val == max_val:
            st.info(f"All values in `{column}` are {min_val}.")
            return df[column] == min_val
        start, end = st.slider(f"Range for `{column}`", min_val, max_val, (min_val, max_val), key=key)
        return df[column].between(start, end)

    def create_date_condition(df, column, key):
        date_series = pd.to_datetime(df[column], errors='coerce')
        valid_dates = date_series.dropna()
        if valid_dates.empty:
            st.warning(f"Column `{column}` has no valid date data to filter.")
            return None
        min_date, max_date = valid_dates.min(), valid_dates.max()
        start, end = st.date_input(
            f"Date range for `{column}`",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            key=key
        )
        return date_series.between(pd.to_datetime(start), pd.to_datetime(end))

    def create_categorical_condition(df, column, key):
        unique_vals = sorted(df[column].dropna().unique().tolist())
        if not unique_vals:
            st.warning(f"Column `{column}` has no categorical data to filter.")
            return None
        selected = st.multiselect(f"Values for `{column}`", unique_vals, default=unique_vals, key=key)
        if not selected:
            return pd.Series([False] * len(df), index=df.index)
        return df[column].isin(selected)

    with st.expander("🛠️ Configure Filter Conditions", expanded=True):
        col1, col2 = st.columns(2)
        num_conditions = col1.number_input("Number of filter conditions:", 1, 5, 1, key=f"{filter_key_prefix}_num")
        logic = col2.radio("Combine conditions using:", ["AND", "OR"], horizontal=True, key=f"{filter_key_prefix}_logic")

        conditions = []
        for i in range(int(num_conditions)):
            st.markdown("---")
            st.markdown(f"**Condition #{i+1}**")
            col_name = st.selectbox("Column", adv_df.columns, key=f"{filter_key_prefix}_col_{i}")
            col_type = adv_df[col_name].dtype

            condition_mask = None
            if pd.api.types.is_numeric_dtype(col_type):
                condition_mask = create_numeric_condition(adv_df, col_name, key=f"{filter_key_prefix}_num_{i}")
            elif pd.api.types.is_datetime64_any_dtype(col_type) or "date" in col_name.lower():
                condition_mask = create_date_condition(adv_df, col_name, key=f"{filter_key_prefix}_date_{i}")
            else:
                condition_mask = create_categorical_condition(adv_df, col_name, key=f"{filter_key_prefix}_cat_{i}")

            if condition_mask is not None:
                conditions.append(condition_mask)

    if not conditions:
        final_mask = pd.Series([True] * len(adv_df), index=adv_df.index)
    else:
        from functools import reduce
        import operator
        combiner = operator.and_ if logic == "AND" else operator.or_
        final_mask = reduce(combiner, conditions)

    adv_filtered_df = adv_df[final_mask]

    st.markdown("### 👁️ Filtered View")
    st.dataframe(adv_filtered_df, use_container_width=True)
    st.write(f"Showing **{len(adv_filtered_df)}** of **{len(adv_df)}** rows.")

    is_filtered = len(adv_filtered_df) != len(adv_df)

    c1, c2, c3 = st.columns(3)
    if c1.button("✅ Apply These Filters to Dataset", key="apply_adv_filter", disabled=not is_filtered):
        st.session_state.df_clean = adv_filtered_df.copy()
        st.success("✅ Advanced filters have been applied to the working dataset.")
        st.session_state.adv_filter_reset_key += 1
        import time
        time.sleep(1)
        st.rerun()

    if c2.button("🧹 Clear Filter Controls", key="clear_adv_filter_ui"):
        st.session_state.adv_filter_reset_key += 1
        st.rerun()

    if c3.button("🔄 Reset Data to Original", key="reset_adv_data", type="primary"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.session_state.adv_filter_reset_key += 1
        st.success("✅ Dataset reset to original state. Filter controls also reset.")
        import time
        time.sleep(2)
        st.rerun()
        
# --- ⬇️ Export Tab ---
elif st.session_state.active_tab == "⬇️ Export":
    st.subheader("⬇️ Finalize and Export Your Dataset")

    if st.session_state.df_clean is None:
        st.warning("⚠️ No dataset loaded. Please load data from the sidebar.")
        st.stop()

    final_df = st.session_state.df_clean
    original_df = st.session_state.df_original

    original_rows, original_cols = original_df.shape
    final_rows, final_cols = final_df.shape
    rows_changed = original_rows - final_rows
    cols_changed = original_cols - final_cols

    st.markdown("### 📊 Summary of All Cleaning Operations")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📜 Original Dataset")
            st.metric(label="Rows", value=f"{original_rows:,}")
            st.metric(label="Columns", value=f"{original_cols:,}")
        with col2:
            st.markdown("#### ✨ Final Cleaned Dataset")
            st.metric(label="Rows", value=f"{final_rows:,}", delta=f"{-rows_changed:,}", delta_color="inverse")
            st.metric(label="Columns", value=f"{final_cols:,}", delta=f"{-cols_changed:,}", delta_color="inverse")

    st.markdown("### 👁️ Final Preview")
    st.dataframe(final_df, use_container_width=True)

    st.markdown("### 🚀 Export")
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            file_name = st.text_input("File name:", "cleaned_dataset.csv")
            include_index = st.toggle("Include row index in export", value=False)

        if not file_name.lower().endswith('.csv'):
            file_name += '.csv'

        @st.cache_data
        def convert_df_to_csv(df, index=False):
            return df.to_csv(index=index).encode('utf-8')

        csv_data = convert_df_to_csv(final_df, index=include_index)

        with col2:
            st.write("")
            st.write("")
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=file_name,
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("### 🔄 Final Actions")
    st.warning(
        "**Warning:** This is the final step. Resetting here will discard **ALL** changes "
        "made across every tab."
    )

    if st.button("Reset Entire Application to Original State", use_container_width=True, type="primary"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        if "adv_filter_reset_key" in st.session_state:
            st.session_state.adv_filter_reset_key += 1
        st.success("✅ Application state fully reset. You are back to the beginning.")
        import time
        time.sleep(2)
        st.rerun()
