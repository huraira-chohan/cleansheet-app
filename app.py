import re
import time
from io import StringIO
import numpy as np
import pandas as pd
import requests
import streamlit as st

# --- 2. CONSTANTS ---
APP_TITLE = "CleanSheet - All-in-One CSV Cleaner"
TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
NULL_VALUE_REPLACEMENTS = ["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"]
TAB_LABELS = [
    "📊 Preview", "🧹 Clean", "🧮 Columns", "🔍 Filter",
    "📈 Sort", "🧠 Advanced Filter", "⬇️ Export"
]

# --- 3. PAGE CONFIG ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("🧹 CleanSheet")
st.caption("An all-in-one, no-code data cleaning assistant")

# --- 4. SESSION STATE INIT ---
def initialize_session_state():
    ss = st.session_state
    if "df_clean" not in ss:
        ss.df_clean = None
        ss.df_original = None
        ss.active_tab = "📊 Preview"
        ss.adv_filter_reset_key = 0
        ss.staged_ops = {
            "clean": [],
            "columns": {"drop": [], "rename": {}},
            "filter": {"simple": [], "advanced": ""},
            "sort": {}
        }

initialize_session_state()

# --- 5. LOAD DATA ---
@st.cache_data(show_spinner="Downloading sample data...")
def load_titanic_sample():
    try:
        r = requests.get(TITANIC_URL)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.replace(NULL_VALUE_REPLACEMENTS, np.nan, inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load sample data: {e}")
        return None

def load_uploaded_file(file):
    try:
        df = pd.read_csv(file)
        df.replace(NULL_VALUE_REPLACEMENTS, np.nan, inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")
        return None

# --- 6. SIDEBAR UPLOAD ---
st.sidebar.markdown("### 📦 Load Dataset")
load_sample_clicked = st.sidebar.button("📂 Load Sample Titanic Dataset")
uploaded_file = st.sidebar.file_uploader("📤 Or Upload your CSV file", type=["csv"])

newly_loaded_df = None
if load_sample_clicked:
    newly_loaded_df = load_titanic_sample()
    if newly_loaded_df is not None:
        st.success("✅ Sample loaded.")
elif uploaded_file is not None:
    newly_loaded_df = load_uploaded_file(uploaded_file)
    if newly_loaded_df is not None:
        st.success("✅ File uploaded.")

if newly_loaded_df is not None:
    st.session_state.df_original = newly_loaded_df.copy()
    st.session_state.df_clean = newly_loaded_df.copy()
    st.rerun()

# --- 7. DATA GUARD ---
if st.session_state.df_clean is None:
    st.info("📎 Please upload a CSV file or load the sample dataset to get started.")
    st.stop()

# --- 8. NAVIGATION ---
st.radio(
    "Navigation", options=TAB_LABELS,
    key="active_tab", horizontal=True,
    label_visibility="collapsed"
)

# --- 9. HELPER FUNCTIONS ---
def convert_to_numeric(x):
    if pd.isnull(x): return np.nan
    try: return float(re.sub(r"[^0-9.]+", "", str(x)))
    except (ValueError, TypeError): return np.nan

def clean_text(text):
    return str(text).strip().title() if pd.notnull(text) else ""

def auto_clean_column(col):
    col = col.astype(str).str.strip()
    col = col.str.replace(r"[^\w\s]", "", regex=True)
    return col.str.title()

# --- 10. TAB HANDLERS ---

# Place all your previous tab logic here:
# - 📊 Preview
# - 🧹 Clean (with apply_action and both interactive/bulk logic)
# - 🧮 Columns (drop/rename/merge)
# - 🔍 Filter (numeric/date/category)
# - 📈 Sort (multi-column + duplicates)
# - 🧠 Advanced Filter (multi-condition builder)
# - ⬇️ Export (summary + download + reset)

# Use your previously working code blocks and place them under:
if st.session_state.active_tab == "📊 Preview":
        st.subheader("🔎 Dataset Preview")
        
    
df_display = st.session_state.df_clean.copy()

    view_opt = st.radio(
        "How much data to show?",
        options={"Top 5": 5, "Top 50": 50, "All": None},
        horizontal=True,
        key="preview_view_opt"
    )

    rows_to_show = {"Top 5": 5, "Top 50": 50, "All": None}[view_opt]

    if rows_to_show is None:
        st.dataframe(df_display, use_container_width=True)
    else:
        st.dataframe(df_display.head(rows_to_show), use_container_width=True)

    st.markdown("### ℹ️ Column Summary")
    st.dataframe(df_display.describe(include='all').T.fillna("N/A"), use_container_width=True)

    if st.button("🔄 Reset Dataset to Original"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("🔄 Dataset has been reset!")
        time.sleep(1)
        st.rerun()

   

elif st.session_state.active_tab == "🧹 Clean":
       st.subheader("🧹 Clean Your Dataset")

    if st.session_state.df_clean is None or st.session_state.df_clean.empty:
        st.warning("⚠️ Please load a dataset to begin cleaning.")
        st.stop()

    df_preview = st.session_state.df_clean.copy()

    def apply_action(df, column, action, params=None):
        if params is None:
            params = {}
        df_out = df.copy()
        if column not in df_out.columns:
            return df_out

        if action == "Remove NaNs":
            df_out.dropna(subset=[column], inplace=True)
        elif action == "Fill NaNs with 0":
            df_out[column].fillna(0, inplace=True)
        elif action == "Fill NaNs with Mean":
            if pd.api.types.is_numeric_dtype(df_out[column]):
                df_out[column].fillna(df_out[column].mean(), inplace=True)
        elif action == "Fill NaNs with Median":
            if pd.api.types.is_numeric_dtype(df_out[column]):
                df_out[column].fillna(df_out[column].median(), inplace=True)
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
                mean, std = df_out[column].mean(), df_out[column].std()
                df_out = df_out[df_out[column].between(mean - 3*std, mean + 3*std)]
        return df_out

    # --- INTERACTIVE CLEANING ---
    st.markdown("### 🔧 Interactive Column Cleaning")

    col_to_clean = st.selectbox("Select column to clean", df_preview.columns, key="interactive_clean_col")

    actions = st.multiselect(
        "Select cleaning steps (in order):",
        [
            "Remove NaNs", "Fill NaNs with 0", "Fill NaNs with Mean", "Fill NaNs with Median", "Custom Fill",
            "To Lowercase", "To Title Case", "Convert to Numeric", "Auto Clean", "Strip Whitespace",
            "Replace Values", "Remove Duplicates", "Remove Outliers"
        ],
        key="interactive_clean_actions"
    )

    # Parameter inputs
    action_params = {}
    if "Custom Fill" in actions:
        action_params['custom_value'] = st.text_input("Value for 'Custom Fill'", key="custom_fill_val")
    if "Replace Values" in actions:
        with st.expander("🔄 Configure 'Replace Values'"):
            old_vals = st.text_input("Values to replace (comma-separated)", key="old_vals")
            new_val = st.text_input("New value", key="new_val")
            if old_vals:
                action_params['replace_dict'] = {v.strip(): new_val for v in old_vals.split(",")}

    # Apply preview
    df_after_preview = df_preview.copy()
    for action in actions:
        df_after_preview = apply_action(df_after_preview, col_to_clean, action, action_params)

    col1, col2 = st.columns(2)
    col1.write("**Before Cleaning** (Top 10 rows)")
    col1.dataframe(df_preview[[col_to_clean]].head(10), use_container_width=True)

    col2.write("**After Preview** (Top 10 rows)")
    col2.dataframe(df_after_preview[[col_to_clean]].head(10), use_container_width=True)

    if st.button("✅ Apply These Changes", disabled=not actions):
        st.session_state.df_clean = df_after_preview.copy()
        st.success(f"✅ Changes applied to column '{col_to_clean}'")
        time.sleep(1)
        st.rerun()

    # --- BULK CLEANING ---
    st.markdown("---")
    st.markdown("### 🧹 Bulk Column Cleaning")

    for col in st.session_state.df_clean.columns:
        with st.expander(f"Clean Column: `{col}`"):
            c1, c2, c3 = st.columns([2, 2, 1])
            clean_opt = c1.selectbox(
                f"Transform `{col}`",
                ["None", "Text Normalize", "Convert to Numeric"],
                key=f"clean_{col}"
            )
            fill_opt = c2.selectbox(
                f"NaN Handling for `{col}`",
                ["None", "Drop Rows with NaNs", "Fill with Mean", "Fill with Median", "Fill with Mode"],
                key=f"fill_{col}"
            )

            if c3.button(f"Apply", key=f"apply_bulk_{col}"):
                df_bulk = st.session_state.df_clean.copy()
                if clean_opt == "Text Normalize":
                    df_bulk = apply_action(df_bulk, col, "To Title Case")
                elif clean_opt == "Convert to Numeric":
                    df_bulk = apply_action(df_bulk, col, "Convert to Numeric")

                if fill_opt == "Drop Rows with NaNs":
                    df_bulk = apply_action(df_bulk, col, "Remove NaNs")
                elif fill_opt == "Fill with Mean":
                    df_bulk = apply_action(df_bulk, col, "Fill NaNs with Mean")
                elif fill_opt == "Fill with Median":
                    df_bulk = apply_action(df_bulk, col, "Fill NaNs with Median")
                elif fill_opt == "Fill with Mode":
                    try:
                        mode_val = df_bulk[col].mode().iloc[0]
                        df_bulk[col].fillna(mode_val, inplace=True)
                    except:
                        st.warning(f"No mode found for `{col}`")

                st.session_state.df_clean = df_bulk.copy()
                st.success(f"✅ Cleaned `{col}` successfully.")
                time.sleep(1)
                st.rerun()

    # --- RESET ---
    st.markdown("---")
    st.markdown("### 🔄 Reset All Cleaning")
    if st.button("Reset ALL Cleaning to Original Dataset", use_container_width=True, type="primary"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ All cleaning reset.")
        time.sleep(1)
        st.rerun()


elif st.session_state.active_tab == "🧮 Columns":
        st.subheader("🧮 Manage Columns (Drop, Rename, Merge)")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please load a dataset to begin.")
        st.stop()

    working_df = st.session_state.df_clean.copy()
    available_columns = working_df.columns.tolist()

    col1, col2 = st.columns(2)

    # --- DROP COLUMNS ---
    with col1:
        st.markdown("### 🗑️ Drop Columns")
        cols_to_drop = st.multiselect(
            "Select columns to drop:",
            options=available_columns,
            key="cols_to_drop"
        )
        if st.button("Drop Selected Columns", key="drop_btn"):
            if not cols_to_drop:
                st.warning("⚠️ Select columns to drop.")
            else:
                df_dropped = working_df.drop(columns=cols_to_drop)
                st.session_state.df_clean = df_dropped
                st.success(f"✅ Dropped columns: {', '.join(cols_to_drop)}")
                time.sleep(1)
                st.rerun()

    # --- RENAME COLUMN ---
    with col2:
        st.markdown("### ✏️ Rename Column")
        col_to_rename = st.selectbox("Select column to rename:", options=available_columns, key="rename_col")
        new_col_name = st.text_input("Enter new name:", key="new_col_name").strip()

        if st.button("Rename Column", key="rename_btn"):
            if not new_col_name:
                st.warning("⚠️ New column name cannot be empty.")
            elif new_col_name == col_to_rename:
                st.info("ℹ️ New name is same as old.")
            elif new_col_name in available_columns:
                st.error("❌ Name already exists. Choose unique name.")
            else:
                working_df.rename(columns={col_to_rename: new_col_name}, inplace=True)
                st.session_state.df_clean = working_df
                st.success(f"✅ Renamed `{col_to_rename}` to `{new_col_name}`.")
                time.sleep(1)
                st.rerun()

    # --- MERGE COLUMNS ---
    st.markdown("---")
    st.markdown("### ➕ Merge Columns")

    cols_to_merge = st.multiselect(
        "Select 2+ columns to merge:",
        options=available_columns,
        key="cols_to_merge"
    )
    merged_col_name = st.text_input("New column name:", key="merged_col_name").strip()
    separator = st.text_input("Separator between values:", value=" ", key="separator")

    if st.button("Merge Selected Columns", key="merge_btn"):
        if len(cols_to_merge) < 2:
            st.warning("⚠️ Select at least two columns.")
        elif not merged_col_name:
            st.warning("⚠️ Enter a name for the merged column.")
        elif merged_col_name in available_columns:
            st.error("❌ Name already exists. Drop or rename first.")
        else:
            working_df[merged_col_name] = working_df[cols_to_merge].astype(str).agg(separator.join, axis=1)
            st.session_state.df_clean = working_df
            st.success(f"✅ Merged columns into `{merged_col_name}`.")
            time.sleep(1)
            st.rerun()

    # --- RESET ---
    st.markdown("---")
    st.markdown("### 🔄 Reset Column Changes")
    if st.button("Reset All Column Changes to Original", use_container_width=True, type="primary"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ All column changes reset to original.")
        time.sleep(2)
        st.rerun()


elif st.session_state.active_tab == "🔍 Filter":
        st.subheader("🔍 Filter and Preview Data")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please load a dataset to begin.")
        st.stop()

    working_df = st.session_state.df_clean.copy()
    col_to_filter = st.selectbox("1. Choose a column to filter by:", working_df.columns)

    def render_numeric_filter(df, column):
        series = df[column].dropna()
        if series.empty:
            st.warning(f"`{column}` has no numeric values.")
            return df
        min_val, max_val = float(series.min()), float(series.max())
        step = max((max_val - min_val) / 100, 0.01)
        start, end = st.slider("Select value range:", min_val, max_val, (min_val, max_val), step=step)
        return df[df[column].between(start, end)]

    def render_date_filter(df, column):
        series = pd.to_datetime(df[column], errors="coerce")
        if series.dropna().empty:
            st.error(f"❌ `{column}` could not be parsed as date.")
            return df
        min_date, max_date = series.min(), series.max()
        start, end = st.date_input("Date range:", (min_date, max_date), min_value=min_date, max_value=max_date)
        return df[series.between(pd.to_datetime(start), pd.to_datetime(end))]

    def render_categorical_filter(df, column):
        unique_vals = sorted(df[column].dropna().unique().tolist())
        if not unique_vals:
            st.warning(f"`{column}` has no values.")
            return df
        selected_vals = st.multiselect("Select values to keep:", unique_vals, default=unique_vals)
        if not selected_vals:
            return df.head(0)
        return df[df[column].isin(selected_vals)]

    col_type = working_df[col_to_filter].dtype
    if pd.api.types.is_numeric_dtype(col_type):
        filtered_df = render_numeric_filter(working_df, col_to_filter)
    elif pd.api.types.is_datetime64_any_dtype(col_type) or "date" in col_to_filter.lower():
        filtered_df = render_date_filter(working_df, col_to_filter)
    else:
        filtered_df = render_categorical_filter(working_df, col_to_filter)

    st.markdown("### 👁️ Filtered Preview")
    st.dataframe(filtered_df, use_container_width=True)
    st.write(f"Showing **{len(filtered_df)}** of **{len(working_df)}** rows.")

    col1, col2 = st.columns(2)
    is_filtered = len(filtered_df) != len(working_df)

    if col1.button("✅ Apply This Filter", disabled=not is_filtered, key="apply_filter_btn"):
        st.session_state.df_clean = filtered_df.copy()
        st.success("✅ Filter applied to dataset.")
        time.sleep(1)
        st.rerun()

    if col2.button("🔄 Reset All Filters", key="reset_filter_btn"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ Dataset reset to original.")
        time.sleep(1)
        st.rerun()

elif st.session_state.active_tab == "📈 Sort":
        st.subheader("📈 Sort Data and Manage Duplicates")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please load a dataset to begin.")
        st.stop()

    working_df = st.session_state.df_clean.copy()
    available_columns = working_df.columns.tolist()

    st.markdown("### 📊 Multi-Column Sorting")
    st.info("Sort by multiple columns. Order of selection determines priority.")

    cols_to_sort = st.multiselect("Columns to sort by:", options=available_columns, key="sort_cols")

    ascending_flags = []
    if cols_to_sort:
        st.markdown("#### Select sort order:")
        sort_order_cols = st.columns(len(cols_to_sort))
        for i, col in enumerate(cols_to_sort):
            with sort_order_cols[i]:
                ascending_flags.append(
                    st.checkbox(f"`{col}` Ascending", value=True, key=f"sort_asc_{col}")
                )

    if st.button("Apply Sort", disabled=not cols_to_sort, key="sort_btn"):
        sorted_df = working_df.sort_values(by=cols_to_sort, ascending=ascending_flags)
        st.session_state.df_clean = sorted_df
        sort_desc = ", ".join([f"{col} ({'ASC' if asc else 'DESC'})"
                              for col, asc in zip(cols_to_sort, ascending_flags)])
        st.success(f"✅ Sorted by {sort_desc}")
        time.sleep(1)
        st.rerun()

    # --- DUPLICATE REMOVAL ---
    st.markdown("---")
    st.markdown("### 🗑️ Remove Duplicate Rows")
    dup_count = working_df.duplicated().sum()

    if dup_count > 0:
        st.info(f"Found **{dup_count}** duplicate row(s).")
        if st.button("Remove All Duplicates", key="dedup_btn"):
            deduped_df = working_df.drop_duplicates()
            st.session_state.df_clean = deduped_df
            st.success(f"✅ Removed {dup_count} duplicate row(s).")
            time.sleep(1)
            st.rerun()
    else:
        st.success("✅ No duplicate rows found.")

    # --- RESET ---
    st.markdown("---")
    st.markdown("### 🔄 Reset Sort & Duplicates")
    if st.button("Reset Sorting & Duplicates", key="reset_sort_btn", use_container_width=True, type="primary"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.success("✅ Dataset restored to original order.")
        time.sleep(2)
        st.rerun()


elif st.session_state.active_tab == "🧠 Advanced Filter":
        st.subheader("🧠 Advanced Multi-Condition Filtering")
    st.info("Build multiple filter conditions and apply them using AND or OR logic.")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please load a dataset to begin.")
        st.stop()

    adv_df = st.session_state.df_clean.copy()

    if "adv_filter_reset_key" not in st.session_state:
        st.session_state.adv_filter_reset_key = 0
    key_prefix = f"adv_{st.session_state.adv_filter_reset_key}"

    # --- Helpers ---
    def create_numeric_condition(df, col, key):
        s = df[col].dropna()
        if s.empty: return None
        min_v, max_v = float(s.min()), float(s.max())
        start, end = st.slider(f"Range for `{col}`", min_v, max_v, (min_v, max_v), key=key)
        return df[col].between(start, end)

    def create_date_condition(df, col, key):
        dates = pd.to_datetime(df[col], errors="coerce").dropna()
        if dates.empty: return None
        min_d, max_d = dates.min(), dates.max()
        start, end = st.date_input(f"Date range for `{col}`", (min_d.date(), max_d.date()), key=key)
        return pd.to_datetime(df[col], errors="coerce").between(start, end)

    def create_category_condition(df, col, key):
        values = sorted(df[col].dropna().unique().tolist())
        selected = st.multiselect(f"Values for `{col}`", values, default=values, key=key)
        if not selected:
            return pd.Series([False] * len(df), index=df.index)
        return df[col].isin(selected)

    # --- Filter Builder UI ---
    with st.expander("🛠️ Build Filter Conditions", expanded=True):
        col1, col2 = st.columns(2)
        num_conds = col1.number_input("How many conditions?", min_value=1, max_value=5, value=1, key=f"{key_prefix}_num")
        logic = col2.radio("Combine using:", ["AND", "OR"], horizontal=True, key=f"{key_prefix}_logic")

        masks = []
        for i in range(int(num_conds)):
            st.markdown(f"#### Condition #{i+1}")
            col = st.selectbox(f"Column", adv_df.columns, key=f"{key_prefix}_col_{i}")
            dtype = adv_df[col].dtype

            mask = None
            if pd.api.types.is_numeric_dtype(dtype):
                mask = create_numeric_condition(adv_df, col, key=f"{key_prefix}_num_{i}")
            elif pd.api.types.is_datetime64_any_dtype(dtype) or "date" in col.lower():
                mask = create_date_condition(adv_df, col, key=f"{key_prefix}_date_{i}")
            else:
                mask = create_category_condition(adv_df, col, key=f"{key_prefix}_cat_{i}")

            if mask is not None:
                masks.append(mask)

    # --- Combine Conditions ---
    from functools import reduce
    import operator

    if masks:
        op = operator.and_ if logic == "AND" else operator.or_
        final_mask = reduce(op, masks)
    else:
        final_mask = pd.Series([True] * len(adv_df), index=adv_df.index)

    adv_filtered_df = adv_df[final_mask]

    # --- Results + Actions ---
    st.markdown("### 👁️ Filtered Preview")
    st.dataframe(adv_filtered_df, use_container_width=True)
    st.write(f"Showing **{len(adv_filtered_df)}** of **{len(adv_df)}** rows.")

    is_filtered = len(adv_filtered_df) != len(adv_df)
    col1, col2, col3 = st.columns(3)

    if col1.button("✅ Apply Filter", disabled=not is_filtered, key="apply_adv"):
        st.session_state.df_clean = adv_filtered_df.copy()
        st.success("✅ Filter applied to dataset.")
        st.session_state.adv_filter_reset_key += 1
        time.sleep(1)
        st.rerun()

    if col2.button("🧹 Clear Filters", key="clear_adv"):
        st.session_state.adv_filter_reset_key += 1
        st.rerun()

    if col3.button("🔄 Reset Dataset", key="reset_adv", type="primary"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        st.session_state.adv_filter_reset_key += 1
        st.success("✅ Reset to original dataset.")
        time.sleep(2)
        st.rerun()


elif st.session_state.active_tab == "⬇️ Export":
      st.subheader("⬇️ Finalize and Export Your Dataset")

    if st.session_state.df_clean is None:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    final_df = st.session_state.df_clean
    original_df = st.session_state.df_original

    # --- Summary Stats ---
    st.markdown("### 📊 Summary of Changes")
    orig_rows, orig_cols = original_df.shape
    final_rows, final_cols = final_df.shape

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📜 Original Dataset")
            st.metric("Rows", f"{orig_rows:,}")
            st.metric("Columns", f"{orig_cols:,}")

        with col2:
            st.markdown("#### ✨ Final Dataset")
            st.metric("Rows", f"{final_rows:,}", delta=f"{final_rows - orig_rows:+,}")
            st.metric("Columns", f"{final_cols:,}", delta=f"{final_cols - orig_cols:+,}")

    # --- Final Preview ---
    st.markdown("### 👁️ Preview Cleaned Data")
    st.dataframe(final_df, use_container_width=True)

    # --- Export ---
    st.markdown("### 🚀 Export Dataset")

    col1, col2 = st.columns([3, 1])

    with col1:
        file_name = st.text_input("Filename:", "cleaned_dataset.csv")
        include_index = st.toggle("Include row index", value=False)

    if not file_name.lower().endswith(".csv"):
        file_name += ".csv"

    @st.cache_data
    def convert_df_to_csv(df, index=False):
        return df.to_csv(index=index).encode("utf-8")

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

    # --- Reset Entire App ---
    st.markdown("---")
    st.markdown("### 🔄 Reset Entire Application")

    st.warning("⚠️ This will discard **ALL** changes across all tabs.")

    if st.button("Reset to Original State", use_container_width=True, type="primary"):
        st.session_state.df_clean = st.session_state.df_original.copy()
        if "adv_filter_reset_key" in st.session_state:
            st.session_state.adv_filter_reset_key += 1
        st.success("✅ Fully reset to original dataset.")
        time.sleep(2)
        st.rerun()

