import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import requests

# --- Initialize Session State ---
# This ensures that active_tab and df_clean are always present,
# preventing potential KeyError on first run.
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Preview"
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "df_original" not in st.session_state: # Also initialize df_original
    st.session_state.df_original = None
if "df_backup" not in st.session_state: # For Undo functionality
    st.session_state.df_backup = None
# --- FIX START: More robust initialization of df_full and df_temp ---
if "df_full" not in st.session_state or not isinstance(st.session_state.df_full, pd.DataFrame):
    st.session_state.df_full = pd.DataFrame() # Initialize as an empty DataFrame
if "df_temp" not in st.session_state or not isinstance(st.session_state.df_temp, pd.DataFrame):
    st.session_state.df_temp = pd.DataFrame() # Initialize as an empty DataFrame
# --- FIX END ---

# --- Page Configuration ---
st.set_page_config(page_title="CleanSheet - All-in-One CSV Cleaner", layout="wide")
st.title("🧹 CleanSheet")
st.caption("An all-in-one, no-code data cleaning assistant")

# --- Sidebar: Load Dataset ---
st.sidebar.markdown("### 📦 Load Dataset")

load_sample = st.sidebar.button("📂 Load Sample Titanic Dataset")
uploaded_file = st.sidebar.file_uploader("📤 Or Upload your CSV file", type=["csv"], key="sidebar_uploader") # Added key

# Load dataset into session state based on user action
if load_sample:
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        response = requests.get(titanic_url)
        df_loaded = pd.read_csv(StringIO(response.text))
        # Replace common null representations upon loading
        df_loaded.replace(["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"], np.nan, inplace=True)
        st.session_state.df_clean = df_loaded.copy()
        st.session_state.df_original = df_loaded.copy() # Store original for reset
        st.session_state.df_full = df_loaded.copy() # Store full for filter limits
        st.success("✅ Sample Titanic dataset loaded successfully!")
        st.rerun() # Rerun to update the main content
    except Exception as e:
        st.error(f"❌ Failed to load sample data: {e}")

elif uploaded_file is not None:
    try:
        df_loaded = pd.read_csv(uploaded_file)
        # Replace common null representations upon loading
        df_loaded.replace(["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"], np.nan, inplace=True)
        st.session_state.df_clean = df_loaded.copy()
        st.session_state.df_original = df_loaded.copy() # Store original for reset
        st.session_state.df_full = df_loaded.copy() # Store full for filter limits
        st.success("✅ Your dataset was uploaded successfully!")
        st.rerun() # Rerun to update the main content
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")

# Ensure data is available before proceeding
if st.session_state.df_clean is None:
    st.info("📎 Please upload a CSV file or load the sample dataset to get started.")
    st.stop() # Stop further execution if no data

# Use the session state dataframe for current operations
# df = st.session_state.df_clean.copy() # This line is often problematic.
# Instead, access df from session_state when needed within tabs to ensure it's always current.
# The previous solution already did this in each tab's context, so this specific line can be removed
# if you're consistent with `st.session_state.get("df_clean", pd.DataFrame()).copy()`

# --- Helper Functions ---
NULL_VALUES = ["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"]

def clean_text(x):
    """Normalizes text by stripping whitespace and converting to title case."""
    return str(x).strip().title() if pd.notnull(x) else x

def convert_to_numeric(x):
    """Converts input to a float, removing non-numeric characters first."""
    try:
        # Convert to string first to handle various input types
        s = str(x)
        # Remove all characters that are not digits or a period
        cleaned_s = re.sub(r"[^0-9.]+", "", s)
        return float(cleaned_s)
    except ValueError:
        return np.nan # Return NaN if conversion fails

# --- Tabs for Navigation ---
tab_labels = ["📊 Preview", "🧹 Clean", "🧮 Columns", "🔍 Filter", "📈 Sort", "🧠 Advanced Filter", "⬇️ Export"]
# Find the index of the active tab for initial selection
try:
    tab_index = tab_labels.index(st.session_state.active_tab)
except ValueError:
    tab_index = 0 # Default to Preview if active_tab is not found
tabs = st.tabs(tab_labels, index=tab_index)

# --- Tab Content ---

# --- Preview Tab ---
with tabs[0]:
    st.session_state.active_tab = tab_labels[0]
    st.subheader("🔎 Dataset Preview")
    # Access the current df_clean here
    df_preview = st.session_state.get("df_clean", pd.DataFrame())
    if df_preview.empty:
        st.warning("No data to preview.")
    else:
        view_opt = st.radio("How much data to show?", ["Top 5", "Top 50", "All"], horizontal=True, key="preview_view_opt")
        
        if view_opt == "Top 5":
            st.dataframe(df_preview.head(), use_container_width=True)
        elif view_opt == "Top 50":
            st.dataframe(df_preview.head(50), use_container_width=True)
        else:
            st.dataframe(df_preview, use_container_width=True)

        st.markdown("---")
        st.write("#### ℹ️ Column Summary")
        st.dataframe(df_preview.describe(include='all').T.fillna("N/A"))

# --- Clean Tab ---
with tabs[1]:
    st.subheader("🔎 Clean Columns")
    st.session_state.active_tab = tab_labels[1]
    # Access the current df_clean here
    df_clean_tab = st.session_state.get("df_clean", pd.DataFrame())

    if df_clean_tab.empty:
        st.warning("No data to clean.")
        st.stop() # Stop if no data

    columns = df_clean_tab.columns.tolist()

    # Create a deep copy of df for local modifications in this tab before saving
    df_temp_clean = df_clean_tab.copy()

    for col in columns:
        with st.expander(f"⚙️ {col}"):
            clean_opt = st.selectbox(f"Cleaning for `{col}`", [
                "None", "Text Normalize", "Convert to Numeric"
            ], key=f"clean_opt_{col}")

            fill_na = st.selectbox(f"Missing values in `{col}`", [
                "None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"
            ], key=f"na_opt_{col}")

            if st.button(f"✅ Apply cleaning to `{col}`", key=f"apply_clean_{col}"):
                # Apply cleaning operations
                if clean_opt == "Text Normalize":
                    df_temp_clean[col] = df_temp_clean[col].apply(clean_text)
                elif clean_opt == "Convert to Numeric":
                    df_temp_clean[col] = df_temp_clean[col].apply(convert_to_numeric)

                # Apply missing value handling
                if df_temp_clean[col].isnull().any(): # Only apply if there are NaNs
                    if fill_na == "Drop Rows":
                        # Ensure to drop rows across the whole temp dataframe
                        df_temp_clean.dropna(subset=[col], inplace=True)
                    elif fill_na == "Fill with Mean" and pd.api.types.is_numeric_dtype(df_temp_clean[col]):
                        df_temp_clean[col].fillna(df_temp_clean[col].mean(), inplace=True)
                    elif fill_na == "Fill with Median" and pd.api.types.is_numeric_dtype(df_temp_clean[col]):
                        df_temp_clean[col].fillna(df_temp_clean[col].median(), inplace=True)
                    elif fill_na == "Fill with Mode":
                        # Mode can return multiple values, pick the first
                        mode_val = df_temp_clean[col].mode()
                        if not mode_val.empty:
                            df_temp_clean[col].fillna(mode_val.iloc[0], inplace=True)
                        else:
                            st.warning(f"No mode found for `{col}`. Missing values not filled.")

                st.success(f"✅ Cleaning applied to `{col}`")
                st.session_state.df_clean = df_temp_clean.copy() # Update session state after each column's apply
                st.dataframe(st.session_state.df_clean.head(), use_container_width=True) # Show quick preview
                st.rerun() # Rerun to update the dataframe reflected in other tabs

# --- Column Tab ---
with tabs[2]:
    st.session_state.active_tab = tab_labels[2]
    st.subheader("🧮 Manage Columns")
    # Access the current df_clean here
    df_cols_tab = st.session_state.get("df_clean", pd.DataFrame())

    if df_cols_tab.empty:
        st.warning("No data to manage columns.")
        st.stop() # Stop if no data

    # Use a copy for operations within this tab before committing to session_state
    df_temp_cols = df_cols_tab.copy()

    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🗑 Drop Columns")
        drop_cols = st.multiselect("Select columns to drop", df_temp_cols.columns.tolist(), key="drop_cols_select")
        if st.button("Drop Selected Columns", key="drop_cols_btn"):
            if drop_cols:
                df_temp_cols.drop(columns=drop_cols, inplace=True)
                st.success("✅ Dropped selected columns")
                st.session_state.df_clean = df_temp_cols.copy()
                st.rerun()
            else:
                st.warning("Please select columns to drop.")

    with col2:
        st.write("### ✏️ Rename Column")
        old_col = st.selectbox("Column to rename", df_temp_cols.columns.tolist(), key="old_col_select")
        new_col = st.text_input("New column name", key="new_col_input")
        if st.button("Rename Column", key="rename_col_btn"):
            if new_col.strip() and old_col in df_temp_cols.columns:
                if new_col not in df_temp_cols.columns or new_col == old_col: # Allow renaming to self or unique name
                    df_temp_cols.rename(columns={old_col: new_col}, inplace=True)
                    st.success(f"✅ Renamed `{old_col}` to `{new_col}`")
                    st.session_state.df_clean = df_temp_cols.copy()
                    st.rerun()
                else:
                    st.error(f"Column '{new_col}' already exists. Please choose a unique name.")
            else:
                st.warning("Please enter a valid new column name.")

    st.markdown("---")
    st.write("### ➕ Merge Columns")
    merge_cols = st.multiselect("Select 2 or more columns to merge", df_temp_cols.columns.tolist(), key="merge_cols_select")
    merge_name = st.text_input("Merged column name", key="merge_name_input")
    sep = st.text_input("Separator (e.g. space, comma)", value=" ", key="merge_sep_input")
    if st.button("Merge Columns", key="merge_cols_btn"):
        if len(merge_cols) >= 2 and merge_name.strip():
            # Convert selected columns to string to prevent errors with mixed types during join
            df_temp_cols[merge_name] = df_temp_cols[merge_cols].astype(str).agg(sep.join, axis=1)
            st.success(f"✅ Created merged column `{merge_name}`")
            st.session_state.df_clean = df_temp_cols.copy()
            st.rerun()
        else:
            st.warning("Please select at least 2 columns and provide a name for the merged column.")

    st.markdown("---")
    st.write("### 🔤 Split Alphanumeric Column")
    split_col = st.selectbox("Select alphanumeric column to split", df_temp_cols.columns.tolist(), key="split_col_select")
    new_alpha = st.text_input("New column for alphabets", key="alpha_part_input")
    new_num = st.text_input("New column for numbers", key="num_part_input")

    if st.button("Split Alphanumeric", key="split_alphanum_btn"):
        if new_alpha.strip() and new_num.strip():
            # Check if source column is string-like
            if not pd.api.types.is_string_dtype(df_temp_cols[split_col]):
                st.warning(f"Column `{split_col}` is not a string type. Attempting to convert to string for splitting.")
                df_temp_cols[split_col] = df_temp_cols[split_col].astype(str)

            df_temp_cols[new_alpha] = df_temp_cols[split_col].apply(lambda x: ''.join(re.findall(r'[A-Za-z]+', str(x))))
            df_temp_cols[new_num] = df_temp_cols[split_col].apply(lambda x: ''.join(re.findall(r'\d+', str(x))))
            st.success(f"✅ Split `{split_col}` into `{new_alpha}` and `{new_num}`")
            st.session_state.df_clean = df_temp_cols.copy()
            st.rerun()
        else:
            st.warning("Please provide names for both new alphabet and number columns.")

# --- Filter Tab (IMPROVED) ---
with tabs[3]:
    st.session_state.active_tab = tab_labels[3]
    st.subheader("🔍 Filter Rows (temporary view only)")
    st.markdown("---")

    # Use a fresh copy of the clean DataFrame for temporary filtering
    current_df_for_filtering = st.session_state.get("df_clean", pd.DataFrame()).copy()

    # --- Robust Empty DataFrame Handling ---
    if current_df_for_filtering.empty:
        st.warning("⚠️ No data available to filter. Please upload or process data first.")
        # Provide reset options even if current df is empty, to potentially load data
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("🔄 Reset to Original Uploaded Dataset", key="filter_tab_reset_empty_btn"):
                if st.session_state.df_original is not None and not st.session_state.df_original.empty:
                    st.session_state.df_clean = st.session_state.df_original.copy()
                    st.session_state.df_temp = st.session_state.df_original.copy()
                    st.success("✅ Reset to original uploaded dataset.")
                    st.rerun() # Rerun to show the now available data
                else:
                    st.warning("⚠️ No original dataset available to reset to.")
        with col_r2:
            if st.button("↩️ Undo Last Filter", key="filter_tab_undo_empty_btn"):
                if st.session_state.df_backup is not None and not st.session_state.df_backup.empty:
                    st.session_state.df_clean = st.session_state.df_backup.copy()
                    st.session_state.df_temp = st.session_state.df_backup.copy()
                    st.success("🔁 Undo successful. Reverted to previous dataset.")
                    st.rerun()
                else:
                    st.warning("⚠️ No previous dataset found to undo.")
        st.stop() # Stop further execution if no data

    # Keep a reference to the full, original dataset for slider limits.
    # This prevents slider ranges from shrinking after filtering or cleaning.
    # Ensure df_full is always a valid DataFrame when accessed.
    if st.session_state.df_full.empty and st.session_state.df_original is not None:
        st.session_state.df_full = st.session_state.df_original.copy()
    elif st.session_state.df_full.empty: # If df_original is also empty/None
        st.session_state.df_full = current_df_for_filtering.copy() # Use current as fallback

    # Ensure column selection is safe, especially if df becomes empty somehow
    available_columns = current_df_for_filtering.columns.tolist()
    if not available_columns:
        st.warning("⚠️ No columns available to filter. Please check your dataset.")
        st.stop() # Stop here if no columns

    col_to_filter = st.selectbox("Choose column to filter", available_columns, key="filter_column_select")
    # Initialize filtered_df with a copy of the current_df_for_filtering
    # This ensures that if no filter is applied (e.g., due to warnings), the original data is shown.
    filtered_df = current_df_for_filtering.copy()

    st.markdown("---") # Visual separator

    # --- Numeric filtering ---
    try:
        if pd.api.types.is_numeric_dtype(current_df_for_filtering[col_to_filter]):
            st.write(f"📏 Numeric Range Filter for `{col_to_filter}`")
            # --- FIX START: Safely access full_data_col in filter tab ---
            # Ensure col exists in df_full before trying to access it
            if col_to_filter not in st.session_state.df_full.columns:
                st.warning(f"⚠️ Column `{col_to_filter}` not found in original dataset for range limits. Using current data for limits.")
                full_col_numeric = pd.to_numeric(current_df_for_filtering[col_to_filter], errors='coerce').dropna()
            else:
                full_col_numeric = pd.to_numeric(st.session_state.df_full[col_to_filter], errors='coerce').dropna()
            # --- FIX END ---

            if full_col_numeric.empty:
                st.warning(f"⚠️ No valid numeric values in `{col_to_filter}` to filter.")
            else:
                min_val = float(full_col_numeric.min())
                max_val = float(full_col_numeric.max())

                if min_val == max_val:
                    st.warning(f"⚠️ All values in `{col_to_filter}` are the same: {min_val}. No range to filter.")
                    # If all values are the same, no filtering occurs, filtered_df remains current_df_for_filtering
                else:
                    # Calculate step_val more robustly, ensuring it's never zero
                    step_val = max((max_val - min_val) / 100, 0.01) # Minimum step 0.01
                    if step_val == 0: # Double-check for edge cases near zero difference
                        step_val = 0.01

                    start, end = st.slider(
                        "Select value range",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        step=step_val,
                        key=f"numeric_slider_{col_to_filter}"
                    )
                    # Apply filter using the *current* DataFrame for filtering, converting column to numeric robustly
                    numeric_col_for_filter = pd.to_numeric(current_df_for_filtering[col_to_filter], errors='coerce')
                    filtered_df = current_df_for_filtering[numeric_col_for_filter.between(start, end, inclusive='both').fillna(False)]
                    st.info(f"Showing rows where `{col_to_filter}` is between {start:.2f} and {end:.2f}.")

    except Exception as e:
        st.error(f"An unexpected error occurred during numeric filtering for `{col_to_filter}`: {e}")
        # Fallback: if an error occurs, filtered_df remains a copy of the original (no filter applied)

    # --- Datetime filtering ---
    # Check for datetime type or if 'date' is in the name AND it's not a numeric column
    elif pd.api.types.is_datetime64_any_dtype(current_df_for_filtering[col_to_filter]) or \
         ("date" in col_to_filter.lower() and not pd.api.types.is_numeric_dtype(current_df_for_filtering[col_to_filter])):
        st.write(f"🗓 Date Range Filter for `{col_to_filter}`")

        # Coerce to datetime for both current and full DataFrames, dropping NaT values
        current_col_dt = pd.to_datetime(current_df_for_filtering[col_to_filter], errors='coerce')
        
        # --- FIX START: Safely access full_data_col for datetime filter ---
        if col_to_filter not in st.session_state.df_full.columns:
            st.warning(f"⚠️ Column `{col_to_filter}` not found in original dataset for date range limits. Using current data for limits.")
            full_col_dt = pd.to_datetime(current_df_for_filtering[col_to_filter], errors='coerce').dropna()
        else:
            full_col_dt = pd.to_datetime(st.session_state.df_full[col_to_filter], errors='coerce').dropna()
        # --- FIX END ---

        if full_col_dt.empty:
            st.warning(f"⚠️ No valid date/time values found in `{col_to_filter}` to filter.")
        else:
            min_date_full = full_col_dt.min()
            max_date_full = full_col_dt.max()

            if pd.isnull(min_date_full) or pd.isnull(max_date_full):
                st.warning(f"⚠️ Could not convert `{col_to_filter}` to datetime. Please ensure it's a valid date format.")
                # filtered_df remains current_df_for_filtering if conversion fails
            else:
                try:
                    date_start, date_end = st.date_input(
                        "Select date range",
                        # Pass Python date objects to date_input widget
                        value=(min_date_full.date(), max_date_full.date()),
                        min_value=min_date_full.date(),
                        max_value=max_date_full.date(),
                        key=f"date_input_{col_to_filter}"
                    )
                    # Convert date_input output back to datetime for filtering
                    # Add time component for inclusive range if dates are only precise to day
                    date_start_dt = pd.to_datetime(date_start).normalize()
                    date_end_dt = pd.to_datetime(date_end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) # End of the selected day

                    # Apply filter using the *current* DataFrame for filtering
                    filtered_df = current_df_for_filtering[current_col_dt.between(date_start_dt, date_end_dt, inclusive='both').fillna(False)]
                    st.info(f"Showing rows where `{col_to_filter}` is between {date_start} and {date_end}.")
                except Exception as e:
                    st.error(f"An error occurred while selecting dates for `{col_to_filter}`: {e}")
                    # Fallback: if an error occurs, filtered_df remains a copy of the original (no filter applied)

    # --- Categorical filtering ---
    else:
        st.write(f"🔠 Categorical Filter for `{col_to_filter}`")
        # --- FIX START: Safely access full_data_col for categorical filter ---
        if col_to_filter not in st.session_state.df_full.columns:
            st.warning(f"⚠️ Column `{col_to_filter}` not found in original dataset for categorical options. Using current data for options.")
            unique_vals_full = current_df_for_filtering[col_to_filter].astype(str).dropna().unique().tolist()
        else:
            # Use df_full for unique values, convert to string to handle mixed types
            unique_vals_full = st.session_state.df_full[col_to_filter].astype(str).dropna().unique().tolist()
        # --- FIX END ---
        unique_vals_full.sort() # Sort for better user experience

        if not unique_vals_full:
            st.warning("⚠️ No valid values to filter.")
        else:
            selected_vals = st.multiselect(
                f"Select values to include from `{col_to_filter}`:",
                options=unique_vals_full,
                key=f"categorical_multiselect_{col_to_filter}"
            )
            if selected_vals:
                # Apply filter using the *current* DataFrame for filtering, converting column to string
                filtered_df = current_df_for_filtering[current_df_for_filtering[col_to_filter].astype(str).isin(selected_vals)]
                st.info(f"Filtered by selected values in `{col_to_filter}`.")
            else:
                filtered_df = current_df_for_filtering.copy() # If nothing selected, show all
                st.caption("No filter applied — showing full dataset.")

    st.markdown("---") # Visual separator
    # Show filtered data preview
    st.dataframe(filtered_df, use_container_width=True)
    st.write(f"Displaying **{len(filtered_df)}** of **{len(current_df_for_filtering)}** rows.")
    st.markdown("---")

    # --- Apply filter to export ---
    st.markdown("### 📤 Apply Filter to Export?")

    if st.button("✅ Apply Filter", key="apply_filter_btn_filter_tab"):
        # Before applying, save the current state of df_clean to df_backup
        # This allows a true "undo" to the state *before* this filter was applied.
        st.session_state.df_backup = st.session_state.df_clean.copy()
        st.session_state.df_clean = filtered_df.copy() # Permanently update the cleaned DataFrame
        st.session_state.df_temp = filtered_df.copy() # Update temporary view
        st.success("✅ Filter applied to exported dataset (affects Export tab and all views).")
        st.toast("Filter applied!", icon="✅")
        st.rerun() # Rerun to reflect the new df_clean state immediately

    st.markdown("---") # Visual separator

    # --- Undo / Reset ---
    col1_undo, col2_reset = st.columns(2)
    with col1_undo:
        if st.button("↩️ Undo Last Filter", key="undo_last_filter_btn_filter_tab"):
            if st.session_state.df_backup is not None and not st.session_state.df_backup.empty:
                st.session_state.df_clean = st.session_state.df_backup.copy()
                st.session_state.df_temp = st.session_state.df_backup.copy()
                st.success("🔁 Undo successful. Reverted to previous dataset.")
                st.toast("Undo successful!", icon="↩️")
                st.rerun() # Rerun to update the display
            else:
                st.warning("⚠️ No previous dataset found to undo.")

    with col2_reset:
        if st.button("🔄 Reset to Original Uploaded Dataset", key="reset_to_original_btn_filter_tab"):
            if st.session_state.df_original is not None and not st.session_state.df_original.empty:
                st.session_state.df_clean = st.session_state.df_original.copy()
                st.session_state.df_temp = st.session_state.df_original.copy()
                # df_full should also revert to original data range
                st.session_state.df_full = st.session_state.df_original.copy()
                st.success("✅ Reset to original uploaded dataset.")
                st.toast("Dataset reset!", icon="🔄")
                st.rerun() # Rerun to update the display
            else:
                st.warning("⚠️ No original dataset available.")


# --- Sort Tab ---
with tabs[4]:
    st.session_state.active_tab = tab_labels[4]
    st.subheader("📈 Sort Data")
    st.info("ℹ️ Sorting affects the 'clean' dataset. Use 'Reset to Original' to undo any sort or filter applied.")

    # Access the current df_clean here
    df_sort_tab = st.session_state.get("df_clean", pd.DataFrame())

    if df_sort_tab.empty:
        st.warning("⚠️ No data available to sort.")
        st.stop() # Stop if no data

    # Use a copy for this tab's operations
    df_temp_sort = df_sort_tab.copy()

    sort_col = st.selectbox("Column to sort by", df_temp_sort.columns.tolist(), key="sort_column_select")
    ascending = st.checkbox("Sort ascending", value=True, key="sort_ascending_checkbox")
    if st.button("Apply Sort", key="apply_sort_btn"):
        df_temp_sort = df_temp_sort.sort_values(by=sort_col, ascending=ascending, ignore_index=True)
        st.success(f"✅ Sorted by `{sort_col}`.")
        st.session_state.df_clean = df_temp_sort.copy() # Update session state
        st.dataframe(st.session_state.df_clean.head(), use_container_width=True) # Show preview
        st.rerun()

    st.markdown("---")
    st.write("### 🧬 Remove Duplicate Rows")
    if st.button("Remove Duplicates", key="remove_duplicates_btn"):
        if df_temp_sort.empty:
            st.warning("⚠️ No data to remove duplicates from.")
        else:
            before = len(df_temp_sort)
            df_temp_sort.drop_duplicates(inplace=True, ignore_index=True)
            after = len(df_temp_sort)
            st.success(f"✅ Removed {before - after} duplicate rows.")
            st.session_state.df_clean = df_temp_sort.copy() # Update session state
            st.dataframe(st.session_state.df_clean.head(), use_container_width=True) # Show preview
            st.rerun()


# --- Advanced Filter Tab ---
with tabs[5]:
    st.session_state.active_tab = tab_labels[5]
    st.subheader("🧠 Advanced Multi-Column Filtering")
    st.info("ℹ️ This filter allows combining multiple conditions using AND/OR logic.")
    st.markdown("---")

    # Use a fresh copy of the clean DataFrame for temporary filtering
    current_df_for_advanced_filtering = st.session_state.get("df_clean", pd.DataFrame()).copy()

    if current_df_for_advanced_filtering.empty:
        st.warning("⚠️ No data available for advanced filtering. Please upload or process data first.")
        st.stop()

    # Select number of filters
    num_conditions = st.number_input("How many filter conditions?", min_value=1, max_value=5, value=1, key="num_conditions_input")
    logic = st.radio("Combine filters using:", ["AND", "OR"], horizontal=True, key="logic_radio")

    conditions = []
    st.markdown("---")

    for i in range(int(num_conditions)):
        st.markdown(f"### ➕ Condition #{i+1}")
        col_options = current_df_for_advanced_filtering.columns.tolist()
        if not col_options:
            st.warning("No columns available for filtering.")
            continue # Skip this condition if no columns

        col = st.selectbox(f"Choose column for condition #{i+1}", col_options, key=f"adv_col_{i}")
        
        # Get the dtype of the selected column in the CURRENT dataframe
        dtype = current_df_for_advanced_filtering[col].dtype

        # --- FIX START: Safely access full_data_col in advanced filter ---
        # Ensure col exists in df_full before trying to access it, provide fallback
        if st.session_state.df_full is None or st.session_state.df_full.empty or col not in st.session_state.df_full.columns:
            st.warning(f"⚠️ Column `{col}` not found in original dataset for range/options. Using current data for limits.")
            full_data_col = current_df_for_advanced_filtering[col]
        else:
            full_data_col = st.session_state.df_full[col]
        # --- FIX END ---


        if pd.api.types.is_numeric_dtype(dtype):
            # Numeric values should be extracted from the full_data_col
            numeric_full_col = pd.to_numeric(full_data_col, errors='coerce').dropna()

            if numeric_full_col.empty:
                st.warning(f"🔒 `{col}` has no valid numeric values for filtering.")
                # Create a condition that includes all rows if no valid numeric data
                cond = pd.Series([True] * len(current_df_for_advanced_filtering), index=current_df_for_advanced_filtering.index)
            else:
                min_val, max_val = float(numeric_full_col.min()), float(numeric_full_col.max())
                if min_val != max_val:
                    step_val = max((max_val - min_val) / 100, 0.01)
                    range_val = st.slider(
                        f"Range for `{col}`", min_val, max_val, (min_val, max_val),
                        step=step_val, key=f"adv_range_{i}"
                    )
                    # Ensure column is numeric for filtering, handling NaNs safely
                    current_col_numeric = pd.to_numeric(current_df_for_advanced_filtering[col], errors='coerce')
                    cond = current_col_numeric.between(range_val[0], range_val[1], inclusive='both').fillna(False)
                else:
                    st.warning(f"🔒 `{col}` has a constant value ({min_val}). No range to filter.")
                    cond = pd.Series([True] * len(current_df_for_advanced_filtering), index=current_df_for_advanced_filtering.index)

        elif pd.api.types.is_datetime64_any_dtype(dtype) or ("date" in col.lower() and not pd.api.types.is_numeric_dtype(dtype)):
            # Datetime values should be extracted from the full_data_col
            datetime_full_col = pd.to_datetime(full_data_col, errors="coerce").dropna()

            if datetime_full_col.empty:
                st.warning(f"🔒 `{col}` has no valid date/time values for filtering.")
                cond = pd.Series([True] * len(current_df_for_advanced_filtering), index=current_df_for_advanced_filtering.index)
            else:
                min_date, max_date = datetime_full_col.min(), datetime_full_col.max()
                try:
                    date_start, date_end = st.date_input(
                        f"Date range for `{col}`", (min_date.date(), max_date.date()), key=f"adv_date_{i}"
                    )
                    # Convert date_input output back to datetime for filtering
                    date_start_dt = pd.to_datetime(date_start).normalize()
                    date_end_dt = pd.to_datetime(date_end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                    # Ensure column is datetime for filtering, handling NaNs safely
                    current_col_datetime = pd.to_datetime(current_df_for_advanced_filtering[col], errors='coerce')
                    cond = current_col_datetime.between(date_start_dt, date_end_dt, inclusive='both').fillna(False)
                except Exception as e:
                    st.error(f"Error with date selection for `{col}`: {e}")
                    cond = pd.Series([True] * len(current_df_for_advanced_filtering), index=current_df_for_advanced_filtering.index)

        else: # Categorical
            # Categorical values should be extracted from the full_data_col and converted to string
            unique_values = full_data_col.astype(str).dropna().unique().tolist()
            unique_values.sort()

            if not unique_values:
                st.warning(f"🔒 `{col}` has no valid categorical values for filtering.")
                cond = pd.Series([True] * len(current_df_for_advanced_filtering), index=current_df_for_advanced_filtering.index)
            else:
                selected = st.multiselect(f"Select values for `{col}`", unique_values, key=f"adv_cat_{i}")
                if selected:
                    # Ensure column is string for filtering, handling NaNs safely
                    cond = current_df_for_advanced_filtering[col].astype(str).isin(selected)
                else:
                    cond = pd.Series([True] * len(current_df_for_advanced_filtering), index=current_df_for_advanced_filtering.index) # no filtering

        conditions.append(cond)

    st.markdown("---")

    if conditions:
        # Align all conditions to the index of current_df_for_advanced_filtering
        aligned_conditions = [c.reindex(current_df_for_advanced_filtering.index, fill_value=False) for c in conditions]

        combined = aligned_conditions[0]
        for c in aligned_conditions[1:]:
            combined = combined & c if logic == "AND" else combined | c

        filtered_df = current_df_for_advanced_filtering[combined]
        st.success(f"✅ {len(filtered_df)} rows matched your filters out of {len(current_df_for_advanced_filtering)}.")
        st.dataframe(filtered_df, use_container_width=True)

        st.markdown("---")
        col1_adv, col2_adv, col3_adv = st.columns(3)

        with col1_adv:
            if st.button("✅ Apply Filters", key="apply_advanced_filters_btn"):
                st.session_state.df_backup = st.session_state.df_clean.copy() # Backup before applying
                st.session_state.df_clean = filtered_df.copy()
                # df_full should NOT change here, as it's the original for range limits
                st.success("✅ Filters applied to cleaned dataset (affects Export tab and all views).")
                st.toast("Advanced filter applied!", icon="✅")
                st.rerun()

        with col2_adv:
            if st.button("↩️ Undo Last Filter", key="undo_advanced_filter_btn"):
                if st.session_state.df_backup is not None and not st.session_state.df_backup.empty:
                    st.session_state.df_clean = st.session_state.df_backup.copy()
                    st.success("🔁 Reverted to previous dataset (before this advanced filter).")
                    st.toast("Undo successful!", icon="↩️")
                    st.rerun()
                else:
                    st.warning("⚠️ No backup available to undo.")

        with col3_adv:
            if st.button("🔄 Reset to Original Data", key="reset_advanced_filter_btn"):
                if st.session_state.df_original is not None and not st.session_state.df_original.empty:
                    st.session_state.df_clean = st.session_state.df_original.copy()
                    st.session_state.df_full = st.session_state.df_original.copy() # Ensure df_full is also reset
                    st.success("✅ Reset to original uploaded dataset.")
                    st.toast("Dataset reset!", icon="🔄")
                    st.rerun()
                else:
                    st.warning("⚠️ No original dataset available.")
    else:
        st.info("Please add at least one filter condition to begin.")


# --- Export Tab ---
with tabs[6]:
    st.session_state.active_tab = tab_labels[6]
    st.subheader("⬇️ Export Cleaned CSV")
    st.info("ℹ️ The data shown below is the currently cleaned and filtered dataset ready for export.")

    # Use the current st.session_state.df_clean for export preview
    df_for_export = st.session_state.get("df_clean", pd.DataFrame())

    if df_for_export.empty:
        st.warning("⚠️ No data available to export.")
    else:
        export_view = st.radio("How much data to preview?", ["Top 5", "Top 50", "All"], horizontal=True, key="export_preview_radio")
        if export_view == "Top 5":
            st.dataframe(df_for_export.head(), use_container_width=True)
        elif export_view == "Top 50":
            st.dataframe(df_for_export.head(50), use_container_width=True)
        else:
            st.dataframe(df_for_export, use_container_width=True)

        st.markdown("---")
        # Ensure the CSV is encoded correctly
        csv = df_for_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Cleaned CSV",
            csv,
            "cleaned_data.csv",
            "text/csv",
            key="download_csv_btn"
        )
