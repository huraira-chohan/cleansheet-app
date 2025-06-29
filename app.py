import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import requests
from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import requests

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Preview"

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

if load_sample:
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        response = requests.get(titanic_url)
        df = pd.read_csv(StringIO(response.text))
        st.session_state.df_clean = df.copy()
        st.success("✅ Sample Titanic dataset loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load sample data: {e}")

elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.replace(["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"], np.nan, inplace=True)
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
tab_labels = ["📊 Preview", "🧹 Clean", "🧮 Columns", "🔍 Filter", "📈 Sort", "🧠 Advanced Filter", "⬇️ Export"]
tab_index = tab_labels.index(st.session_state.active_tab)
tabs = st.tabs(tab_labels)
def auto_clean_column(series):
    import dateutil.parser

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

# --- Preview Tab ---
with tabs[0]:
    st.session_state.active_tab = tab_labels[0]
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
    with tabs[1]:
        st.session_state.active_tab = tab_labels[1]
        st.subheader("🧹 Clean Column Values")

    df = st.session_state.get("df_clean", pd.DataFrame()).copy()
    if df.empty:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    col = st.selectbox("Select a column to clean", df.columns)

    cleaning_action = st.selectbox(
        "Choose cleaning operation",
        ["-- Select --", "Remove NaNs", "Fill NaNs with 0", "To lowercase", "To title case", "Auto Clean"]
    )

    preview_col1, preview_col2 = st.columns(2)
    preview_col1.markdown("**Before Cleaning**")
    preview_col1.write(df[[col]].head(10))

    if cleaning_action == "Remove NaNs":
        df = df[df[col].notna()]
        st.success("✅ NaN rows removed.")

    elif cleaning_action == "Fill NaNs with 0":
        df[col] = df[col].fillna(0)
        st.success("✅ NaNs filled with 0.")

    elif cleaning_action == "To lowercase":
        df[col] = df[col].astype(str).str.lower()
        st.success("✅ Converted to lowercase.")

    elif cleaning_action == "To title case":
        df[col] = df[col].astype(str).str.title()
        st.success("✅ Converted to title case.")

    elif cleaning_action == "Auto Clean":
        import dateutil.parser
        def auto_clean_column(series):
            try:
                numeric = pd.to_numeric(series, errors='coerce')
                if numeric.notna().sum() >= 0.8 * len(series):
                    return numeric
            except: pass

            try:
                dates = pd.to_datetime(series, errors='coerce')
                if dates.notna().sum() >= 0.8 * len(series):
                    return dates
            except: pass

            cat = series.astype(str).str.strip().str.lower().replace({
                "m": "Male", "f": "Female",
                "male": "Male", "female": "Female",
                "yes": "Yes", "y": "Yes", "1": "Yes",
                "no": "No", "n": "No", "0": "No",
                "forty": "40", "thirty": "30", "twenty": "20"
            })
            return cat

        df[col] = auto_clean_column(df[col])
        st.success("✅ Auto-cleaning applied to column.")

    # Show cleaned preview
    preview_col2.markdown("**After Cleaning**")
    preview_col2.write(df[[col]].head(10))

    # Update the cleaned version in session
    st.session_state.df_clean = df.copy()

    st.markdown("---")
    st.dataframe(df, use_container_width=True)

    st.subheader("🔎 Clean Columns")
    st.session_state.active_tab = tab_labels[1]
    columns = df.columns.tolist()
st.session_state.active_tab = tab_labels[1]
    st.subheader("🧹 Clean Column Values")

    df = st.session_state.get("df_clean", pd.DataFrame()).copy()

    if df.empty:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    selected_col = st.selectbox("Select column to clean", df.columns)

    clean_opt = st.selectbox(
        "Choose cleaning operation",
        ["-- Select --", "Remove NaNs", "Fill NaNs with 0", "Convert to lowercase", "Convert to title case", "Auto Detect & Clean"]
    )

    if clean_opt != "-- Select --":
        if clean_opt == "Remove NaNs":
            df = df[df[selected_col].notna()]
            st.success("✅ NaN values removed.")

        elif clean_opt == "Fill NaNs with 0":
            df[selected_col] = df[selected_col].fillna(0)
            st.success("✅ NaNs filled with 0.")

        elif clean_opt == "Convert to lowercase":
            df[selected_col] = df[selected_col].astype(str).str.lower()
            st.success("✅ Converted to lowercase.")

        elif clean_opt == "Convert to title case":
            df[selected_col] = df[selected_col].astype(str).str.title()
            st.success("✅ Converted to title case.")

        elif clean_opt == "Auto Detect & Clean":
            df[selected_col] = auto_clean_column(df[selected_col])
            st.success("✅ Auto cleaning applied.")

        # Update session state
        st.session_state.df_clean = df.copy()
        st.dataframe(df, use_container_width=True)
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
with tabs[2]:
    st.session_state.active_tab = tab_labels[2]
    columns = df.columns.tolist()
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
with tabs[3]:
    st.session_state.active_tab = tab_labels[3]
    st.subheader("🔍 Filter Rows (temporary view only)")

    # Load clean dataset
    df = st.session_state.get("df_clean", pd.DataFrame()).copy()

    # Save for filtering preview
    if "df_temp" not in st.session_state:
        st.session_state.df_temp = df.copy()

    col_to_filter = st.selectbox("Choose column to filter", df.columns.tolist())
    filtered_df = df.copy()

    # Numeric filtering
    if pd.api.types.is_numeric_dtype(df[col_to_filter]):
        st.write(f"📏 Numeric Range Filter for `{col_to_filter}`")
        full_col = st.session_state.df_original[col_to_filter] if "df_original" in st.session_state else df[col_to_filter]
        min_val, max_val = float(full_col.min()), float(full_col.max())

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
            filtered_df = df[df[col_to_filter].between(start, end)]
            st.info(f"Showing rows where `{col_to_filter}` is between {start:.2f} and {end:.2f}.")

    # Datetime filtering
    elif pd.api.types.is_datetime64_any_dtype(df[col_to_filter]) or "date" in col_to_filter.lower():
        st.write(f"🗓 Date Range Filter for `{col_to_filter}`")
        df[col_to_filter] = pd.to_datetime(df[col_to_filter], errors='coerce')
        full_col = st.session_state.df_original[col_to_filter] if "df_original" in st.session_state else df[col_to_filter]
        min_date = pd.to_datetime(full_col.min(), errors="coerce")
        max_date = pd.to_datetime(full_col.max(), errors="coerce")

        if pd.isnull(min_date) or pd.isnull(max_date):
            st.warning(f"⚠️ Could not convert `{col_to_filter}` to datetime.")
        else:
            date_start, date_end = st.date_input(
                "Select date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            filtered_df = df[df[col_to_filter].between(date_start, date_end)]
            st.info(f"Showing rows where `{col_to_filter}` is between {date_start} and {date_end}.")

    # Categorical filtering
    else:
        st.write(f"🔠 Categorical Filter for `{col_to_filter}`")
        full_col = st.session_state.df_original[col_to_filter] if "df_original" in st.session_state else df[col_to_filter]
        unique_vals = full_col.dropna().unique().tolist()

        if not unique_vals:
            st.warning("⚠️ No valid values to filter.")
        else:
            selected_vals = st.multiselect(
                f"Select values to include from `{col_to_filter}`:",
                options=unique_vals
            )
            if selected_vals:
                filtered_df = df[df[col_to_filter].isin(selected_vals)]
                st.info(f"Filtered by selected values in `{col_to_filter}`.")
            else:
                filtered_df = df.copy()
                st.caption("No filter applied — showing full dataset.")

    # Show filtered data
    st.dataframe(filtered_df, use_container_width=True)

    # --- Apply filter to export ---
    if st.button("✅ Apply Filter"):
        st.session_state.df_backup = st.session_state.df_clean.copy()  # for undo
        st.session_state.df_clean = filtered_df.copy()  # permanent export update
        st.session_state.df_temp = filtered_df.copy()   # temporary preview update
        st.success("✅ Filter applied to exported dataset.")

    # --- Undo / Reset ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button("↩️ Undo Last Filter"):
            if "df_backup" in st.session_state:
                st.session_state.df_clean = st.session_state.df_backup.copy()
                st.session_state.df_temp = st.session_state.df_backup.copy()
                st.success("🔁 Undo successful. Reverted to previous filtered dataset.")
            else:
                st.warning("⚠️ No backup available to undo.")

    with col2:
        if st.button("🔄 Reset to Original Uploaded Dataset"):
            if "df_original" in st.session_state:
                st.session_state.df_clean = st.session_state.df_original.copy()
                st.session_state.df_temp = st.session_state.df_original.copy()
                st.success("✅ Reset to original uploaded dataset.")
            else:
                st.warning("⚠️ No original dataset found.")




# --- Sort Tab ---
with tabs[4]:
    st.session_state.active_tab = tab_labels[4]
    columns = df.columns.tolist()
    st.subheader("📈 Sort Data")
    st.info("ℹ️ Use the Reset button at the top to undo any sort or filter applied.")
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

with tabs[5]:
    st.session_state.active_tab = tab_labels[5]
    st.subheader("🧠 Advanced Multi-Column Filtering (Export Only)")

    df = st.session_state.get("df_clean", pd.DataFrame()).copy()

    if df.empty:
        st.warning("⚠️ No dataset loaded.")
        st.stop()

    num_conditions = st.number_input("How many filter conditions?", min_value=1, max_value=5, value=1)
    logic = st.radio("Combine filters using:", ["AND", "OR"], horizontal=True)

    conditions = []
    for i in range(int(num_conditions)):
        st.markdown(f"### ➕ Condition #{i+1}")
        col = st.selectbox(f"Choose column", df.columns, key=f"adv_col_{i}")
        dtype = df[col].dtype

        # Numeric
        if pd.api.types.is_numeric_dtype(dtype):
            min_val, max_val = float(df[col].min()), float(df[col].max())
            if min_val == max_val:
                st.warning(f"⚠️ All values in `{col}` are the same: {min_val}")
                cond = pd.Series([True] * len(df))
            else:
                range_val = st.slider(f"Range for `{col}`", min_val, max_val, (min_val, max_val), key=f"adv_range_{i}")
                cond = df[col].between(range_val[0], range_val[1])

        # Datetime
        elif pd.api.types.is_datetime64_any_dtype(dtype) or "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
            min_date, max_date = df[col].min(), df[col].max()
            if pd.isnull(min_date) or pd.isnull(max_date):
                st.warning(f"⚠️ Cannot parse `{col}` as datetime.")
                cond = pd.Series([True] * len(df))
            else:
                start_date, end_date = st.date_input(f"Date range for `{col}`", (min_date, max_date), key=f"adv_date_{i}")
                cond = df[col].between(start_date, end_date)

        # Categorical
        else:
            values = df[col].dropna().unique().tolist()
            selected = st.multiselect(f"Select values for `{col}`", values, key=f"adv_cat_{i}")
            cond = df[col].isin(selected) if selected else pd.Series([True] * len(df))

        conditions.append(cond)

    # Combine all
    if conditions:
        combined = conditions[0]
        for c in conditions[1:]:
            combined = combined & c if logic == "AND" else combined | c

        # Safe length check
        if len(combined) == len(df):
            filtered_df = df[combined]
            st.dataframe(filtered_df, use_container_width=True)
            st.success(f"✅ {len(filtered_df)} rows matched your filters.")
        else:
            st.error("❌ Filter condition length mismatch.")
            st.stop()

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Apply Filters to Export"):
            st.session_state.df_export = filtered_df.copy()
            st.success("✅ Filters applied to export only.")

    with col2:
        if st.button("↩️ Undo Export Filter"):
            if "df_clean" in st.session_state:
                st.session_state.df_export = st.session_state.df_clean.copy()
                st.info("🔁 Reverted export to cleaned dataset.")
            else:
                st.warning("⚠️ No cleaned dataset found.")

    with col3:
        if st.button("🔄 Reset Export to Original"):
            if "df_original" in st.session_state:
                st.session_state.df_export = st.session_state.df_original.copy()
                st.success("✅ Export reset to original dataset.")
            else:
                st.warning("⚠️ No original dataset found.")



# --- Export Tab ---
with tabs[6]:
    st.session_state.active_tab = tab_labels[6]
    st.subheader("⬇️ Export Cleaned CSV")

    # Use export-specific filtered data if available
    export_df = st.session_state.get("df_export", st.session_state.get("df_clean", pd.DataFrame()))

    export_view = st.radio("How much data to preview?", ["Top 5", "Top 50", "All"], horizontal=True, key="export_view")
    if export_view == "Top 5":
        st.dataframe(export_df.head(), use_container_width=True)
    elif export_view == "Top 50":
        st.dataframe(export_df.head(50), use_container_width=True)
    else:
        st.dataframe(export_df, use_container_width=True)

    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")

