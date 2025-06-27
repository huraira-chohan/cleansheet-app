import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="CleanSheet v2 - Smart CSV Cleaner", layout="wide")

st.title("🧹 CleanSheet v2")
st.subheader("Your smarter, customizable CSV data cleaning tool")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📄 Original Data Preview")
    st.dataframe(df.head())

    with st.form("cleaning_options"):
        st.write("### ⚙️ Cleaning Options")

        drop_duplicates = st.checkbox("Remove duplicate rows", value=True)
        fill_missing = st.selectbox("Handle missing values:", ["Do nothing", "Fill with Median (numeric only)", "Drop Rows with Nulls"])
        standardize_columns = st.checkbox("Standardize column names (lowercase, no spaces)", value=True)
        detect_outliers = st.checkbox("Detect numeric outliers (z-score > 3)", value=True)

        submitted = st.form_submit_button("🧼 Clean Data")

    if submitted:
        report = []

        if drop_duplicates:
            before = len(df)
            df.drop_duplicates(inplace=True)
            after = len(df)
            report.append(f"Removed {before - after} duplicate rows.")

        if standardize_columns:
            df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
            report.append("Standardized column names.")

        if fill_missing == "Fill with Median (numeric only)":
            num_cols = df.select_dtypes(include=np.number).columns
            for col in num_cols:
                nulls = df[col].isnull().sum()
                if nulls > 0:
                    df[col].fillna(df[col].median(), inplace=True)
                    report.append(f"Filled {nulls} nulls in '{col}' with median.")
        elif fill_missing == "Drop Rows with Nulls":
            before = len(df)
            df.dropna(inplace=True)
            after = len(df)
            report.append(f"Dropped {before - after} rows with null values.")

        if detect_outliers:
            for col in df.select_dtypes(include=np.number).columns:
                z = (df[col] - df[col].mean()) / df[col].std()
                outliers = (abs(z) > 3).sum()
                if outliers > 0:
                    report.append(f"'{col}' has {outliers} potential outliers (z > 3).")

        st.success("✅ Cleaning complete!")

        st.write("### 🧼 Cleaned Data Preview")
        st.dataframe(df.head())

        st.write("### 📝 Cleaning Report")
        for line in report:
            st.write("- ", line)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned CSV", data=csv, file_name="cleansheet_cleaned.csv", mime="text/csv")

