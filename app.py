import streamlit as st
import pandas as pd
import numpy as np

# Cleaning function (from earlier)
def clean_csv_df(df):
    report = []

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    report.append(f"Removed {before - after} duplicate rows.")

    # Standardize column names
    old_columns = df.columns.tolist()
    df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    report.append(f"Standardized column names.")

    # Drop constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(columns=constant_cols, inplace=True)
    report.append(f"Dropped constant columns: {constant_cols}")

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
                report.append(f"Filled missing values in '{col}' with median.")
            else:
                df[col].fillna("missing", inplace=True)
                report.append(f"Filled missing text in '{col}' with 'missing'.")

    # Outlier detection (z-score)
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        z = (df[col] - df[col].mean()) / df[col].std()
        outliers = (abs(z) > 3).sum()
        if outliers > 0:
            report.append(f"Column '{col}' has {outliers} potential outliers (z > 3).")

    return df, report


# Streamlit App
st.title("🧹 CleanSheet: Data Cleaner for CSVs")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Original Data")
    st.dataframe(df.head())

    if st.button("Clean My Data"):
        cleaned_df, report = clean_csv_df(df)

        st.subheader("✅ Cleaned Data Preview")
        st.dataframe(cleaned_df.head())

        st.subheader("📝 Cleaning Report")
        for line in report:
            st.write("- " + line)

        # Download
        csv = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv")
