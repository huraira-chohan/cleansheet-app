import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="CleanSheet v3 - Smart CSV Cleaner", layout="wide")

st.title("🧹 CleanSheet v3")
st.subheader("AI-Enhanced Real-World CSV Cleaner")

# Helper functions
def clean_salary(s):
    try:
        return float(re.sub(r"[^\d.]", "", str(s)))
    except:
        return np.nan

def convert_text_to_number(text):
    text = str(text).lower()
    words_to_numbers = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50
    }
    return words_to_numbers.get(text.strip(), text)

def normalize_gender(g):
    g = str(g).strip().lower()
    if g in ["m", "male", "男"]:
        return "male"
    elif g in ["f", "female", "女"]:
        return "female"
    else:
        return "other"

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Standardize null-like values
    df.replace(["", "na", "NA", "NaN", "None"], np.nan, inplace=True)

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

        # Normalize gender column if it exists
        for col in df.columns:
            if df[col].astype(str).str.lower().isin(["m", "f", "male", "female", "男", "女"]).any():
                df[col] = df[col].apply(normalize_gender)
                report.append(f"Normalized gender in column '{col}'.")

        # Convert written numbers in 'age'-like columns
        for col in df.columns:
            if "age" in col:
                df[col] = df[col].apply(convert_text_to_number)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                report.append(f"Converted age values to numeric in column '{col}'.")

        # Clean salary columns
        for col in df.columns:
            if "salary" in col:
                df[col] = df[col].apply(clean_salary)
                report.append(f"Cleaned and converted salary values in column '{col}'.")

        # Clean join date columns
        for col in df.columns:
            if "date" in col:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                report.append(f"Parsed dates in column '{col}'.")

        # Remove invalid emails
        for col in df.columns:
            if "email" in col:
                invalids = df[~df[col].apply(is_valid_email)]
                count = len(invalids)
                df = df[df[col].apply(is_valid_email)]
                report.append(f"Removed {count} invalid email addresses in column '{col}'.")

        # Handle missing values
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

        # Drop constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            df.drop(columns=constant_cols, inplace=True)
            report.append(f"Dropped constant columns: {constant_cols}")

        # Outlier detection
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


