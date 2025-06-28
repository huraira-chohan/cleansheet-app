import streamlit as st
import pandas as pd
import numpy as np
import re
from dateutil import parser
from difflib import get_close_matches
from rapidfuzz import fuzz, process
import phonenumbers
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CleanSheet v6 - Smartest CSV Cleaner", layout="wide")

st.title("🧹 CleanSheet v6")
st.subheader("AI-Powered Real-World CSV Cleaner with Smart Detection + AI Inference")

NULL_VALUES = ["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"]

KNOWN_COLUMN_TYPES = {
    "email": ["email", "e-mail", "user_email"],
    "phone": ["phone", "mobile", "contact", "phone_number"],
    "url": ["website", "url", "link"],
    "name": ["name", "full_name", "username"],
    "date": ["dob", "birthdate", "joined", "date"],
    "gender": ["gender", "sex"],
    "age": ["age", "years_old"],
    "salary": ["salary", "income", "pay"]
}


def ai_guess_column_type(col):
    vectorizer = TfidfVectorizer().fit([" ".join(v) for v in KNOWN_COLUMN_TYPES.values()])
    known_labels = list(KNOWN_COLUMN_TYPES.keys())
    known_vectors = vectorizer.transform([" ".join(v) for v in KNOWN_COLUMN_TYPES.values()])
    test_vector = vectorizer.transform([col.replace("_", " ")])
    similarities = cosine_similarity(test_vector, known_vectors).flatten()
    max_score = similarities.max()
    if max_score > 0.4:
        return known_labels[similarities.argmax()]
    return None

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
    return "other"

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def is_valid_phone(val):
    try:
        num = phonenumbers.parse(str(val), None)
        return phonenumbers.is_valid_number(num)
    except:
        return False

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def parse_any_date(date_str):
    try:
        return parser.parse(str(date_str), fuzzy=True)
    except:
        return np.nan

def is_constant_column(series):
    return series.nunique(dropna=False) <= 1

def clean_text_column(col):
    return col.fillna("").apply(lambda x: str(x).strip().title())

def fuzzy_dedupe(df, col):
    seen = set()
    mask = []
    for val in df[col].fillna("").astype(str):
        norm_val = val.strip().lower()
        if any(fuzz.ratio(norm_val, s) > 90 for s in seen):
            mask.append(False)
        else:
            seen.add(norm_val)
            mask.append(True)
    return df[mask]

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.replace(NULL_VALUES, np.nan, inplace=True)

    st.write("### 📄 Original Data Preview")
    st.dataframe(df.head())

    with st.form("cleaning_options"):
        st.write("### ⚙️ Cleaning Options")
        drop_duplicates = st.checkbox("Remove duplicate rows", value=True)
        fill_missing = st.selectbox("Handle missing values:", ["Do nothing", "Fill with Median (numeric only)", "Drop Rows with Nulls"])
        standardize_columns = st.checkbox("Standardize column names (lowercase, no spaces)", value=True)
        detect_outliers = st.checkbox("Detect numeric outliers (z-score > 3)", value=True)
        fuzzy_all = st.checkbox("Fuzzy deduplicate and normalize all text columns", value=True)
        submitted = st.form_submit_button("🧼 Clean Data")

    if submitted:
        report = []

        if standardize_columns:
            df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
            report.append("Standardized column names.")

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = clean_text_column(df[col])

        if fuzzy_all:
            for col in df.select_dtypes(include='object').columns:
                before = len(df)
                df = fuzzy_dedupe(df, col)
                after = len(df)
                if before != after:
                    report.append(f"Fuzzy de-duplicated {before - after} rows based on '{col}'.")

        if drop_duplicates:
            before = len(df)
            df.drop_duplicates(inplace=True)
            after = len(df)
            report.append(f"Removed {before - after} exact duplicate rows.")

        for col in df.columns:
            guessed_type = ai_guess_column_type(col)
            if guessed_type == "gender":
                df[col] = df[col].apply(normalize_gender)
                report.append(f"Normalized gender in column '{col}' (AI guessed).")
            elif guessed_type == "age":
                df[col] = df[col].apply(convert_text_to_number)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                report.append(f"Converted age values to numeric in column '{col}' (AI guessed).")
            elif guessed_type == "salary":
                df[col] = df[col].apply(clean_salary)
                report.append(f"Cleaned salary values in column '{col}' (AI guessed).")
            elif guessed_type == "date":
                df[col] = df[col].apply(parse_any_date)
                report.append(f"Parsed date values in column '{col}' (AI guessed).")
            elif guessed_type == "email":
                df[f"{col}_valid"] = df[col].apply(is_valid_email)
                report.append(f"Flagged email validity in column '{col}' (AI guessed).")
            elif guessed_type == "phone":
                df[f"{col}_valid"] = df[col].apply(is_valid_phone)
                report.append(f"Flagged phone number validity in column '{col}' (AI guessed).")
            elif guessed_type == "url":
                df[f"{col}_valid"] = df[col].apply(is_valid_url)
                report.append(f"Flagged URL validity in column '{col}' (AI guessed).")

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

        constant_cols = [col for col in df.columns if is_constant_column(df[col])]
        if constant_cols:
            df.drop(columns=constant_cols, inplace=True)
            report.append(f"Dropped constant columns: {constant_cols}")

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



