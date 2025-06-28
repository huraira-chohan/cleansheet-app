import streamlit as st
import pandas as pd
import numpy as np
import re
from dateutil import parser
from rapidfuzz import fuzz, process
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

st.set_page_config(page_title="CleanSheet v9 - Smartest Data Cleaner", layout="wide")

st.title("🧹 CleanSheet v9")
st.subheader("Your all-in-one AI-powered CSV data cleaning app with full control and outlier handling")

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

# --- UTILITY FUNCTIONS ---
def ai_guess_column_type(col):
    vectorizer = TfidfVectorizer().fit([" ".join(v) for v in KNOWN_COLUMN_TYPES.values()])
    known_labels = list(KNOWN_COLUMN_TYPES.keys())
    known_vectors = vectorizer.transform([" ".join(v) for v in KNOWN_COLUMN_TYPES.values()])
    test_vector = vectorizer.transform([col.replace("_", " ")])
    similarities = cosine_similarity(test_vector, known_vectors).flatten()
    max_score = similarities.max()
    if max_score > 0.4:
        return known_labels[similarities.argmax()]
    return "unknown"

def clean_text(x):
    return str(x).strip().title() if pd.notnull(x) else x

def normalize_gender(g):
    g = str(g).strip().lower()
    if g in ["m", "male"]:
        return "Male"
    elif g in ["f", "female"]:
        return "Female"
    return "Other"

def convert_to_numeric(x):
    try:
        return float(re.sub(r"[^0-9.]+", "", str(x)))
    except:
        return np.nan

def parse_any_date(date_str):
    try:
        return parser.parse(str(date_str), fuzzy=True)
    except:
        return np.nan

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def profile_column(col_data):
    return {
        "type": col_data.dtype,
        "% missing": round(col_data.isnull().mean() * 100, 2),
        "unique": col_data.nunique(),
        "top": col_data.value_counts().index[0] if not col_data.dropna().empty else "",
        "top freq": col_data.value_counts().iloc[0] if not col_data.dropna().empty else 0
    }

def remove_outliers_zscore(series, threshold=3):
    if pd.api.types.is_numeric_dtype(series):
        non_na = series.dropna()
        z = np.abs(stats.zscore(non_na))
        mask = (z < threshold)
        filtered = non_na[mask]
        return series.where(series.isin(filtered))
    return series[(z < threshold).reindex(series.index, fill_value=False)]
    return series

# --- APP INTERFACE ---
import requests
from io import StringIO

st.sidebar.markdown("### 📦 Load Sample Dataset")
if st.sidebar.button("Load Titanic Dataset"):
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    response = requests.get(titanic_url)
    uploaded_file = StringIO(response.text)
    uploaded_file.name = "titanic.csv"
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.replace(NULL_VALUES, np.nan, inplace=True)
    st.write("### 📄 Data Preview")
    st.dataframe(df.head())

    st.write("### 🧪 Column Profiling")
    profile = pd.DataFrame({col: profile_column(df[col]) for col in df.columns}).T
    st.dataframe(profile)

    st.write("### ⚙️ Column Cleaning Options")
    col_config = {}
    with st.form("column_config"):
        drop_threshold = st.slider("Drop columns with more than X% missing values", 0, 100, 95)
        for col in df.columns:
            if profile.loc[col, "% missing"] > drop_threshold:
                st.warning(f"'{col}' exceeds missing threshold ({profile.loc[col, '% missing']}%). Will be dropped.")
                continue

            guessed = ai_guess_column_type(col)
            st.markdown(f"#### Column: `{col}` (AI guess: `{guessed}`)")
            type_map = {
                "name": "text_normalize",
                "age": "numeric",
                "salary": "numeric",
                "date": "date",
                "gender": "gender",
                "email": "email_validate",
                "url": "url_validate"
            }
            default_type = type_map.get(guessed, "none")
            index = ["none", "text_normalize", "numeric", "date", "gender", "email_validate", "url_validate", "drop"].index(default_type)
            clean_type = st.selectbox(
                f"Cleaning rule for `{col}`:",
                ["none", "text_normalize", "numeric", "date", "gender", "email_validate", "url_validate", "drop"],
                index=index,
                key=f"type_{col}"
            )

            numeric_fill_options = ["none", "drop_rows", "fill_mean", "fill_median", "fill_mode"]
categorical_fill_options = ["none", "drop_rows", "fill_mode"]
is_numeric = pd.api.types.is_numeric_dtype(df[col])
fill_missing = st.selectbox(
    f"Missing value handling for `{col}`:",
    numeric_fill_options if is_numeric else categorical_fill_options,
    key=f"null_{col}"
)
            handle_outliers = st.checkbox(
    f"Remove outliers from `{col}` using Z-score", 
    value=False, 
    key=f"outlier_{col}") if is_numeric else False
            col_config[col] = (clean_type, fill_missing, handle_outliers)

        submit = st.form_submit_button("🧼 Clean My Data")

    if submit:
        for col, (action, fill, outliers) in col_config.items():
            if action == "drop":
                df.drop(columns=col, inplace=True)
                continue

            if action == "text_normalize":
                df[col] = df[col].apply(clean_text)
            elif action == "numeric":
                df[col] = df[col].apply(convert_to_numeric)
            elif action == "date":
                df[col] = df[col].apply(parse_any_date)
            elif action == "gender":
                df[col] = df[col].apply(normalize_gender)
            elif action == "email_validate":
                df[f"{col}_valid"] = df[col].apply(is_valid_email)
            elif action == "url_validate":
                df[f"{col}_valid"] = df[col].apply(is_valid_url)

            if fill == "drop_rows":
                df = df[df[col].notna()]
            elif fill == "fill_mean" and df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].mean(), inplace=True)
            elif fill == "fill_median" and df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].median(), inplace=True)
            elif fill == "fill_mode":
                mode = df[col].mode().iloc[0] if not df[col].mode().empty else None
                df[col].fillna(mode, inplace=True)

            if outliers:
                df[col] = remove_outliers_zscore(df[col])

        st.success("✅ Cleaning complete!")
        st.write("### ✅ Cleaned Data Preview")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned CSV", data=csv, file_name="cleansheet_cleaned.csv", mime="text/csv")


