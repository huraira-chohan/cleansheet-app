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
import requests
from io import StringIO
from openai import OpenAI

# Setup
st.set_page_config(page_title="CleanSheet v9", layout="wide")
st.title("🧹 CleanSheet v9")
st.subheader("AI-powered CSV cleaning with assistant suggestions and smart controls")

# API key input
api_key = st.sidebar.text_input("🔐 Enter your OpenAI API Key", type="password")
client = OpenAI(api_key=api_key) if api_key else None

NULL_VALUES = ["", "na", "n/a", "null", "none", "-", "--", "NaN", "NAN", "?", "unknown"]

# --- Cleaning helpers ---
def clean_text(x):
    return str(x).strip().title() if pd.notnull(x) else x

def normalize_gender(g):
    g = str(g).strip().lower()
    if g in ["m", "male", "M"]:
        return "Male"
    elif g in ["f", "female", "F"]:
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
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$', str(email)))

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
    return series

# --- Assistant ---
def ask_ai_about_data(df):
    if not client:
        return "❌ Please enter a valid OpenAI API key."

    schema = {
        col: {
            "type": str(df[col].dtype),
            "missing": f"{df[col].isnull().mean()*100:.2f}%",
            "example": str(df[col].dropna().astype(str).head(1).values[0]) if not df[col].dropna().empty else "N/A"
        } for col in df.columns
    }

    prompt = f"""
You are a data cleaning assistant. Here's the dataset schema:\n{schema}
Suggest cleaning steps for each column: normalize, drop, outlier removal, parse date, etc. Be practical and concise.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {e}"

# --- Dataset Load ---
st.sidebar.markdown("### 📦 Load Dataset")
if st.sidebar.button("Load Titanic Dataset"):
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    uploaded_file = StringIO(requests.get(url).text)
    uploaded_file.name = "titanic.csv"
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.replace(NULL_VALUES, np.nan, inplace=True)
    submit = st.form_submit_button("🧼 Clean My Data")

if submit:
    for col, (action, fill, outliers) in col_config.items():
        # Cleaning logic remains the same...
        

    st.success("✅ Data cleaned successfully!")
    st.write("### 🧼 Final Cleaned Preview")
    st.dataframe(df.head())

    # ✅ MOVE THIS BUTTON OUTSIDE the form
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv_data, file_name="cleaned_data.csv", mime="text/csv")
    if st.checkbox("🤖 Show AI Assistant Suggestions", value=True):
        with st.spinner("Thinking..."):
            ai_response = ask_ai_about_data(df)
        st.markdown("### 💡 Assistant Recommendations")
        st.info(ai_response)

    # View control
    view_option = st.radio("🔍 How much data do you want to see?", ["Top 5 rows", "Top 50 rows", "All"], horizontal=True)
    if view_option == "Top 5 rows":
        st.dataframe(df.head(), use_container_width=True, height=400)
    elif view_option == "Top 50 rows":
        st.dataframe(df.head(50), use_container_width=True, height=600)
    else:
        st.dataframe(df.reset_index(drop=True), use_container_width=True, height=800)

    # --- Profiling ---
    st.write("### 🧪 Column Profiling")
    profile = pd.DataFrame({col: profile_column(df[col]) for col in df.columns}).T
    st.dataframe(profile)

    # --- Cleaning UI ---
    st.write("### ⚙️ Column Cleaning Options")
    col_config = {}

    with st.form("column_config"):
        drop_threshold = st.slider("Drop columns with more than X% missing values", 0, 100, 95)

        for col in df.columns:
            if profile.loc[col, "% missing"] > drop_threshold:
                st.warning(f"'{col}' has {profile.loc[col, '% missing']}% missing and will be dropped.")
                continue

            sample_vals = df[col].dropna().astype(str).head(10).tolist()
            default_type = "none"
            for val in sample_vals:
                if re.match(r'^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$', val): default_type = "email_validate"; break
                elif re.match(r'https?://', val): default_type = "url_validate"; break
                elif re.search(r'\d{1,4}[-/\s][A-Za-z]{3,}|\d{1,4}[-/\s]\d{1,2}', val): default_type = "date"; break
                elif val.lower() in ["m", "f", "male", "female"]: default_type = "gender"; break
                elif re.search(r'\+?\d{1,3}?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', val): default_type = "text_normalize"; break
                elif re.search(r'(usd|eur|inr|£|\$|€)', val.lower()): default_type = "text_normalize"; break
                elif re.search(r'^[A-Z]{2,3}$', val.strip()): default_type = "text_normalize"; break
            else:
                if df[col].dtype == object: default_type = "text_normalize"
                elif pd.api.types.is_numeric_dtype(df[col]): default_type = "numeric"

            index = ["none", "text_normalize", "numeric", "date", "gender", "email_validate", "url_validate", "drop"].index(default_type)

            with st.expander(f"🧠 How should we clean `{col}`?"):
                clean_type = st.radio("Cleaning method:", 
                                      options=["none", "text_normalize", "numeric", "date", "gender", "email_validate", "url_validate", "drop"],
                                      index=index, key=f"type_{col}")

            fill_opts = ["none", "drop_rows", "fill_mode"]
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_opts += ["fill_mean", "fill_median"]

            with st.expander(f"❓ Missing value strategy for `{col}`"):
                fill_missing = st.radio("Strategy:", options=fill_opts, key=f"null_{col}")

            outliers = False
            if pd.api.types.is_numeric_dtype(df[col]):
                with st.expander(f"⚠️ Outliers in `{col}`?"):
                    outliers = st.checkbox("Remove using Z-score", value=False, key=f"outlier_{col}")

            col_config[col] = (clean_type, fill_missing, outliers)

        if st.form_submit_button("🧼 Clean My Data"):
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
                elif fill == "fill_mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif fill == "fill_median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif fill == "fill_mode":
                    mode = df[col].mode().iloc[0] if not df[col].mode().empty else None
                    df[col].fillna(mode, inplace=True)

                if outliers:
                    df[col] = remove_outliers_zscore(df[col])

            st.success("✅ Data cleaned successfully!")
            st.write("### 🧼 Final Cleaned Preview")
            st.dataframe(df.head())

            st.download_button("📥 Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="cleaned_data.csv", mime="text/csv")


