import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="AI Data Cleaner", layout="wide")
st.title("🧹 Smart Data Cleaning App")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("🔍 Raw Dataset Preview")
    st.dataframe(df.head())

    if st.button("Clean Dataset"):
        df_clean = df.copy()

        # Drop empty columns
        df_clean.dropna(axis=1, how='all', inplace=True)

        # Fill missing values
        for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        # Remove duplicates
        df_clean.drop_duplicates(inplace=True)

        # Outlier removal using IQR
        Q1 = df_clean.quantile(0.25)
        Q3 = df_clean.quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[~((df_clean < (Q1 - 1.5 * IQR)) | (df_clean > (Q3 + 1.5 * IQR))).any(axis=1)]

        # Encode categorical columns
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col])

        # Normalize numeric columns
        numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        df_clean[numeric_cols] = StandardScaler().fit_transform(df_clean[numeric_cols])

        st.success("✅ Dataset cleaned successfully!")
        st.subheader("🧼 Cleaned Dataset Preview")
        st.dataframe(df_clean.head())

        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned CSV", csv, "cleaned_dataset.csv", "text/csv")