```markdown
# 🧹 Smart Data Cleaning App

Upload, clean, and download your dataset — now with outlier removal, encoding, and normalization!

## 🚀 Features
- Upload `.csv` or `.xlsx`
- Remove duplicates, empty columns
- Fill missing values (median/mode)
- Detect and remove outliers (IQR method)
- Normalize numeric columns (StandardScaler)
- Encode categorical columns (LabelEncoder)
- Preview raw and cleaned dataset
- Download cleaned file as `.csv`

## 🛠 Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```