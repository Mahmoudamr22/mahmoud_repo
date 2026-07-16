"""
train_model.py
----------------
يعمل نفس خطوات المعالجة (preprocessing) اللي في نوتبوك Customer__Churn.ipynb
بالظبط، ويدرب موديل Decision Tree (أفضل F1-Score في تجاربك)، ويحفظ:
    - churn_model.pkl        (الموديل المدرب)
    - scaler.pkl              (StandardScaler المدرب على tenure/MonthlyCharges/TotalCharges)
    - feature_columns.pkl     (ترتيب/أسماء الأعمدة النهائية اللي الموديل اتدرب عليها)

شغّل السكريبت ده مرة واحدة عندك (لازم يكون معاك ملف telecom_churn_data.csv
في نفس الفولدر)، وهيطلعلك الملفات اللي تطبيق Streamlit محتاجها.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score


def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    """نفس خطوات المعالجة اللي في النوتبوك بالظبط."""
    df = raw_df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Demographics
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["Partner"] = df["Partner"].map({"Yes": 1, "No": 0})
    df["Dependents"] = df["Dependents"].map({"Yes": 1, "No": 0})

    # tenure group
    def segment_tenure(d):
        bins = [0, 12, 24, 48, d["tenure"].max()]
        labels = ["New", "Regular", "Loyal", "Super_Loyal"]
        d["tenure_group"] = pd.cut(d["tenure"], bins=bins, labels=labels, include_lowest=True)
        return d

    df = segment_tenure(df)
    encoding_dict = {"New": 0, "Regular": 1, "Loyal": 2, "Super_Loyal": 3}
    df["tenure_group"] = df["tenure_group"].map(encoding_dict).astype(int)

    # Contract / Billing / Payment
    contract_dict = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["Contract"] = df["Contract"].map(contract_dict)
    df["PaperlessBilling"] = df["PaperlessBilling"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, columns=["PaymentMethod"], drop_first=True, dtype=int)

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Services
    df["PhoneService"] = df["PhoneService"].map({"Yes": 1, "No": 0})
    df["MultipleLines"] = df["MultipleLines"].map(
        {"Yes": 1, "No": 0, "No phone service": 0}
    )
    df = pd.get_dummies(df, columns=["InternetService"], drop_first=True, dtype=int)

    services_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df[services_cols] = df[services_cols].replace(
        {"Yes": 1, "No": 0, "No internet service": 0}
    )

    # Target
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Feature engineering
    df["TotalServices"] = df[services_cols].sum(axis=1)

    return df


def main():
    raw_data = pd.read_csv("telecom_churn_data.csv")
    df = preprocess(raw_data)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1-Score:", f1_score(y_test, pred))
    print(classification_report(y_test, pred))

    joblib.dump(model, "churn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")
    print("\n✅ تم حفظ: churn_model.pkl, scaler.pkl, feature_columns.pkl")


if __name__ == "__main__":
    main()
