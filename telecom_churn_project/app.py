"""
app.py
------
تطبيق Streamlit للتنبؤ بـ Customer Churn.
شغّل الأمر: streamlit run app.py
لازم يكون معاك في نفس الفولدر:
    churn_model.pkl, scaler.pkl, feature_columns.pkl
(الناتجة من train_model.py)
"""

import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns


model, scaler, feature_columns = load_artifacts()

st.title("📉 Customer Churn Prediction")
st.write("املأ بيانات العميل وهيتوقع الموديل هل العميل متوقع يعمل Churn (يسيب الخدمة) ولا لأ.")

st.header("👤 بيانات شخصية")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("النوع", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen؟", ["No", "Yes"])
with col2:
    partner = st.selectbox("متجوز/مرتبط (Partner)؟", ["No", "Yes"])
    dependents = st.selectbox("عنده معالين (Dependents)؟", ["No", "Yes"])

st.header("📄 بيانات الحساب والتعاقد")
col3, col4 = st.columns(2)
with col3:
    tenure = st.number_input("عدد شهور الاشتراك (tenure)", min_value=0, max_value=100, value=12)
    contract = st.selectbox("نوع العقد", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("فاتورة إلكترونية (Paperless Billing)؟", ["No", "Yes"])
with col4:
    payment_method = st.selectbox(
        "طريقة الدفع",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.number_input("المصاريف الشهرية (MonthlyCharges)", min_value=0.0, value=70.0, step=1.0)
    total_charges = st.number_input("إجمالي المصاريف (TotalCharges)", min_value=0.0, value=800.0, step=1.0)

st.header("📡 الخدمات المشترك فيها")
col5, col6 = st.columns(2)
with col5:
    phone_service = st.selectbox("خدمة التليفون (PhoneService)؟", ["No", "Yes"])
    multiple_lines = st.selectbox("خطوط متعددة (MultipleLines)؟", ["No", "Yes"])
    internet_service = st.selectbox("خدمة الإنترنت", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security؟", ["No", "Yes"])
with col6:
    online_backup = st.selectbox("Online Backup؟", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection؟", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support؟", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV؟", ["No", "Yes"])

streaming_movies = st.selectbox("Streaming Movies؟", ["No", "Yes"])


def build_input_row():
    yn = {"Yes": 1, "No": 0}
    services_flags = {
        "OnlineSecurity": yn[online_security],
        "OnlineBackup": yn[online_backup],
        "DeviceProtection": yn[device_protection],
        "TechSupport": yn[tech_support],
        "StreamingTV": yn[streaming_tv],
        "StreamingMovies": yn[streaming_movies],
    }

    # tenure_group (نفس تقسيم النوتبوك)
    if tenure <= 12:
        tenure_group = 0
    elif tenure <= 24:
        tenure_group = 1
    elif tenure <= 48:
        tenure_group = 2
    else:
        tenure_group = 3

    contract_dict = {"Month-to-month": 0, "One year": 1, "Two year": 2}

    row = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": yn[partner],
        "Dependents": yn[dependents],
        "tenure": tenure,
        "PhoneService": yn[phone_service],
        "MultipleLines": yn[multiple_lines],
        "Contract": contract_dict[contract],
        "PaperlessBilling": yn[paperless],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "tenure_group": tenure_group,
        **services_flags,
    }
    row["TotalServices"] = sum(services_flags.values())

    # PaymentMethod dummies (drop_first=True شال 'Bank transfer (automatic)')
    row["PaymentMethod_Credit card (automatic)"] = 1 if payment_method == "Credit card (automatic)" else 0
    row["PaymentMethod_Electronic check"] = 1 if payment_method == "Electronic check" else 0
    row["PaymentMethod_Mailed check"] = 1 if payment_method == "Mailed check" else 0

    # InternetService dummies (drop_first=True شال 'DSL')
    row["InternetService_Fiber optic"] = 1 if internet_service == "Fiber optic" else 0
    row["InternetService_No"] = 1 if internet_service == "No" else 0

    return pd.DataFrame([row])


if st.button("🔮 توقع Churn", type="primary", use_container_width=True):
    input_df = build_input_row()

    # نفس الـ scaler اللي اتدرب في النوتبوك
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # ترتيب الأعمدة زي ما الموديل اتدرب بالظبط (وأي عمود ناقص = صفر)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ العميل متوقع يعمل **Churn** (احتمال: {proba:.1%})")
    else:
        st.success(f"✅ العميل متوقع **يفضل مشترك** (احتمال الـ Churn: {proba:.1%})")

    st.progress(float(proba))
