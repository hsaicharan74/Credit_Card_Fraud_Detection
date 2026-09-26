import streamlit as st
import joblib
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# SESSION HISTORY
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = joblib.load("fraud_model.pkl")
except Exception:
    st.error("Could not load fraud_model.pkl")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("💳 Fraud Detection")

    st.write("### About the Model")

    st.write(
        "Random Forest Machine Learning model used to "
        "detect potentially fraudulent credit card transactions."
    )

    st.write("### Features")
    st.write(
        "• Transaction Time\n"
        "• Transaction Amount\n"
        "• V1 to V28"
    )


# =========================================================
# HEADER
# =========================================================

st.title("💳 Credit Card Fraud Detection")

st.caption("Machine Learning Based Transaction Risk Analysis")

st.write(
    "Enter transaction details below to predict whether "
    "a transaction is Normal or Fraud."
)


# =========================================================
# SAMPLE TRANSACTION
# =========================================================

st.divider()

use_sample = st.checkbox(
    "🧪 Use Sample Fraud Transaction"
)


# =========================================================
# SAMPLE VALUES
# =========================================================

if use_sample:

    time_value = 4462.0
    amount_value = 239.93

    v1_value = -2.30334956758553
    v2_value = 1.759247460267
    v3_value = -0.359744743330052
    v4_value = 2.33024305053917
    v5_value = -0.821628328375422
    v6_value = -0.0757875706194599
    v7_value = 0.562319782266954
    v8_value = -0.399146578487216
    v9_value = -0.238253367661746
    v10_value = -1.52541162656194
    v11_value = 2.03291215755072
    v12_value = -6.56012429505962
    v13_value = 0.0229373234890961
    v14_value = -1.47010153611197
    v15_value = -0.698826068579047
    v16_value = -2.28219382856251
    v17_value = -4.78183085597533
    v18_value = -2.61566494476124
    v19_value = -1.3344106667307
    v20_value = -0.430021867171611
    v21_value = -0.294166317554753
    v22_value = -0.932391057274991
    v23_value = 0.172726295799422
    v24_value = -0.0873295379700724
    v25_value = -0.156114264651172
    v26_value = -0.542627889040196
    v27_value = 0.0395659889264757
    v28_value = -0.153028796529788

else:

    time_value = 0.0
    amount_value = 0.0

    v1_value = 0.0
    v2_value = 0.0
    v3_value = 0.0
    v4_value = 0.0
    v5_value = 0.0
    v6_value = 0.0
    v7_value = 0.0
    v8_value = 0.0
    v9_value = 0.0
    v10_value = 0.0
    v11_value = 0.0
    v12_value = 0.0
    v13_value = 0.0
    v14_value = 0.0
    v15_value = 0.0
    v16_value = 0.0
    v17_value = 0.0
    v18_value = 0.0
    v19_value = 0.0
    v20_value = 0.0
    v21_value = 0.0
    v22_value = 0.0
    v23_value = 0.0
    v24_value = 0.0
    v25_value = 0.0
    v26_value = 0.0
    v27_value = 0.0
    v28_value = 0.0


# =========================================================
# TRANSACTION DETAILS
# =========================================================

st.subheader("💰 Transaction Details")

col1, col2 = st.columns(2)

with col1:
    time = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=time_value,
        format="%.6f"
    )

with col2:
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=amount_value,
        format="%.2f"
    )


# =========================================================
# FEATURES
# =========================================================

st.subheader("🔢 Transaction Features")

col1, col2 = st.columns(2)

with col1:

    v1 = st.number_input("V1", value=v1_value, format="%.6f")
    v2 = st.number_input("V2", value=v2_value, format="%.6f")
    v3 = st.number_input("V3", value=v3_value, format="%.6f")
    v4 = st.number_input("V4", value=v4_value, format="%.6f")
    v5 = st.number_input("V5", value=v5_value, format="%.6f")
    v6 = st.number_input("V6", value=v6_value, format="%.6f")
    v7 = st.number_input("V7", value=v7_value, format="%.6f")
    v8 = st.number_input("V8", value=v8_value, format="%.6f")
    v9 = st.number_input("V9", value=v9_value, format="%.6f")
    v10 = st.number_input("V10", value=v10_value, format="%.6f")
    v11 = st.number_input("V11", value=v11_value, format="%.6f")
    v12 = st.number_input("V12", value=v12_value, format="%.6f")
    v13 = st.number_input("V13", value=v13_value, format="%.6f")
    v14 = st.number_input("V14", value=v14_value, format="%.6f")

with col2:

    v15 = st.number_input("V15", value=v15_value, format="%.6f")
    v16 = st.number_input("V16", value=v16_value, format="%.6f")
    v17 = st.number_input("V17", value=v17_value, format="%.6f")
    v18 = st.number_input("V18", value=v18_value, format="%.6f")
    v19 = st.number_input("V19", value=v19_value, format="%.6f")
    v20 = st.number_input("V20", value=v20_value, format="%.6f")
    v21 = st.number_input("V21", value=v21_value, format="%.6f")
    v22 = st.number_input("V22", value=v22_value, format="%.6f")
    v23 = st.number_input("V23", value=v23_value, format="%.6f")
    v24 = st.number_input("V24", value=v24_value, format="%.6f")
    v25 = st.number_input("V25", value=v25_value, format="%.6f")
    v26 = st.number_input("V26", value=v26_value, format="%.6f")
    v27 = st.number_input("V27", value=v27_value, format="%.6f")
    v28 = st.number_input("V28", value=v28_value, format="%.6f")


# =========================================================
# PREDICTION
# =========================================================

st.divider()

predict = st.button(
    "🔍 Predict Transaction",
    use_container_width=True
)

if predict:

    input_data = pd.DataFrame(
        [[
            time,
            v1, v2, v3, v4, v5,
            v6, v7, v8, v9, v10,
            v11, v12, v13, v14, v15,
            v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25,
            v26, v27, v28,
            amount
        ]],
        columns=[
            "Time",
            "V1", "V2", "V3", "V4", "V5",
            "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15",
            "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25",
            "V26", "V27", "V28",
            "Amount"
        ]
    )

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    result = "FRAUD" if prediction[0] == 1 else "NORMAL"

    st.session_state.history.append({
        "Prediction": result,
        "Fraud Probability": f"{probability * 100:.2f}%"
    })

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error("🚨 FRAUD TRANSACTION DETECTED")
    else:
        st.success("✅ NORMAL TRANSACTION")

    st.write(
        f"**Fraud Probability:** {probability * 100:.2f}%"
    )

    st.progress(float(probability))

    normal_probability = 1 - probability

    st.write(
        f"**Normal Probability:** {normal_probability * 100:.2f}%"
    )

    if probability >= 0.75:

        st.error("🔴 Risk Level: HIGH")

        st.write(
            "⚠️ Please review this transaction carefully."
        )

    elif probability >= 0.40:

        st.warning("🟡 Risk Level: MEDIUM")

        st.write(
            "🔎 This transaction may require additional verification."
        )

    else:

        st.success("🟢 Risk Level: LOW")

        st.write(
            "✅ This transaction appears to have a low fraud risk."
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Precision", "94.12%")

with col2:
    st.metric("Recall", "81.63%")

with col3:
    st.metric("F1 Score", "87.43%")

with col4:
    st.metric("ROC-AUC", "96.30%")

st.metric("PR-AUC", "87.34%")


# =========================================================
# MODEL ANALYSIS
# =========================================================

st.divider()

st.subheader("📊 Model Analysis")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "confusion_matrix.png",
        caption="Confusion Matrix",
        use_container_width=True
    )

with col2:
    st.image(
        "roc_curve.png",
        caption="ROC Curve",
        use_container_width=True
    )

st.image(
    "feature_importance.png",
    caption="Top 10 Important Features",
    use_container_width=True
)


# =========================================================
# PREDICTION HISTORY
# =========================================================

st.divider()

st.subheader("🧾 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

else:

    st.info("No predictions made yet.")


# =========================================================
# ABOUT PROJECT
# =========================================================

st.divider()

st.subheader("📌 About This Project")

st.write(
    "This application uses a Random Forest Machine Learning "
    "model to classify credit card transactions as Normal or Fraudulent."
)

st.write(
    "The model was trained using Time, Amount, and anonymized "
    "features V1–V28 from the credit card transaction dataset."
)