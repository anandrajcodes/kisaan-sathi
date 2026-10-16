import requests
import streamlit as st
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from datetime import datetime
import sys
from pathlib import Path

BACKEND_ML_PATH = (
    Path(__file__).resolve().parents[1] / "backend" / "ml"
)

if str(BACKEND_ML_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_ML_PATH))

from explain_prediction import explain_prediction

def generate_soil_report(input_values, predicted_crop, top_candidates):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Kissaan AI", styles["Title"]))
    story.append(
        Paragraph("Soil & Crop Analysis Report", styles["Heading2"])
    )
    story.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 20))

    story.append(Paragraph("1. Input Measurements", styles["Heading2"]))

    input_rows = [["Parameter", "Value"]]
    for key, value in input_values.items():
        input_rows.append([str(key), str(value)])

    input_table = Table(input_rows, colWidths=[220, 240])
    input_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B5E20")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(input_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("2. Predicted Crop", styles["Heading2"]))
    story.append(Paragraph(str(predicted_crop), styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. Top Crop Candidates", styles["Heading2"]))

    candidate_rows = [["Crop", "Model Score"]]
    for candidate in top_candidates:
        candidate_rows.append([
            str(candidate["crop"]),
            f'{float(candidate["model_score"]) * 100:.2f}%',
        ])

    candidate_table = Table(candidate_rows, colWidths=[220, 240])
    candidate_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(candidate_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("4. Important Limitations", styles["Heading2"]))
    story.append(
        Paragraph(
            "This report contains dataset-based crop suitability predictions. "
            "Model scores are not guaranteed probabilities of crop success. "
            "The system does not provide validated fertilizer dosages or a "
            "complete soil-health diagnosis. Consult suitable soil-test "
            "results and local agricultural experts before making farm "
            "decisions.",
            styles["Normal"],
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------- CONFIGURATION ----------------
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Kissaan AI | Soil Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
    .stApp {
        background: #f5f8f2;
        color: #20352a;
    }

    [data-testid="stSidebar"] {
        background: #173d2b;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .hero {
        background: linear-gradient(120deg, #173d2b, #367b4b);
        padding: 30px;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 24px;
    }

    .hero h1, .hero p {
        color: #ffffff !important;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #286b3c !important;
        color: #ffffff !important;
        border-radius: 9px;
        border: none !important;
        min-height: 44px;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #1d512e !important;
        color: #ffffff !important;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce7d9;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(25, 60, 35, 0.06);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        color: #526456 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: #1f5935 !important;
        font-weight: 700;
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #263d2d !important;
        font-weight: 600 !important;
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #20352a !important;
        border: 1px solid #cbd9c9 !important;
        border-radius: 9px !important;
    }

    /* Explicit readable alert colors: avoid pale yellow text/background clashes */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] li {
        color: #20352a !important;
    }

    [data-testid="stAlert"][kind="success"] {
        background: #e5f4e8 !important;
        border: 1px solid #b9dfc1 !important;
    }

    [data-testid="stAlert"][kind="info"] {
        background: #e8f1fb !important;
        border: 1px solid #c6dbf3 !important;
    }

    [data-testid="stAlert"][kind="warning"] {
        background: #fff3d6 !important;
        border: 1px solid #f0d28a !important;
    }

    [data-testid="stAlert"][kind="error"] {
        background: #fde8e7 !important;
        border: 1px solid #f2c0bd !important;
    }

    .prediction-card {
        background: linear-gradient(120deg, #173d2b, #367b4b);
        padding: 24px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 20px;
    }

    .prediction-card p {
        color: #e0f2df !important;
        margin: 0;
    }

    .prediction-card h1 {
        color: #ffffff !important;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- API HELPERS ----------------
def call_api(endpoint, payload):
    try:
        response = requests.post(
            f"{API_URL}/{endpoint}",
            json=payload,
            timeout=15,
        )
        if response.ok:
            return response.json()

        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        st.error(f"Request failed ({response.status_code}): {detail}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the backend. Start FastAPI in a separate "
            "terminal using: python -m uvicorn main:app --reload"
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"API request failed: {exc}")

    return None


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🌱 Kissaan AI")
    st.caption("SOIL INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Crop Recommendation",
        "Soil Health",
        "Fertilizer Guidance",
        "Model Performance",
        "About Project",
    ],
)

    st.divider()
    st.caption("AI-assisted agricultural decision support")
    st.caption("Predictions depend on input quality and dataset limits.")


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🌾 Smarter Farming Starts Here</h1>
    <p>Understand your soil. Explore suitable crops. Make informed decisions.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- DASHBOARD ----------------

if page == "Dashboard":
    st.markdown(
        """
        <div style="
            padding: 28px;
            border-radius: 18px;
            background: linear-gradient(120deg, #123d2b, #23734b);
            color: white;
            margin-bottom: 24px;
        ">
            <h1 style="color: white; margin-bottom: 8px;">
                🌱 Welcome to Kissaan AI
            </h1>
            <p style="font-size: 16px; margin-bottom: 0;">
                Smarter crop suitability predictions and soil insights
                powered by machine learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Your Soil Intelligence Dashboard")
    st.write(
        "Explore crop suitability, assess soil pH, review nutrient "
        "measurements, and understand the model's performance."
    )

    # Project statistics
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            label="🤖 AI Model",
            value="Random Forest",
        )

    with c2:
        st.metric(
            label="📊 Training Records",
            value="2,200",
        )

    with c3:
        st.metric(
            label="🌾 Crop Classes",
            value="22",
        )

    st.divider()

    # Feature cards
    st.markdown("### Explore the Platform")

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            """
            <div style="
                background: #e8f5e9;
                padding: 20px;
                border-radius: 14px;
                min-height: 190px;
                border-top: 4px solid #2e7d32;
            ">
                <h3 style="color: #1b5e20;">🌱 Crop Recommendation</h3>
                <p style="color: #26382a;">
                    Estimate suitable crops using soil and weather inputs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            """
            <div style="
                background: #e3f2fd;
                padding: 20px;
                border-radius: 14px;
                min-height: 190px;
                border-top: 4px solid #1976d2;
            ">
                <h3 style="color: #0d47a1;">🧪 Soil Health</h3>
                <p style="color: #263746;">
                    Assess soil pH and record nitrogen, phosphorus,
                    and potassium measurements.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            """
            <div style="
                background: #fff3e0;
                padding: 20px;
                border-radius: 14px;
                min-height: 190px;
                border-top: 4px solid #ef6c00;
            ">
                <h3 style="color: #e65100;">🌿 Fertilizer Guidance</h3>
                <p style="color: #4a3524;">
                    Review nutrient inputs and learn why reliable
                    fertilizer dosage needs additional reference data.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### ⚠️ Important Information")

    st.info(
        "Kissaan AI provides dataset-based crop suitability predictions "
        "and basic soil measurements. It does not diagnose complete soil "
        "health or provide validated fertilizer dosages. Always use "
        "appropriate soil-test results and local agricultural guidance "
        "for real farming decisions."
    )

    st.caption(
        "Academic mini project · Machine Learning · Soil Intelligence"
    )

# ---------------- CROP RECOMMENDATION ----------------

elif page == "Crop Recommendation":
    st.subheader("🌱 AI Crop Recommendation")
    st.write(
        "Adjust the soil and weather conditions using the sliders below."
    )

    with st.form("crop_form"):
        col1, col2 = st.columns(2)

        with col1:
            nitrogen = st.slider(
                "Nitrogen (N)", min_value=0, max_value=140, value=90, step=1
            )
            phosphorus = st.slider(
                "Phosphorus (P)", min_value=5, max_value=145, value=42, step=1
            )
            potassium = st.slider(
                "Potassium (K)", min_value=5, max_value=205, value=43, step=1
            )
            ph = st.slider(
                "Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1
            )

        with col2:
            temperature = st.slider(
                "Temperature (°C)", min_value=-10.0, max_value=60.0,
                value=25.0, step=0.5
            )
            humidity = st.slider(
                "Humidity (%)", min_value=0.0, max_value=100.0,
                value=80.0, step=1.0
            )
            rainfall = st.slider(
                "Rainfall (mm)", min_value=0.0, max_value=5000.0,
                value=100.0, step=10.0
            )

        submitted = st.form_submit_button("Predict Suitable Crop")

    if submitted:
        result = call_api(
            "predict-crop",
            {
                "N": nitrogen,
                "P": phosphorus,
                "K": potassium,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            },
        )

        if result:
            st.markdown("### 🌾 Top 3 Crop Predictions")
            predicted_crop = result.get("predicted_crop", "Unknown")
            candidates = result.get("top_3_candidates", [])

            st.markdown(
                f"""
                <div class="prediction-card">
                    <p>BEST MODEL PREDICTION</p>
                    <h1>🌱 {predicted_crop.title()}</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if candidates:
                for index, candidate in enumerate(candidates):
                    crop_name = candidate["crop"].title()
                    score = float(candidate["model_score"])
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**{index + 1}. {crop_name}**")
                        st.progress(max(0.0, min(1.0, score)))

                    with col2:
                        st.metric("Model score", f"{score * 100:.2f}%")
            else:
                st.info(
                    "The API did not return alternative crop candidates."
                )

            st.info(
                "Scores represent the model's relative confidence, not "
                "verified probabilities of successful cultivation or yield."
            )
            
            
            # Farmer-friendly crop explanation
            st.markdown("### 🌱 Why Is This Crop Recommended?")

            try:
                
                explanation_inputs = {
                    "N": nitrogen,
                    "P": phosphorus,
                    "K": potassium,
                    "temperature": temperature,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall,
                }

                explanations = explain_prediction(explanation_inputs)


                feature_names = {
                    "N": "Nitrogen in the soil",
                    "P": "Phosphorus in the soil",
                    "K": "Potassium in the soil",
                    "temperature": "Temperature",
                    "humidity": "Air moisture (humidity)",
                    "ph": "Soil acidity or alkalinity (pH)",
                    "rainfall": "Rainfall",
                }

                positive_features = [
                    item for item in explanations
                    if item["shap_value"] > 0
                ]

                negative_features = [
                    item for item in explanations
                    if item["shap_value"] < 0
                ]

                if positive_features:
                    st.write(
                        f"**What supports this prediction?** "
                        f"The values you entered for "
                        f"{', '.join(feature_names.get(item['feature'], item['feature']) for item in positive_features[:3])} "
                        f"helped the AI choose {predicted_crop.title()}."
                    )

                if negative_features:
                    st.write(
                        f"**What makes the prediction less suitable?** "
                        f"The values for "
                        f"{', '.join(feature_names.get(item['feature'], item['feature']) for item in negative_features[:3])} "
                        f"pushed the AI away from its prediction."
                    )

                if not positive_features and not negative_features:
                    st.write(
                        "The AI did not identify a strong contribution "
                        "from these inputs in this explanation."
                    )

                st.info(
                    "This explanation describes how the AI made its choice. "
                    "It does not prove that the crop will grow successfully. "
                    "Check local soil conditions, water availability, season, "
                    "and advice from an agricultural expert before planting."
                )

            
            except Exception as error:
                st.error("Explanation error:")
                st.exception(error)


            # PDF soil analysis report
            report_inputs = {
                "Nitrogen (N)": nitrogen,
                "Phosphorus (P)": phosphorus,
                "Potassium (K)": potassium,
                "Soil pH": ph,
                "Temperature (°C)": temperature,
                "Humidity (%)": humidity,
                "Rainfall (mm)": rainfall,
            }

            pdf_bytes = generate_soil_report(
                input_values=report_inputs,
                predicted_crop=predicted_crop,
                top_candidates=candidates,
            )

            st.download_button(
                label="📄 Download Soil Analysis Report (PDF)",
                data=pdf_bytes,
                file_name="Kissaan_AI_Soil_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ---------------- SOIL HEALTH ----------------
elif page == "Soil Health":
    st.subheader("🧪 Soil Health Assessment")
    tab1, tab2 = st.tabs(["pH Assessment", "NPK Measurements"])

    with tab1:
        st.write("Adjust the slider to the pH value from your soil test.")
        soil_ph = st.slider(
            "Soil pH value",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            step=0.1,
            key="soil_ph",
        )

        if st.button("Assess Soil pH"):
            result = call_api("assess-soil-ph", {"ph": soil_ph})
            if result:
                st.metric("Measured pH", result.get("ph", soil_ph))
                st.subheader(
                    result.get("assessment", "Assessment complete")
                )
                st.write(result.get("interpretation", ""))
                if result.get("notice"):
                    st.info(result["notice"])

    with tab2:
        st.write(
            "Adjust the measurements from a soil test. Units are not defined "
            "by the crop recommendation dataset."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            n = st.slider("Nitrogen (N)", 0.0, 140.0, 90.0, 1.0)
        with col2:
            p = st.slider("Phosphorus (P)", 5.0, 145.0, 42.0, 1.0)
        with col3:
            k = st.slider("Potassium (K)", 5.0, 205.0, 43.0, 1.0)

        if st.button("Assess NPK Measurements"):
            result = call_api("assess-soil-npk", {"n": n, "p": p, "k": k})
            if result:
                x, y, z = st.columns(3)
                x.metric("Nitrogen (N)", result.get("nitrogen", n))
                y.metric("Phosphorus (P)", result.get("phosphorus", p))
                z.metric("Potassium (K)", result.get("potassium", k))

                if result.get("notice"):
                    st.info(result["notice"])

                st.caption(
                    "Recording measurements does not classify nutrients "
                    "as deficient or sufficient."
                )


# ---------------- FERTILIZER GUIDANCE ----------------
elif page == "Fertilizer Guidance":
    st.subheader("🌿 Fertilizer Guidance")
    st.write(
        "Submit a crop name and nutrient measurements to check whether "
        "the system has enough reference data for a recommendation."
    )

    with st.form("fertilizer_form"):
        crop = st.text_input("Crop name", value="rice")
        col1, col2, col3 = st.columns(3)

        with col1:
            fn = st.slider("Nitrogen (N)", 0.0, 140.0, 90.0, 1.0)
        with col2:
            fp = st.slider("Phosphorus (P)", 5.0, 145.0, 42.0, 1.0)
        with col3:
            fk = st.slider("Potassium (K)", 5.0, 205.0, 43.0, 1.0)

        submitted = st.form_submit_button("Check Fertilizer Guidance")

    if submitted:
        result = call_api(
            "recommend-fertilizer",
            {
                "crop": crop,
                "nitrogen": fn,
                "phosphorus": fp,
                "potassium": fk,
            },
        )

        if result:
            st.write("Crop:", result.get("crop", crop))
            st.write("Status:", result.get("status", ""))
            recommendation = result.get("fertilizer_recommendation")

            if recommendation is None:
                st.warning(
                    "No fertilizer dosage is available yet. Reliable soil-test "
                    "units, crop-specific calibration, and regional reference "
                    "data are required."
                )
            else:
                st.write(recommendation)

            if result.get("notice"):
                st.info(result["notice"])
                

# ---------------- MODEL PERFORMANCE ----------------
elif page == "Model Performance":
    st.subheader("📊 Model Performance")
    st.write(
        "Evaluate how the Random Forest model performs on the "
        "held-out test dataset."
    )

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    evaluation_dir = project_root / "docs" / "model_evaluation"

    matrix_path = evaluation_dir / "confusion_matrix.png"
    report_path = evaluation_dir / "classification_report.txt"

    if matrix_path.exists():
        st.markdown("### Confusion Matrix")
        st.image(
            str(matrix_path),
            caption="Random Forest crop classification results",
            use_container_width=True,
        )
    else:
        st.warning(
            "Confusion matrix not found. Run the model training script first."
        )

    if report_path.exists():
        st.markdown("### Classification Report")
        report_text = report_path.read_text(encoding="utf-8")
        st.code(report_text, language="text")
        st.download_button(
            "Download Classification Report",
            data=report_text,
            file_name="classification_report.txt",
            mime="text/plain",
        )
    else:
        st.warning(
            "Classification report not found. Run the model training script first."
        )



# ---------------- ABOUT ----------------
elif page == "About Project":
    st.subheader("About Kissaan AI")

    st.markdown("""
    **Project:** AI-Driven Soil Health Prediction & Precision Fertilizer
    Optimization System

    **Current components**
    - Random Forest crop recommendation model
    - Soil pH classification
    - NPK measurement validation
    - Fertilizer recommendation safety checks
    - FastAPI backend and Streamlit interface

    **Dataset**
    - 2,200 records
    - 22 crop classes
    - Soil and weather input features

    **Current limitations**
    - Crop suitability predictions are based on the training dataset.
    - NPK measurement units are not established by that dataset.
    - Fertilizer dosage recommendations require validated crop-specific
      and regional reference data.
    """)
