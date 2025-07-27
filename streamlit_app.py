import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from fpdf import FPDF
import os
import datetime
import tempfile
import gdown
# ------------------------- App Config -------------------------
st.set_page_config(page_title="Mammogram Cancer Classifier", layout="centered", initial_sidebar_state="expanded")

# ------------------------- Visitor Count -------------------------
if "visitor_count" not in st.session_state:
    st.session_state.visitor_count = 1
else:
    st.session_state.visitor_count += 1

# ------------------------- Load Model -------------------------
# Path to save downloaded model
MODEL_PATH = "mammogram_cancer_model.h5"

# Google Drive direct download link
GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1z_wDk2YPzvOz2DFt5WqoVvrq4Ia82edu"

# Download only if not already downloaded
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Google Drive..."):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)


# ------------------------- Prediction Logic -------------------------
def preprocess_and_predict(image: Image.Image):
    image = image.convert("L")
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 256, 256, 1)
    prediction = model.predict(img_array)[0][0]
    label = "🔴 Malignant (Cancerous)" if prediction > 0.5 else "🟢 Benign (Non-cancerous)"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, f"{confidence:.2%}"

# ------------------------- PDF Report -------------------------
def generate_pdf(label, confidence):
    label_text = "Malignant (Cancerous)" if "Malignant" in label else "Benign (Non-cancerous)"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Mammogram Cancer Prediction Report", ln=1, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction Result: {label_text}", ln=1)
    pdf.cell(200, 10, txt=f"Confidence Score: {confidence}", ln=2)
    pdf.cell(200, 10, txt=f"Date Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=3)

    temp_dir = tempfile.gettempdir()
    report_path = os.path.join(temp_dir, "prediction_report.pdf")
    pdf.output(report_path)
    return report_path

# ------------------------- Sidebar Navigation -------------------------
with st.sidebar:
    selected = option_menu("Mammogram Classifier", 
                           ["Welcome", "Prediction", "Recommendations", "FAQs", "Disclaimers", "Analytics"], 
                           icons=['house', 'search', 'clipboard-check', 'question-circle', 'exclamation', 'bar-chart'],
                           menu_icon="cast", default_index=0, orientation="vertical")

# ------------------------- Pages -------------------------
if selected == "Welcome":
    st.markdown("<h1 style='color:green;'>👋 Welcome to the Mammogram Cancer Classifier</h1>", unsafe_allow_html=True)

    st.markdown("""
    The **Mammogram Cancer Classifier** is an advanced decision-support tool powered by **Convolutional Neural Networks (CNNs)** to assist in the preliminary analysis of grayscale mammographic images for cancer risk assessment.

    ---
    ### 🔍 Key Features:
    - ⚡ **Fast and Accurate Predictions** using deep learning.
    - 📈 **Confidence Scores** to support interpretability of results.
    - 📋 **Personalized Output Summary** with clinical notes.
    - 📄 **Automated PDF Report Generation** for documentation and sharing.
    - ❓ **Interactive FAQs and Analytics Dashboard** to support continuous learning and departmental insights.
    ---

    > 🧠 *This tool is intended to augment radiological decision-making — not replace it. Final diagnosis must be made by a qualified healthcare professional.*
    """)

    st.success("✅ You may proceed by uploading a grayscale mammogram image for analysis.")

elif selected == "Prediction":
    st.title("🔍 Upload and Predict")
    uploaded_file = st.file_uploader("Upload a mammogram image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Mammogram", use_column_width=True)

        if st.button("🔎 Predict"):
            with st.spinner("Analyzing..."):
                label, confidence = preprocess_and_predict(image)
                st.session_state.label = label  # Save label to session
                st.session_state.confidence = confidence
                st.success(f"Prediction: {label}")
                st.info(f"Confidence: {confidence}")

                pdf_path = generate_pdf(label, confidence)
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download PDF Report", f, file_name="prediction_report.pdf")

elif selected == "Recommendations":
    st.title("🩺 Recommendations Based on Prediction")
    label = st.session_state.get("label", None)
    if not label:
        st.warning("Please make a prediction first.")
    else:
        if "Malignant" in label:
            st.error("⚠ Urgent: The mass appears malignant.")
            st.markdown("""
            - Consult a specialist (oncologist) immediately
            - Schedule biopsy and advanced imaging
            - Begin treatment planning based on staging
            - Maintain emotional and mental health support
            """)
        else:
            st.success("✅ The mass appears benign.")
            st.markdown("""
            - Maintain regular screening schedule
            - Re-scan every 6–12 months as advised
            - Practice a healthy lifestyle (diet, exercise, alcohol moderation)
            - Report any new symptoms to your physician
            """)

elif selected == "FAQs":
    st.title("❓ Frequently Asked Questions")
    with st.expander("What is this tool used for?"):
        st.write("This tool is for predicting breast cancer likelihood using a trained CNN model from grayscale mammogram images.")
    with st.expander("Is this a replacement for a doctor?"):
        st.write("No. Always consult a certified medical professional for diagnosis and treatment.")
    with st.expander("What types of images are supported?"):
        st.write("Grayscale mammogram images in .jpg, .jpeg, or .png format.")
    with st.expander("Can I trust the prediction?"):
        st.write("The model is trained on real mammogram datasets and validated, but is not 100% accurate.")
    with st.expander("Is my image stored?"):
        st.write("No. Your uploaded image is never stored or transmitted outside this session.")
    with st.expander("How is confidence calculated?"):
        st.write("It is the model's probability for the predicted class, scaled between 0% and 100%.")
    with st.expander("Who can use this tool?"):
        st.write("Researchers, students, clinicians, and developers exploring ML in healthcare.")

elif selected == "Disclaimers":
    st.title("⚠️ Medical Disclaimer")
    
    st.markdown("""
    This application is developed to support **educational, research, and departmental use** within the Radiology Department. Please carefully read the disclaimers below before relying on any output:

    - 🧪 **Not a Certified Diagnostic Tool**: This system is not approved as a standalone diagnostic device and should **not** replace radiological assessments or expert interpretation.
    
    - 👨‍⚕️ **Clinical Judgment Required**: Final medical decisions must always be made by licensed radiologists or authorized healthcare professionals. The tool is meant to assist, not substitute, professional evaluation.
    
    - 🛠 **For Departmental Use Only**: The system is designed for internal use within the hospital environment and may not generalize to other clinical settings.
    
    - 🚫 **No Liability for Medical Outcomes**: The development team and associated personnel are **not liable** for any clinical outcomes or interpretations based on this tool's results.

    - 🔄 **Ongoing Development**: This tool is under continuous development and refinement. Updates may affect performance or outputs.
    """)

    st.info("📌 In all cases of clinical uncertainty or emergency, consult a senior radiologist or physician immediately.")


elif selected == "Analytics":
    st.title("📊 Visitor Analytics Overview")
    
    st.markdown("### 👥 Total Visitors in This Session")
    st.metric(
        label="Total Session Visitors",
        value=st.session_state.visitor_count,
        delta=None
    )
    
    st.info("ℹ️ This count reflects the number of visitors in the **current session only**. "
            "It will reset when the browser or app is refreshed or closed.")
    
