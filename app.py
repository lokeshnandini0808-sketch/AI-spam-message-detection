# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import joblib
import os
import string

# Page Configuration for Premium Look
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🚨",
    layout="centered"
)

# Text cleaning helper function
def basic_text_cleaning(text):
    """
    Apply the exact text cleaning used during model training.
    """
    text = str(text)
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 3. Remove extra spaces
    text = " ".join(text.split())
    return text

# Define model paths dynamically for cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

# UI Header
st.title("AI Spam Message Detector 🚨")
st.write("This app uses a Machine Learning model (Multinomial Naive Bayes) to classify text messages as Spam or Ham (Safe).")
st.write("---")

# Main Loading Phase
if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    # Auto-train the model if missing
    with st.spinner("Training model for the first time…"):
        try:
            import subprocess, sys
            train_script = os.path.join(BASE_DIR, "train.py")
            subprocess.run([sys.executable, train_script], check=True, cwd=BASE_DIR)
        except Exception as e:
            st.error(f"🚨 Failed to auto-train model: {e}")
            st.stop()
# Load components
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except Exception as e:
    st.error(f"❌ Error loading model artifacts: {e}")
    st.stop()        

# User input text area
user_input = st.text_area(
    "Enter your message below:",
    height=150,
    placeholder="Type or paste suspicious text message, email snippet, or OTP text here..."
)

col1, col2 = st.columns([1, 4])
with col1:
    predict_btn = st.button("Predict")
    
with col2:
    pass

if predict_btn:
    # 1. Input Validation
    if not user_input.strip():
        st.warning("⚠️ Please enter a message first before predicting!")
    else:
        try:
            # 2. Text Preprocessing
            cleaned_text = basic_text_cleaning(user_input)
            
            # 3. Vectorization
            vectorized_text = vectorizer.transform([cleaned_text])
            
            # 4. Predict
            prediction = model.predict(vectorized_text)[0]
            
            st.write("### Prediction Result:")
            # 5. Display output
            if prediction == 1:
                st.error("🚨 **SPAM DETECTED** \n\nThis message looks like spam. Be careful with links or call-to-actions!")
            else:
                st.success("✅ **HAM / NORMAL MESSAGE** \n\nThis message is safe. It is classified as normal.")
        except Exception as e:
            st.error(f"❌ Error loading/predicting with the model: {str(e)}")

