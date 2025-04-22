import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load

# ====== Load Model & Tokenizer ======
model = tf.keras.models.load_model('sentiment_model.keras')
tokenizer = load('tokenizer.joblib')
max_length = 100

# ====== Set Page Config ======
st.set_page_config(page_title="Sentiment AI", layout="centered")

# ====== Custom Dark Mode & Futuristic Style ======
st.markdown("""
    <style>
    body, .main, .block-container {
        background-color: #0f1117;
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #1e2130;
        color: #ffffff;
        font-family: 'Courier New', monospace;
    }
    .stButton>button {
        background-color: #00b4d8;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 12px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 0 10px #00b4d8;
    }
    .stButton>button:hover {
        background-color: #0077b6;
        box-shadow: 0 0 15px #0077b6;
    }
    .sentiment-result {
        font-size: 2em;
        font-weight: bold;
        padding: 20px;
        margin-top: 20px;
        border-radius: 12px;
        text-align: center;
        background: linear-gradient(135deg, #1e2130, #292d3e);
        box-shadow: 0 0 20px rgba(0, 180, 216, 0.3);
        animation: popIn 0.6s ease;
    }
    @keyframes popIn {
        0% { transform: scale(0.8); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# ====== App Title ======
st.markdown("<h1 style='text-align: center;'>🧠 Twitter Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #888;'>Understand emotions behind your text</h4>", unsafe_allow_html=True)

# ====== Input Area ======
user_input = st.text_area("✍️ Enter your text here", height=150)

# ====== Sentiment Prediction ======
def predict_sentiment(text):
    sequences = tokenizer.texts_to_sequences([text])
    if not sequences or not sequences[0]:
        st.error("🚫 No recognizable words found in the input.")
        return None
    padded = pad_sequences(sequences, maxlen=max_length, truncating='post')
    prediction = model.predict(padded)
    return prediction

# ====== Prediction Button ======
if st.button("🔍 Analyze Sentiment"):
    if user_input:
        prediction = predict_sentiment(user_input)
        if prediction is not None:
            sentiment = ["😠 Negative", "😐 Neutral", "😊 Positive"][prediction.argmax()]
            st.markdown(f'<div class="sentiment-result">Sentiment: {sentiment}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text to analyze.")
