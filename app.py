import streamlit as st
import joblib
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

# Load model and TF-IDF
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Page settings
st.set_page_config(
    page_title="AI Content Moderator",
    page_icon="🛡️"
)

# Title
st.title("🛡️ AI Content Moderator")
st.write("Check whether your comment is SAFE or TOXIC in multiple languages.")

# User input
text = st.text_area(
    "Enter your comment:",
    placeholder="Type your comment in any supported language..."
)

# Button
if st.button("🔍 Check Content"):

    if text.strip() == "":
        st.warning("Please enter a comment.")

    else:
        # Detect language
        try:
            language = detect(text)
            st.info(f"🌐 Detected Language: **{language.upper()}**")

        except LangDetectException:
            st.error("Could not detect the language.")
            st.stop()

        # Translate to English if necessary
        translated_text = text

        if language != "en":
            try:
                translated_text = GoogleTranslator(
                    source="auto",
                    target="en"
                ).translate(text)

                st.write("🔄 **Translated for analysis:**")
                st.write(translated_text)

            except Exception as e:
                st.error("Translation failed. Please try again.")
                st.stop()

        # Convert translated text to TF-IDF
        text_tfidf = tfidf.transform([translated_text])

        # Prediction
        prediction = model.predict(text_tfidf)[0]

        # Probability
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = max(probabilities) * 100

        # Result
        if prediction == 0:
            st.success("🟢 SAFE")
        else:
            st.error("🔴 TOXIC")

        st.write(f"Confidence: **{confidence:.2f}%**")