import streamlit as st
import joblib
from model import text_pre_processing
# Load the trained model pipeline
model = joblib.load('svm_text_pipeline.pkl')

st.set_page_config(page_title="Spam or Ham Classifier", layout="centered")
st.title("📩 Spam or Ham Classifier")

st.write("Enter a message below to check if it's **Spam** or **Ham**:")

# Text input box
user_input = st.text_area("Your Message:", height=150)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message before checking.")
    else:
        # Predict
        prediction = model.predict([user_input])[0]

        # Display result
        if prediction.lower() == 'spam':
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **HAM** .")
