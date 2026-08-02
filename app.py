import streamlit as st
from transformers import pipeline

# Page Configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# Title
st.title("📝 AI Text Summarizer")
st.write("Summarize long articles, notes, and documents using Artificial Intelligence.")

# Input
text = st.text_area(
    "Enter your text here:",
    height=250,
    placeholder="Paste your content..."
)

# Summary length
length = st.selectbox(
    "Choose summary length:",
    ["Short", "Medium", "Long"]
)

if st.button("✨ Summarize"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):
            if length == "Short":
                result = summarizer(
                    text,
                    max_length=60,
                    min_length=20,
                    do_sample=False
                )
            elif length == "Medium":
                result = summarizer(
                    text,
                    max_length=100,
                    min_length=40,
                    do_sample=False
                )
            else:
                result = summarizer(
                    text,
                    max_length=150,
                    min_length=60,
                    do_sample=False
                )

        st.success("Summary Generated!")
        st.write(result[0]["summary_text"])