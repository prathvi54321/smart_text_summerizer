import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 1. Page Config
st.set_page_config(page_title="Smart AI Summarizer", layout="wide")

# 2. Load AI Model
@st.cache_resource
def load_ai_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# 3. PDF Extraction
def read_pdf(file):
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# 4. User Interface
st.title("📄 Smart AI Text Summarizer")

# Tabs for Input
tab1, tab2 = st.tabs(["📁 Upload PDF", "✍️ Paste Text"])

input_data = ""

with tab1:
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
    if uploaded_file:
        input_data = read_pdf(uploaded_file)

with tab2:
    manual_text = st.text_area("Paste your text here:", height=200)
    if manual_text:
        input_data = manual_text

# --- THE FIX: SHOW PREVIEW ---
if input_data:
    st.subheader("Extracted Text Preview:")
    # This box shows you exactly what the AI sees
    with st.expander("Click to view the text found in your document"):
        st.write(input_data[:1000] + "...") 
    
    if st.button("✨ Generate Summary"):
        # Check if text is actually readable
        if len(input_data.strip()) < 100:
            st.error("❌ The text extracted is too short (less than 100 characters). Is this a scanned image or an empty PDF?")
        else:
            with st.spinner("AI is analyzing..."):
                tokenizer, model = load_ai_model()
                # Process the text
                inputs = tokenizer(input_data[:3000], return_tensors="pt", truncation=True, max_length=1024)
                summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                st.markdown("---")
                st.subheader("📝 Summary:")
                st.success(summary)
                st.download_button("Download Summary", summary, file_name="summary.txt")
else:
    st.info("Please upload a PDF or paste text to start.")