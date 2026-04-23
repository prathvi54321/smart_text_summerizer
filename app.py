import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 1. Page Config
st.set_page_config(page_title="My Custom AI Summarizer", page_icon="📝")

# 2. Load YOUR Manually Trained Model
@st.cache_resource
def load_custom_ai():
    # This points to the folder you just organized
    model_path = "./my_trained_model" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

# 3. PDF Extraction Function
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# 4. User Interface
st.title("📄 Smart AI Text Summarizer")
st.subheader("Fine-tuned on News Summary Dataset")

tab1, tab2 = st.tabs(["📁 Upload PDF", "✍️ Paste Text"])

input_data = ""

with tab1:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        input_data = read_pdf(uploaded_file)

with tab2:
    manual_text = st.text_area("Paste your long text here:", height=200)
    if manual_text:
        input_data = manual_text

if input_data:
    if st.button("✨ Generate Summary"):
        with st.spinner("Processing with my trained model..."):
            tokenizer, model = load_custom_ai()
            
            # T5 models expect the prefix "summarize: "
            input_text = "summarize: " + input_data
            
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            
            summary_ids = model.generate(
                inputs["input_ids"], 
                max_length=250, 
                min_length=40, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
            
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            st.markdown("---")
            st.success("### Summary Results")
            st.write(summary)
            st.download_button("📥 Download Summary", summary, file_name="summary.txt")