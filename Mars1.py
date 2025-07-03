import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
from keybert import KeyBERT
from transformers import pipeline
import json
import io
import requests
from collections import Counter

# Load models
@st.cache_resource
def load_models():
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    import en_core_web_sm
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import os
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return kw_model, nlp, summarizer

kw_model, nlp, summarizer = load_models()

# Helper functions
def get_title_from_text(text):
    for line in text.strip().split("\n"):
        clean_line = line.strip()
        if len(clean_line) > 10:
            return clean_line
    return "No title found."

def get_author_from_text(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[1:6]:
        if ("by" in line.lower() and len(line.split()) < 8):
            return line
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) <= 3:
            return ent.text
    return "Not Found"

def extract_named_entities(text, top_n=5):
    doc = nlp(text)
    ents = [ent.text.strip() for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}]
    freq = Counter(ents)
    return freq.most_common(top_n)

def metadata(text, file_type, file_size_kb):
    title = get_title_from_text(text)
    author = get_author_from_text(text)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=10)
    entities = extract_named_entities(text)
    short_text = text[:1024]
    summary = summarizer(short_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"] if len(short_text.split()) > 10 else "Text too short for summarization."

    return {
        "title": title,
        "author": author,
        "keywords": [kw for kw, _ in keywords],
        "named_entities": [ent for ent, _ in entities],
        "summary": summary,
        "file_format": file_type,
        "file_size_kb": file_size_kb,
        "word_count": len(text.split())
    }

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text_from_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text.strip()

def apply_cloud_ocr_on_pdf(file):
                images = convert_from_bytes(file.getvalue())
                ocr_text = ""
                for img in images:
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    buf.seek(0)
                    response = requests.post(
                         'https://api.ocr.space/parse/image',
                          files={'filename': buf},
                          data={'apikey': 'K81122752688957', 'language': 'eng'}
                      )
                    result = response.json()
                    if result.get("IsErroredOnProcessing"):
                       raise ValueError(result.get("ErrorMessage", "OCR failed"))
                    ocr_text += result['ParsedResults'][0]['ParsedText'] + "\n"
                return ocr_text

def extract_text_from_image_cloud(file):
    response = requests.post(
        'https://api.ocr.space/parse/image',
        files={'filename': file},
        data={'apikey': 'K81122752688957', 'language': 'eng'}
    )
    result = response.json()
    if result.get("IsErroredOnProcessing"):
        raise ValueError(result.get("ErrorMessage", "OCR failed"))
    return result['ParsedResults'][0]['ParsedText']


# UI Layout
st.set_page_config(page_title="📄 Metadata Generator", layout="wide")
st.title("📄 Smart Metadata Generator")
st.markdown("Upload a document and automatically generate structured metadata including title, author, summary, keywords, and more.")

with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    file_size_kb = round(len(uploaded_file.getvalue()) / 1024, 2)

    # Text extraction
    if file_type == "docx":
        text = extract_text_from_docx(uploaded_file)
    elif file_type == "txt":
        text = extract_text_from_txt(uploaded_file)
    elif file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
        if not text or len(text.strip()) < 50:
            st.warning("PDF may be image-based. Applying OCR...")
            text = apply_cloud_ocr_on_pdf(uploaded_file)

            
    elif file_type in ["png", "jpg", "jpeg"]:
        text = extract_text_from_image_cloud(uploaded_file)
    else:
        st.error("Unsupported file type.")
        text = ""

    if text:
        with st.spinner("🔍 Extracting metadata..."):
            metadata_dict = metadata(text, file_type, file_size_kb)

        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Word Count", metadata_dict["word_count"])
        col2.metric("📁 File Size", f"{metadata_dict['file_size_kb']} KB")
        col3.metric("🧾 Format", metadata_dict['file_format'].upper())

        st.markdown("---")
        st.markdown("### 🧠 Extracted Metadata")
        st.json(metadata_dict)

        st.markdown("### 🔑 Keywords")
        st.markdown(" ".join([f"`{kw}`" for kw in metadata_dict["keywords"]]))

        st.markdown("### 🧠 Named Entities")
        st.markdown(" ".join([f"🟢 `{ent}`" for ent in metadata_dict["named_entities"]]))

        st.download_button("📥 Download Metadata as JSON", data=json.dumps(metadata_dict, indent=2, ensure_ascii=False), file_name="metadata_output.json", mime="application/json")

        with st.expander("📃 View Full Extracted Text"):
            st.text_area("Document Text", text, height=300)
    else:
        st.error("No text could be extracted.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit · Yash Arora · 2025")
