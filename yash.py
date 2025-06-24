import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
from keybert import KeyBERT
import spacy
from transformers import pipeline
import json
import io

# Load models
kw_model = KeyBERT(model='all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")



def get_title_from_text(text):
    lines = text.strip().split("\n")
    for line in lines:
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
    return None

def metadata(text):
    title = get_title_from_text(text)
    author = get_author_from_text(text)

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=10
    )

    doc = nlp(text)
    named_entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    short_text = text[:1024]
    if len(short_text.split()) > 10:
        summary = summarizer(short_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
    else:
        summary = "Text too short for summarization."

    return {
        "title": title,
        "author": author if author else "Not Found",
        "keywords": [kw for kw, _ in keywords],
        "named_entities": named_entities,
        "summary": summary
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

def apply_ocr_on_pdf(file):
    images = convert_from_bytes(file.getvalue())
    ocr_text = ""
    for img in images:
        text = pytesseract.image_to_string(img)
        ocr_text += text + "\n"
    return ocr_text

def extract_text_from_image(file):
    img = Image.open(io.BytesIO(file.read()))
    return pytesseract.image_to_string(img)



st.set_page_config(page_title="Document Text Extractor", layout="centered")
st.title("📄 Document Text Extractor with OCR Support")
st.write("Upload a PDF, DOCX, TXT, PNG, or JPG file. OCR will be applied if needed.")

uploaded_file = st.file_uploader("Choose your file", type=["pdf", "docx", "txt", "png", "jpg", "jpeg"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "docx":
        text = extract_text_from_docx(uploaded_file)
    elif file_type == "txt":
        text = extract_text_from_txt(uploaded_file)
    elif file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
        if not text or len(text.strip()) < 50:
            st.warning("PDF may be image-based. Applying OCR...")
            text = apply_ocr_on_pdf(uploaded_file)
    elif file_type in ["png", "jpg", "jpeg"]:
        text = extract_text_from_image(uploaded_file)
    else:
        st.error("Unsupported file type.")
        text = ""

    if text:
        progress_bar = st.progress(0)
        status_text = st.empty()

 
        status_text.text(" Extracting metadata...")
        progress_bar.progress(25)

    
        metadata_dict = metadata(text)

        progress_bar.progress(75)
        status_text.text(" Extraction complete! Rendering results...")


        st.subheader("📜 Extracted Metadata")
        st.json(metadata_dict)

        st.download_button(
            label="📥 Download Metadata as JSON",
            data=json.dumps(metadata_dict, indent=2, ensure_ascii=False),
            file_name="metadata_output.json",
            mime="application/json"
        )

        st.subheader("📄 Extracted Text")
        st.text_area("Document Text", text, height=300)
    else:
        st.error("No text could be extracted.")
