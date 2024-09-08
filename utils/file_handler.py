import logging
import PyPDF2
import docx
from PIL import Image
import pytesseract

def handle_uploaded_file(uploaded_file, file_type):
    logging.info(f"Processing file of type: {file_type}")
    
    try:
        if file_type == 'pdf':
            return extract_text_from_pdf(uploaded_file)
        elif file_type == 'docx':
            return extract_text_from_docx(uploaded_file)
        elif file_type == 'txt':
            return extract_text_from_txt(uploaded_file)
        elif file_type in ['jpg', 'png']:
            return extract_text_from_image(uploaded_file)
        else:
            logging.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_txt(txt_file):
    return txt_file.read().decode('utf-8')

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    return pytesseract.image_to_string(image)
