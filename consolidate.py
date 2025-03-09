import os
import argparse
import logging
import re
import fitz  # PyMuPDF for text extraction
from pdf2image import convert_from_path
import pytesseract
from clean_filters import clean_text # Not in SCR repo as it's specific


# -------------------------------
# Logging Setup
# -------------------------------
def setup_logging(log_file="consolidate.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# -------------------------------
# Load Garbage Patterns from File
# -------------------------------
def load_garbage_patterns(cleaning_file):
    """Load additional garbage patterns from an input file (one pattern per line)."""
    patterns = []
    if cleaning_file and os.path.exists(cleaning_file):
        with open(cleaning_file, "r", encoding="utf-8") as f:
            for line in f:
                pattern = line.strip()
                if pattern:
                    patterns.append(pattern.lower())
    return patterns

# -------------------------------
# Extract Text from a PDF (with OCR fallback)
# -------------------------------
def extract_text_from_pdf(pdf_path, ocr_threshold=100, extra_patterns=None):
    """
    Extract text from a PDF using PyMuPDF. If the extracted text is very short,
    assume it's a scanned document and use OCR.
    """
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in doc)
        text = clean_text(text, extra_patterns)
        if len(text.strip()) < ocr_threshold:
            logging.info(f"Low text detected in {pdf_path}, falling back to OCR.")
            text = extract_text_from_scanned_pdf(pdf_path)
            text = clean_text(text, extra_patterns)
        return text
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
        return ""

# -------------------------------
# Extract Text using OCR
# -------------------------------
def extract_text_from_scanned_pdf(pdf_path):
    """
    Convert PDF pages to images and use Tesseract OCR to extract text.
    Requires Tesseract to be installed and accessible in your PATH.
    """
    try:
        images = convert_from_path(pdf_path)
        text = "\n".join(pytesseract.image_to_string(image) for image in images)
        return text
    except Exception as e:
        logging.error(f"OCR failed for {pdf_path}: {e}")
        return ""

# -------------------------------
# Process Directory of PDFs Recursively
# -------------------------------
def process_directory(input_dir, extra_patterns=None):
    """
    Traverse all subdirectories in the input directory and extract text from all PDF files.
    """
    all_texts = []
    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    total_files = len(pdf_files)
    if total_files == 0:
        logging.warning("No PDF files found in the specified directory or its subdirectories.")
        return ""
    
    for idx, pdf_file in enumerate(pdf_files, start=1):
        logging.info(f"Processing file {idx} of {total_files}: {pdf_file}")
        text = extract_text_from_pdf(pdf_file, extra_patterns=extra_patterns)
        if text:
            all_texts.append(text)
        logging.info(f"Finished processing {pdf_file}. Files left: {total_files - idx}")
    
    combined_text = "\n\n".join(all_texts)
    return combined_text

# -------------------------------
# Main Function
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Consolidate text from multiple PDFs (including subdirectories) into one file with OCR fallback and text cleaning.",
        epilog="Example usage: python consolidate.py -i ~/folder -o consolidated_output.txt --cleaning_file garbage.txt"
    )
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing PDF files (searched recursively).")
    parser.add_argument("-o", "--output_file", default="consolidated_output.txt", help="File to save the consolidated text.")
    parser.add_argument("--cleaning_file", help="Optional file containing garbage text patterns to remove, one per line.")
    args = parser.parse_args()
    
    setup_logging()
    extra_patterns = load_garbage_patterns(args.cleaning_file)
    
    logging.info("Starting PDF extraction from directory and subdirectories...")
    combined_text = process_directory(args.input_dir, extra_patterns=extra_patterns)
    if not combined_text:
        logging.error("No text extracted. Exiting.")
        return

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(combined_text)
    logging.info(f"Consolidated text saved to {args.output_file}")

if __name__ == "__main__":
    main()
