import os
import glob
import argparse
import logging
import torch
import fitz  # PyMuPDF for text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# -------------------------------
# Logging Setup
# -------------------------------
def setup_logging(log_file="gpt_prep.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Remove if you only want file output
        ]
    )

# -------------------------------
# Step 1: Extract Text from a PDF
# -------------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in doc)
        return text
    except Exception as e:
        logging.error(f"Error processing {pdf_path}: {e}")
        return ""

# -------------------------------
# Step 2: Iterate over PDFs in a Directory
# -------------------------------
def process_directory(input_dir):
    """Extract text from all PDF files in the provided directory."""
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    total_files = len(pdf_files)
    all_texts = []
    
    if total_files == 0:
        logging.warning("No PDF files found in the specified directory.")
        return ""
    
    for idx, pdf_file in enumerate(pdf_files, start=1):
        logging.info(f"Processing file {idx} of {total_files}: {pdf_file}")
        text = extract_text_from_pdf(pdf_file)
        if text:
            all_texts.append(text)
        logging.info(f"Finished processing {pdf_file}. Files left: {total_files - idx}")
    # Combine texts from all PDFs into one large string
    combined_text = "\n\n".join(all_texts)
    return combined_text

# -------------------------------
# Step 3: Chunk the Text
# -------------------------------
def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """Split text into manageable chunks using LangChain's splitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

# -------------------------------
# Step 4: Create Embeddings and Build FAISS Index with GPU Auto-detection
# -------------------------------
def build_index(chunks, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for chunks and build a FAISS index with GPU support if available."""
    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Load the embedding model and move to appropriate device
    model = SentenceTransformer(model_name)
    model = model.to(device)
    
    # Generate embeddings (they are generated on the device specified by the model)
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings)
    
    dimension = embeddings.shape[1]
    index_cpu = faiss.IndexFlatL2(dimension)
    
    if device == "cuda":
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            logging.info("FAISS index moved to GPU.")
        except Exception as e:
            logging.warning(f"Failed to use GPU for FAISS index due to: {e}. Falling back to CPU.")
            index = index_cpu
    else:
        index = index_cpu

    index.add(embeddings)
    return index, model, embeddings

# -------------------------------
# Step 5: Save the Index and Chunks
# -------------------------------
def save_pipeline(index, chunks, index_filename="research_index.index", chunks_filename="chunks.pkl"):
    """Save the FAISS index and text chunks to disk."""
    # If the index is on GPU, move it back to CPU before saving
    if faiss.get_num_gpus() > 0 and hasattr(index, 'index'):
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, index_filename)
    with open(chunks_filename, "wb") as f:
        pickle.dump(chunks, f)
    logging.info("Index and chunks have been saved to disk.")

# -------------------------------
# Step 6: Load the Index and Chunks
# -------------------------------
def load_pipeline(index_filename="research_index.index", chunks_filename="chunks.pkl", model_name="all-MiniLM-L6-v2"):
    """Load the FAISS index and text chunks from disk."""
    if not os.path.exists(index_filename) or not os.path.exists(chunks_filename):
        raise FileNotFoundError("Index or chunks file not found. Run the pipeline build first.")
    index = faiss.read_index(index_filename)
    with open(chunks_filename, "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer(model_name)
    return index, chunks, model

# -------------------------------
# Step 7: Query the Index
# -------------------------------
def query_pipeline(query, index, chunks, model, top_k=3):
    """Given a query, return the top-k matching text chunks."""
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    context = "\n\n".join([chunks[i] for i in indices[0]])
    return context

# -------------------------------
# Main Function to Run the Pipeline
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Process PDFs for GPT context retrieval and build a FAISS index for semantic search.",
        epilog="Example usage: python gpt-prep.py -i ~/folder -o results.txt --no-query"
    )
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing PDF files.")
    parser.add_argument("-o", "--output_file", default="query_results.txt", help="File to save query results.")
    parser.add_argument("--no-query", action="store_true", help="Process PDFs and build the index without entering the query loop.")
    args = parser.parse_args()

    setup_logging()  # Initialise logging

    # Process the PDFs in the input directory
    logging.info("Starting PDF extraction from directory...")
    combined_text = process_directory(args.input_dir)
    if not combined_text:
        logging.error("No text extracted. Exiting.")
        return
    
    logging.info("Chunking combined text...")
    chunks = chunk_text(combined_text)
    logging.info(f"Created {len(chunks)} chunks.")

    logging.info("Building FAISS index...")
    index, model, _ = build_index(chunks)
    
    logging.info("Saving pipeline data...")
    save_pipeline(index, chunks)

    if not args.no_query:
        # Query loop: Enter a query and get the context, then write results to a file
        logging.info("Ready to query your documents!")
        with open(args.output_file, "w", encoding="utf-8") as f_out:
            while True:
                query = input("\nEnter your query (or 'exit' to quit): ")
                if query.lower() == 'exit':
                    break
                context = query_pipeline(query, index, chunks, model)
                output = f"\n--- Query: {query} ---\n--- Retrieved Context ---\n{context}\n"
                f_out.write(output)
                f_out.flush()  # Write progress to file immediately
                logging.info(f"Query processed. Results written to {args.output_file}")
    else:
        logging.info("Query loop skipped as per '--no-query' flag.")

if __name__ == "__main__":
    main()
