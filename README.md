# GPT-PREP: PDF Processing for GPT Context Retrieval

GPT-PREP is a Python script that processes a directory of PDF files, extracts and chunks the text, builds a FAISS index for semantic search, and optionally allows you to query the index. The retrieved context can then be manually pasted into ChatGPT for further analysis. The script also automatically detects GPU availability to accelerate embedding generation and similarity searches.

## Features

- **PDF Text Extraction:** Uses PyMuPDF to extract text from PDFs.
- **Text Chunking:** Splits large texts into manageable chunks with LangChain's text splitter.
- **Embedding Generation:** Uses SentenceTransformer to generate embeddings, with GPU support if available.
- **Semantic Search:** Builds a FAISS index (with GPU support if available) for fast similarity search.
- **Progress Logging:** Outputs progress information (e.g. file currently being processed and files left) to a log file.
- **Command-line Interface:** Use flags such as `--no-query` to bypass interactive querying and `-h/--help` for usage instructions.
- **Output File:** Query results are written to an output file.

## Prerequisites

- Python 3.7 or later
- The following Python packages:
  - `PyMuPDF`
  - `langchain`
  - `sentence-transformers`
  - `faiss-cpu` (or GPU variant if using GPU)
  - `torch`
  - `numpy`
  - `pickle`
  - `argparse`
  - `logging`

## Installation

1. **Clone or download the repository.**
2. **Install the required dependencies.** For example, using pip:

   ```bash
   pip install PyMuPDF langchain sentence-transformers faiss-cpu torch numpy
   ```

   If you are planning to use a GPU with FAISS, you might need to install the GPU version of FAISS as per [FAISS documentation](https://github.com/facebookresearch/faiss).

## Usage

Run the script from the command line by specifying the input directory that contains your PDF files. You can also specify an output file for the query results.

### Basic Usage

```bash
python gpt-prep.py -i ~/path/to/pdf_folder
```

This command will process all PDFs in the specified folder, build the FAISS index, and then enter an interactive query loop.

### Skipping the Query Loop

If you want to process and build the index without entering the query loop, use the `--no-query` flag:

```bash
python gpt-prep.py -i ~/path/to/pdf_folder --no-query
```

### Help

To view detailed usage instructions, run:

```bash
python gpt-prep.py -h
```

## Command-line Options

- `-i`, `--input_dir`: **(Required)** Path to the input directory containing PDF files.
- `-o`, `--output_file`: Output file to save query results. Default is `query_results.txt`.
- `--no-query`: If set, the script will process the PDFs and build the index without entering the interactive query loop.
- `-h`, `--help`: Displays help and usage instructions.

## How It Works

1. **PDF Extraction:** The script iterates over all PDF files in the specified directory and extracts text using PyMuPDF.
2. **Text Chunking:** The combined text is split into manageable chunks using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embedding Generation & FAISS Indexing:** Each text chunk is converted into an embedding using a SentenceTransformer model. The embeddings are then added to a FAISS index for efficient semantic search.
4. **GPU Auto-detection:** The script automatically detects if a GPU is available and, if so, moves both the embedding model and the FAISS index to the GPU.
5. **Querying:** Optionally, you can enter queries to retrieve the most relevant text chunks, which are written to the output file.

## GPU Support

GPT-PREP automatically checks for GPU availability using PyTorch’s `torch.cuda.is_available()`. If a GPU is detected, the SentenceTransformer model is moved to the GPU, and the FAISS index is transferred to GPU memory using `faiss.index_cpu_to_gpu()`. If not, the script falls back to CPU processing.

## Licence

This project is open source and available under the MIT License.

---

Enjoy using GPT-PREP for efficient processing and retrieval of context from your research PDFs!
