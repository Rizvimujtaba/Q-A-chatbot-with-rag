from dotenv import load_dotenv
load_dotenv()

import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "pdfs"
PERSIST_DIR = "./chroma_gemini_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Free-tier Gemini embedding limit is 100 requests/minute.
# Keep batches well under that, and sleep between them.
BATCH_SIZE = 50
SLEEP_SECONDS = 65


def main():
    pdf_files = sorted(Path(DOCS_DIR).glob("*.pdf"))
    docx_files = sorted(Path(DOCS_DIR).glob("*.docx"))
    all_files = pdf_files + docx_files

    if not all_files:
        raise SystemExit(f"No PDFs or DOCX files found in {DOCS_DIR}/")

    all_docs = []
    for f in all_files:
        if f.suffix.lower() == ".pdf":
            docs = PyPDFLoader(str(f)).load()
            print(f"  [PDF]  {f.name}: {len(docs)} pages")
        else:
            docs = Docx2txtLoader(str(f)).load()
            print(f"  [DOCX] {f.name}: {len(docs)} document(s)")
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} document(s) from {len(all_files)} file(s).")

    splits = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    ).split_documents(all_docs)
    print(f"Split into {len(splits)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    n_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Embedding in {n_batches} batch(es) of up to {BATCH_SIZE}...")

    vectorstore = None
    for i in range(0, len(splits), BATCH_SIZE):
        batch = splits[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{n_batches}: {len(batch)} chunks...")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=PERSIST_DIR,
            )
        else:
            vectorstore.add_documents(batch)

        # Sleep between batches to respect the RPM limit
        if i + BATCH_SIZE < len(splits):
            print(f"  Sleeping {SLEEP_SECONDS}s to reset quota window...")
            time.sleep(SLEEP_SECONDS)

    print(f"Done. {vectorstore._collection.count()} chunks stored in {PERSIST_DIR}/")


if __name__ == "__main__":
    main()