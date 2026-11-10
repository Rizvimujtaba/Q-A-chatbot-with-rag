from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_chroma import Chroma

PERSIST_DIR = "./chroma_gemini_db"
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"
TOP_K = 4


def build_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL)
    return retriever, llm


def answer(question: str, retriever, llm) -> tuple:
    hits = retriever.invoke(question)
    context = "\n\n".join(h.page_content for h in hits)

    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say 'I don't know based on the provided documents.' "
        "Cite the page number when possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)

    # Extract text from either str or list-of-blocks
    if isinstance(response.content, str):
        text = response.content
    else:
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in response.content
        )

    return text, hits


def main():
    retriever, llm = build_chain()

    print("RAG chatbot ready. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break
        if not question:
            continue

        response, hits = answer(question, retriever, llm)
        print(f"\nBot: {response}\n")
        sources = sorted({(h.metadata.get("source"), h.metadata.get("page_label")) for h in hits})
        if sources:
            print("Sources:")
            for src, pg in sources:
                print(f"  - {src}, page {pg}")
        print()


if __name__ == "__main__":
    main()