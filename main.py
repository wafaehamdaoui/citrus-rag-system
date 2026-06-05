import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Setup free local models
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOllama(model="llama3", temperature=0)

PERSIST_PATH = "./qdrant_citrus_db"
COLLECTION_NAME = "citrus-treatments"
DOCS_DIR = "./citrus_docs"

def main():
    # Create the directory if it doesn't exist yet
    Path(DOCS_DIR).mkdir(exist_ok=True)
    
    client = QdrantClient(path=PERSIST_PATH)

    try:
        # Check if the database already exists
        client.get_collection(collection_name=COLLECTION_NAME)
        vectorstore = QdrantVectorStore(
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
            client=client,
        )
        print("Loaded existing vector database successfully.")
    except Exception:
        client.close()
        print(f"Database not found. Scanning '{DOCS_DIR}' for PDF articles...")

        # 1. Load PDFs from the directory
        loader = PyPDFDirectoryLoader(DOCS_DIR)
        docs = loader.load()
        
        if not docs:
            print(f"\n[Warning]: No PDFs found in '{DOCS_DIR}'. Please drop your files there and restart.")
            return

        # 2. Split text into manageable chunks
        # We use a slightly smaller chunk size here because treatment protocols are precise
        script_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        all_chunks = script_splitter.split_documents(docs)
        print(f"Processed {len(docs)} PDF pages into {len(all_chunks)} text chunks.")

        # 3. Store into Qdrant vector database
        vectorstore = QdrantVectorStore.from_documents(
            all_chunks,
            embedding=embeddings,
            path=PERSIST_PATH,
            collection_name=COLLECTION_NAME,
        )

    # 4. Construct the retriever 
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # 5. Build an expert system prompt tailored for plant pathology
    template = """
    You are an expert AI Agronomist specializing in Citrus Pathology. 
    Analyze the question based ONLY on the provided research papers, extension reports, and official documentation guidelines.

    When recommending treatments (chemical, biological, or cultural control):
    1. Be highly precise regarding active ingredients, dosages, application timing, or sanitary measures if mentioned.
    2. Clearly state the specific citrus disease targeted.
    3. If the provided documents offer conflicting strategies or mention regional restrictions, note that distinction.
    4. If the context does not contain clear, verified treatment protocols for the query, state: 
       "The provided documentation does not detail a verified treatment protocol for this specific issue."

    Context Excerpts:
    {context}

    Agronomy Query: 
    {question}

    Expert Recommendation:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- The Citrus Disease Management System is Live ---")
    while True:
        query = input("\nAsk about a citrus disease or treatment: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = rag_chain.invoke(query)
        print(f"\nAdvisor Response:\n{response}")


if __name__ == "__main__":
    main()