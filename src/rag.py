"""
rag.py
─────────────────────────────────────────────────────────────────────────────
This module implements the core RAG (Retrieval-Augmented Generation) pipeline:
1. Converts the user's natural language question into a semantic vector.
2. Connects to our local persistent ChromaDB to perform a vector search.
3. Retrieves the top-K matching document chunks (with original metadata).
4. Formulates a system prompt injecting the retrieved context chunks.
5. Sends the combined prompt to Google Gemini via LangChain for synthesis.
6. Returns the human-like summarized answer along with original source citations.
"""

import os
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. LOAD CONFIGURATION
# Load environment variables (API keys) from the root .env file.
# Resolving absolute path ensures it loads correctly regardless of which directory python executes from.
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
load_dotenv(dotenv_path=dotenv_path)

# Paths and variables matching our ingestion pipelines
CHROMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chroma_db"))
COLLECTION_NAME = "industrial_knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"  # Local embedding model generating 384-dimensional vectors

# Local sentence transformer instance (loaded lazily on the first function call)
_model = None

def get_embedding_model():
    """Lazily instantiates and caches the SentenceTransformer model to save memory."""
    global _model
    if _model is None:
        # SentenceTransformer loads the pre-trained all-MiniLM-L6-v2 model
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def retrieve_context(query: str, top_k: int = 5):
    """
    Step 2.1: Retrieval Component
    - Vectorizes the user query.
    - Connects to ChromaDB.
    - Searches for chunks that are closest in the multi-dimensional vector space.
    """
    # Initialize a local persistent Chroma DB client pointing to our database folder
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Check if the database collection has been populated
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in existing:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found in ChromaDB at {CHROMA_DIR}.\n"
            "Please run 'python src/embed.py' first to populate vectors."
        )
        
    collection = client.get_collection(COLLECTION_NAME)
    
    # Generate query vector: converts user text into a 384-element list of float numbers
    model = get_embedding_model()
    query_embedding = model.encode([query])[0].tolist()
    
    # Perform similarity search query in ChromaDB.
    # We retrieve the matching documents, their metadata tags (like section, tag, file), and cosine distance.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format and repackage retrieved results for readability
    retrieved_docs = []
    if results and "documents" in results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # Chroma returns cosine distance. Similarity score = 1.0 - cosine distance.
            retrieved_docs.append({
                "text": doc,
                "metadata": meta,
                "score": round(1.0 - dist, 4)
            })
            
    return retrieved_docs

def generate_answer(query: str, top_k: int = 5):
    """
    Step 2.2: Augmentation and Generation Component
    - Fetches the vector search context.
    - Injects it into the system instructions template.
    - Calls Gemini model to generate a professional cited response.
    """
    # 1. Fetch relevant context document chunks from ChromaDB
    docs = retrieve_context(query, top_k=top_k)
    
    # 2. Format the retrieved context into a single clean string block
    # This structure allows the LLM to separate and distinguish each file's source context.
    context_str = ""
    for i, doc in enumerate(docs, 1):
        meta = doc["metadata"]
        context_str += f"--- Document {i} (Source: {meta.get('source_file')}, Section: {meta.get('section', 'General')}) ---\n"
        context_str += f"{doc['text']}\n\n"
        
    # 3. Instantiate Google Gemini model via LangChain
    # 'gemini-2.5-flash' is chosen for its fast speed and high accuracy.
    # temperature=0 ensures responses are deterministic (consistent and factual based on context).
    chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 4. Construct Prompt Template
    # We instruct the model to stick strictly to the context and cite its sources.
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert assistant specialized in industrial plants and operations.\n"
            "Use the provided document context below to answer the user's question. If the context "
            "doesn't contain the answer, say you don't know based on the documents.\n\n"
            "=== Retrieved Document Context ===\n"
            "{context}\n"
            "==================================\n\n"
            "Generate a clear, professional, and accurate response. Always cite the source files "
            "and sections where you found the information."
        )),
        ("human", "{question}")
    ])
    
    # 5. Compose LangChain Execution Chain: Prompt -> ChatModel -> Output Parser
    # Prompt fills template values, ChatModel runs generation, StrOutputParser converts output to plain text.
    chain = prompt | chat | StrOutputParser()
    
    # Execute the chain
    answer = chain.invoke({
        "context": context_str,
        "question": query
    })
    
    return {
        "answer": answer,
        "sources": docs
    }

# 6. LOCAL TERMINAL EXECUTION (For testing and standalone runs)
if __name__ == "__main__":
    import sys
    # Configure UTF-8 stdout encoding on Windows to prevent Unicode console printing issues
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
        
    # Accept user query from command line parameters, or use a default test query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What corrective actions were taken after the dry running incident?"
        
    print(f"Query: {query}\n")
    print("Retrieving context and generating answer...")
    try:
        # Run generation
        res = generate_answer(query)
        
        # Print results
        print("\n=== Answer ===")
        print(res["answer"])
        
        print("\n=== Sources Cited ===")
        for idx, src in enumerate(res["sources"], 1):
            meta = src["metadata"]
            print(f"[{idx}] {meta.get('source_file')} | Section: {meta.get('section')} (Similarity: {src['score']:.4f})")
            
    except Exception as e:
        print(f"Error during execution: {e}")
