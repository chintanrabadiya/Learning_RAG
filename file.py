# Assume imports for LangChain/ChromaDB/etc.

from langchain_community.vectorstores import Chroma
from langchain_classic.document_loaders import TextLoader
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# import PyMuPDFLoader

# --- Setup Phase (Runs once when documents are added) ---
def index_documents(file_paths):
    # 1. Load

    loader = TextLoader('data/my_document.txt')
    # loader = PyMuPDFLoader(file_paths)
    raw_documents = loader.load()
    
    # 2. Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(raw_documents)
    
    # 3, 4. Embed & Store
    # Using a HuggingFace embedding model (you can choose any compatible model)
    embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
    # The framework handles embedding and storing them in the vector store (ChromaDB)
    vector_store = Chroma.from_documents(chunks, embedding)
    print("Indexing complete. Knowledge base ready.")
    
# --- Query Phase (Runs every time the user asks a question) ---
def answer_question(user_query, vector_store):
    # 1. Retrieve (Find the context)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(user_query)
    
    # 2. Build Prompt & Context
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    
    # 3. LLM Call (The actual answer generation)
    prompt = f"""
    You are an expert assistant. Use ONLY the following context to answer the user's question. 
    Context: {context}
    Question: {user_query}
    Answer:
    """
    # Call the LLM API with the constructed prompt
    final_answer = llm_client.generate(prompt)
    return final_answer

