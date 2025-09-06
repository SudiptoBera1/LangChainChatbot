import os
from langchain_community.document_loaders import TextLoader, SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# Allowed file types
include_ext = (".md", ".py", ".ipynb")

# Path for FAISS index cache
INDEX_PATH = "faiss_index"

def safe_loader(path):
    """Return a TextLoader if file is supported, else None."""
    if path.endswith(include_ext):
        return TextLoader(path, encoding="utf-8", errors="ignore")
    return None

def load_repo_docs(repo_path="langchain_repo"):
    """Load documents from GitHub repo."""
    docs = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            path = os.path.join(root, file)
            loader = safe_loader(path)
            if loader:   # ✅ only load safe files
                try:
                    loaded = loader.load()
                    # Add repo metadata
                    for d in loaded:
                        d.metadata.update({
                            "source": path,        # full file path
                            "origin": "repo"
                        })
                    docs.extend(loaded)
                except Exception as e:
                    print(f"⚠️ Skipped {path} due to error: {e}")
    print(f"✅ Loaded {len(docs)} documents from repo")
    return docs

def load_website_docs():
    """Load documents from LangChain website sitemap."""
    sitemap_url = "https://www.langchain.com/sitemap.xml"
    loader = SitemapLoader(sitemap_url, custom_user_agent="my-rag-bot")
    docs = loader.load()

    # Add website metadata
    for d in docs:
        d.metadata.update({
            "source": d.metadata.get("source", "website"),
            "origin": "website"
        })
    print(f"✅ Loaded {len(docs)} documents from website")
    return docs

def build_or_load_index():
    """Either load FAISS index from disk or rebuild it."""
    embeddings = OpenAIEmbeddings()

    # ✅ Check cache
    if os.path.exists(INDEX_PATH):
        print("📦 Loading existing FAISS index from cache...")
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # ❌ If no cache, rebuild
    print("🔨 Building new FAISS index...")

    repo_docs = load_repo_docs()
    site_docs = load_website_docs()
    all_docs = repo_docs + site_docs

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)

    # Add chunk_id metadata
    for i, doc in enumerate(splits):
        doc.metadata["chunk_id"] = i

    print(f"✂️ Split into {len(splits)} chunks")

    # Build FAISS
    vectorstore = FAISS.from_documents(splits, embeddings)

    # Save cache
    vectorstore.save_local(INDEX_PATH)
    print("🎉 New FAISS index built and cached successfully!")

    return vectorstore


# ---------- Main ----------
if __name__ == "__main__":
    build_or_load_index()
