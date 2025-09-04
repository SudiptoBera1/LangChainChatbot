import os
from dotenv import load_dotenv
from langchain_community.document_loaders import SitemapLoader, GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load API key
load_dotenv()

# --------- Load LangChain Docs from Sitemap ---------
print("📘 Loading docs from sitemap...")
sitemap_loader = SitemapLoader("https://www.langchain.com/sitemap.xml")
docs = sitemap_loader.load()

# --------- Load LangChain GitHub Repo (Selective) ---------
print("🐙 Cloning GitHub repo...")
git_loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain.git",
    repo_path="./langchain_repo",   # local path where repo will be cloned
    branch="master",
    file_filter=lambda file_path: (
        file_path.endswith(".py") or file_path.endswith(".md")
    )
)
repo_docs = git_loader.load()

# --------- Combine Both ---------
all_docs = docs + repo_docs
print(f"✅ Loaded {len(all_docs)} documents")

# --------- Split Text ---------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)
print(f"📑 Split into {len(splits)} chunks")

# --------- Create Vector Store ---------
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(splits, embeddings)

# Save Index
db.save_local("faiss_index")
print("🎉 FAISS index built and saved successfully with Docs + GitHub!")
