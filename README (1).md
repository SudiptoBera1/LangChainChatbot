# 🚀 RAG App with LangChain, FAISS, and Chainlit

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that combines **Large Language Models (LLMs)** with a **custom knowledge base**.  
Instead of relying only on the LLM’s internal training data, this app lets you load documents (e.g., GitHub repos, sitemaps, or text files), convert them into **embeddings**, and store them in a **FAISS vector database**.  

When a user asks a question, the system:
1. Retrieves the most relevant chunks from the indexed knowledge base.  
2. Passes them along with the query to an **OpenAI LLM** (via LangChain).  
3. Returns an accurate, context-aware answer in a **Chainlit chat UI**.  

---

## ✨ Features
- **Custom Knowledge Base** → Ingest content from GitHub repos, websites, or docs.  
- **Efficient Vector Search** → Powered by FAISS for fast similarity search.  
- **LLM Integration** → Uses OpenAI’s embeddings & GPT models.  
- **Interactive UI** → Clean chat interface built with Chainlit.  
- **Modular Design** → Separate scripts for index building (`build_index.py`) and querying (`app.py`).  

---

## 📂 Project Structure
```
your-rag-app/
│── app.py                # Main Chainlit app (chat UI)
│── build_index.py        # Builds FAISS index from repo/docs
│── requirements.txt      # Dependencies (pinned versions)
│── README.md             # Project setup & usage
│── .env.example          # Example env file (replace with your keys)
│── .gitignore            # Ignore unnecessary files
│
├── langchain_repo/       # (Optional) Docs/source to index
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
- Copy `.env.example` → `.env`
- Add your API keys inside `.env`, for example:
  ```bash
  OPENAI_API_KEY=your_openai_key_here
  ```

### 5. Build the FAISS Index
```bash
python build_index.py
```
This will create a local `faiss_index/` folder with embeddings.

### 6. Run the App
```bash
chainlit run app.py -w
```
Then open [http://localhost:8000](http://localhost:8000) in your browser 🚀

---

## 🛑 Notes
- Do **NOT** commit your real `.env` file.  
- `faiss_index/` is ignored (can be rebuilt anytime).  
- Tested with **Python 3.10+**.

---

## 📜 License
This project is licensed under the MIT License.
