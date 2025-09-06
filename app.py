import os
import chainlit as cl
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
INDEX_PATH = "faiss_index"

# ------------------------------
# Helper functions
# ------------------------------
def load_vectorstore():
    """Load FAISS vectorstore from disk."""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

def create_qa_chain(vectorstore):
    """Create a ConversationalRetrievalChain with memory."""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Prompt template → ensures structured answers
    template = """
You are a helpful assistant that answers based on retrieved documents.

Always structure your response as:
1. **Answer** → short and clear
2. **Explanation** → why / how it works
3. **Use Case** → at least one real-world scenario
4. **Example with Code** → working code snippet with explanation

Chat history:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
    """
    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=template,
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# ------------------------------
# Chainlit event handlers
# ------------------------------
@cl.on_chat_start
async def start_chat():
    """Initialize the system when chat starts."""
    try:
        vectorstore = load_vectorstore()
        qa_chain = create_qa_chain(vectorstore)
        cl.user_session.set("qa_chain", qa_chain)

        await cl.Message(
            content="👋 Hi! I’m your RAG assistant.\n\n"
                    "Ask me anything about Langchain and I’ll try to reply:\n").send()

    except Exception as e:
        await cl.Message(
            content=f"❌ Startup error: {str(e)}\n\n"
                    "### Troubleshooting:\n"
                    "1. Check if FAISS index exists in `faiss_index/`\n"
                    "2. Ensure `.env` has your `OPENAI_API_KEY`\n"
                    "3. Install required libs: `pip install langchain chainlit faiss-cpu openai`\n"
        ).send()

@cl.on_message
async def process_message(message: cl.Message):
    """Handle user questions and return structured answers."""
    qa_chain = cl.user_session.get("qa_chain")

    try:
        result = await cl.make_async(qa_chain)(
            {"question": message.content}
        )
        answer = result["answer"]
        await cl.Message(content=answer).send()

    except Exception as e:
        await cl.Message(
            content=f"❌ Error: {str(e)}\n\n"
                    "### Troubleshooting:\n"
                    "1. Verify FAISS index and embeddings.\n"
                    "2. Check if `OPENAI_API_KEY` is valid.\n"
                    "3. Restart app with `chainlit run app.py -w`.\n"
        ).send()
