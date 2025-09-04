import os
import chainlit as cl
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()

# Load FAISS index
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Structured response template
template = """
You are an assistant that explains LangChain concepts using both official docs and GitHub code.

Use the following retrieved context:
{context}

Always answer in this structured way:

1. **Explanation** - Clear and simple explanation of the concept.
2. **Use Case** - Why and when this is useful.
3. **Code Example** - Provide a working Python snippet if available.

Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# --------- Chainlit UI ---------
@cl.on_chat_start
async def start():
    await cl.Message(content="🤖 Hi! Ask me anything about LangChain.").send()

@cl.on_message
async def main(message: cl.Message):
    query = message.content
    response = qa_chain.run(query)

    await cl.Message(content=response).send()
