import nbformat as nbf
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Setup Notebook
nb = nbf.v4.new_notebook()

# Markdown: Title
text_cell_1 = nbf.v4.new_markdown_cell("""# Assignment 25: RAG & Chat History (Grand Final Submission)
**Student Name:** Patan Khalilulla Khan  
**Course:** GenAI Training Program  

---
## Assignment Overview
This notebook implements a production-ready conversational RAG pipeline using native LangChain components.
""")

# Code: Data Ingestion & Retrieval
code_cell_1 = nbf.v4.new_code_cell("""# Task 1 & 2: Ingestion & Indexing
loader = TextLoader('data/sample.txt')
docs = loader.load()
print(f"Loaded {len(docs)} document(s).")
print(f"Sample: {docs[0].page_content[:50]}...")

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

print("Vector store indexed.")
""")

# Code: Pipeline
code_cell_2 = nbf.v4.new_code_cell("""# Task 5 & 6: RAG Chain with History
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based ONLY on context: {context}. If unsure, say 'I don't know'."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Trimming History (Task 8)
def trim_history(history, max_msgs=4):
    return history[-max_msgs:]

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough(), "history": lambda x: trim_history(x["history"])})
    | prompt
    | llm
    | StrOutputParser()
)

rag_with_history = RunnableWithMessageHistory(
    chain,
    lambda x: ChatMessageHistory(),
    input_messages_key="question",
    history_messages_key="history",
)
""")

# Code: Task 9 - Evaluation
code_cell_3 = nbf.v4.new_code_cell("""# Task 9: 3-turn Evaluation
config = {"configurable": {"session_id": "test"}}
print("Q1:", rag_with_history.invoke({"question": "What is RAG?"}, config=config))
print("Q2 (Follow-up):", rag_with_history.invoke({"question": "How does it help?"}, config=config))
print("Q3 (Clarification):", rag_with_history.invoke({"question": "Can you clarify the last point?"}, config=config))
""")

nb['cells'] = [text_cell_1, code_cell_1, code_cell_2, code_cell_3]
nbf.write(nb, 'D:\\techtude\\GenAI-Task25-PatanKhalilullaKhan\\notebook\\GenAI_Assignment25_Prompting_Chains.ipynb')
print("Assignment 25 Final Notebook generated!")
